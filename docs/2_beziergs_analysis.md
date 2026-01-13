# BezierGS 项目深度分析与对比

## 一、BezierGS 项目概述

**BezierGS** (ICCV 2025) - 基于Bézier曲线的高斯溅射动态场景重建方法，专门针对自动驾驶场景设计。

### 核心创新
- **参数化运动表示**: 使用Bézier曲线参数化动态物体轨迹
- **高效内存利用**: O(N×P)复杂度，P为控制点数（通常4个）
- **时间连续性**: 连续的曲线表示保证帧间平滑过渡

### 项目结构

```
BezierGS/
├── scene/                        # 核心场景模块
│   ├── bz_gaussian_model.py      # Bézier高斯模型（核心）
│   ├── pvg_gaussian_model.py     # PVG正弦模型
│   ├── deform_model.py           # 可变形模型（DeformableGS）
│   ├── nuplan_loader.py          # nuPlan数据加载器
│   ├── waymo_loader.py           # Waymo数据加载器
│   ├── kittimot_loader.py        # KITTI-MOT数据加载器
│   └── drivex_waymo_loader.py    # DriveX-Waymo数据加载器
├── gaussian_renderer/            # 渲染器
├── configs/                      # 配置文件
│   ├── base.yaml                 # 基础配置
│   ├── waymo/                    # Waymo场景配置
│   └── nuplan/                   # nuPlan场景配置
├── script/                       # 数据处理脚本
└── train.py                      # 训练入口
```

---

## 二、核心模块详解

### 1. BézierGaussianModel 类

**文件**: `scene/bz_gaussian_model.py`

#### 核心数据结构

```python
class GaussianModel:
    def __init__(self, args):
        # Bézier曲线参数
        self.num_control_points = args.order + 1  # 控制点数量，order=3时为4个
        self.binomial_coefs = torch.tensor([
            math.comb(self.num_control_points - 1, k)
            for k in range(self.num_control_points)
        ])  # 二项式系数: [1, 3, 3, 1] for cubic

        # 动态模式选择
        self.dynamic_mode = args.dynamic_mode
        # "Bezier": Bézier曲线参数化
        # "PVG": 正弦波表示（Periodic Volumetric Gaussian）
        # "DeformableGS": MLP变形网络

        # 核心张量
        self._control_points = torch.empty(0)  # (N, num_cp, 3) 控制点
        self._group = torch.empty(0)            # (N,) 分组ID
        self.trajectory_cp_tensor = None        # (obj_num, num_cp, 3) 轨迹控制点
```

#### 分组机制

BezierGS使用**分组(group)**机制区分不同物体：

```python
# group = 0: 静态背景
# group = 1, 2, ...: 动态物体（车辆、行人等）

# 示例：
self._group = torch.tensor([
    0, 0, 0, ...,    # 背景点
    1, 1, 1, ...,    # 第1个车辆
    2, 2, 2, ...,    # 第2个车辆
])
```

### 2. Bézier曲线实现

#### Bézier系数计算

```python
def get_bezier_coefficient(self):
    """
    计算Bézier基函数系数
    输入: ts - (T, 1) 时间参数
    输出: M - (T, num_cp) 基函数矩阵

    Bézier曲线定义: B(t) = Σ C(n,k) * t^k * (1-t)^(n-k) * P_k
    """
    n = self.num_control_points - 1  # 3 for cubic
    ks = torch.arange(self.num_control_points, dtype=torch.float32, device="cuda")

    def bezier_coeff(ts: torch.Tensor):
        t_pow_k = ts ** ks                           # (T, num_cp)
        one_minus_t_pow_n_minus_k = (1.0 - ts) ** (n - ks)  # (T, num_cp)
        M = self.binomial_coefs * t_pow_k * one_minus_t_pow_n_minus_k
        return M

    return bezier_coeff

# 使用示例：
# ts = torch.tensor([[0.0], [0.5], [1.0]])  # 3个时间点
# M = bezier_coeff(ts)  # (3, 4) - cubic Bézier有4个基函数
```

#### 坐标预测函数

```python
def predict_xyz(self, t, group_idx):
    """
    预测group_idx在时间t时的3D坐标偏移

    Args:
        t: 归一化时间参数 ∈ [0, 1]
        group_idx: 物体分组ID

    Returns:
        xyz: (3,) 3D坐标
    """
    M = self.BezierCoeff(t).unsqueeze(1)  # (1, 1, num_cp)
    control_points = self._control_points[self._group == group_idx]  # (N_i, num_cp, 3)

    # Bézier插值
    xyz = torch.matmul(M, control_points).squeeze(1)  # (1, 3)
    return xyz

def predict_xyz_derivative(self, t, group_idx):
    """
    计算速度（一阶导数）
    用于运动一致性损失
    """
    d_M = self.BezierDerivativeCoeff(t)
    return torch.matmul(d_M, self._control_points[self._group == group_idx]).squeeze(1)
```

### 3. 动态渲染核心

#### `get_xyz_with_offset` 方法

这是BezierGS最核心的方法，负责计算任意时刻的高斯位置：

```python
def get_xyz_with_offset(self, world_time):
    """
    计算world_time时刻所有高斯点的3D位置

    工作流程：
    1. 静态背景（group=0）：直接使用控制点
    2. 动态物体（group>0）：
       a. 计算Bézier时间参数
       b. 计算轨迹中心（全局运动）
       c. 计算物体偏移（局部运动）
       d. 合成最终位置
    """
    if self.dynamic_mode == "Bezier":
        valid_mask = torch.zeros(self._control_points.shape[0],
                                dtype=torch.bool, device="cuda")

        # 静态背景
        xyz = self._control_points[:, 0, :].clone()
        xyz_offset = torch.zeros_like(xyz)
        valid_mask[self.get_group == 0] = True

        # 处理每个动态物体
        for idx in range(1, self.group_num + 1):
            # 检查物体是否存在于当前时间
            if world_time < self.exist_range[idx][0] or \
               world_time > self.exist_range[idx][1]:
                continue

            # 1. 计算Bézier时间参数
            bezier_param_cp = self.cp_recorded_timestamp_2_bezier_t[idx-1]
            bezier_param = self.BezierCoeff(
                torch.tensor([[world_time]]).cuda()
            ) @ bezier_param_cp

            # 2. 计算轨迹中心（全局运动）
            traj_cp = self.trajectory_cp_tensor[idx-1]  # (num_cp, 3)
            M = self.BezierCoeff(bezier_param)
            traj_central = torch.matmul(M, traj_cp).squeeze(0)  # (3,)

            # 3. 计算物体偏移（局部运动/形变）
            xyz_offset[self.get_group == idx] = \
                self.predict_xyz(bezier_param, idx).squeeze(0)

            # 4. 合成最终位置
            xyz[self.get_group == idx] = \
                xyz_offset[self.get_group == idx] + traj_central
            valid_mask[self.get_group == idx] = True

    return xyz, xyz_offset, valid_mask
```

**关键设计思想**：
- **分离运动**: 将全局运动（轨迹）和局部运动（形变）分离
- **参数化时间**: 世界时间 → Bézier参数 → 插值位置
- **高效查询**: O(1)时间复杂度查询任意时刻的位置

### 4. 轨迹拟合与初始化

```python
def create_from_ply_dict(self, ply_dict, spatial_lr_scale):
    """
    从PLY字典初始化高斯模型

    对于动态物体：
    1. 获取轨迹点序列（从标注数据）
    2. 使用弦长参数化（chord length parametrization）
    3. 迭代优化Bézier控制点
    """
    for k, v in ply_dict.items():
        if k == 'bkgd':
            # 背景初始化（略）
        else:
            # 动态物体初始化
            trajectory = torch.from_numpy(v['trajectory']).float().cuda()

            # 弦长参数化
            t_parameterized = self.get_chord_len_parametrization(trajectory)

            # 生成初始控制点
            trajectory_cp = self.generate_control_points(
                trajectory.unsqueeze(0),
                t_parameterized.unsqueeze(1)
            ).squeeze(0)

            # 迭代优化控制点（使曲线更好拟合轨迹点）
            for round in range(100):
                t_samples = torch.linspace(0, 1, 10000, ...)
                Bernstein_mat = self.BezierCoeff(t_samples.unsqueeze(1))
                curve_samples = Bernstein_mat @ trajectory_cp

                # 找到每个轨迹点最近的曲线点
                diff = trajectory.unsqueeze(1) - curve_samples.unsqueeze(0)
                indices = torch.argmin(torch.sum(diff**2, dim=-1), dim=-1)
                bezier_t = t_samples[indices]

                # 重新拟合
                new_cp = self.generate_control_points(
                    trajectory.unsqueeze(0),
                    bezier_t.unsqueeze(1)
                ).squeeze(0)

                # 检查收敛
                if torch.norm(new_cp - trajectory_cp, dim=-1).max() < 1e-6:
                    break
                trajectory_cp = new_cp
```

---

## 三、与MTGS的架构对比

### 3.1 整体架构对比

| 方面 | MTGS | BezierGS |
|------|------|----------|
| **设计理念** | 多遍历重建，模块化子模型 | 参数化轨迹，统一建模 |
| **动态表示** | 离散姿态优化 | 连续Bézier曲线 |
| **模块组织** | 分离的RigidSubModel、DeformableSubModel | 统一的GaussianModel + 分组机制 |
| **基类** | VanillaGaussianSplattingModel | GaussianModel（单类） |
| **配置** | 多Config组合（背景、天空、刚体等） | 单一Config + 动态模式选择 |

### 3.2 动态物体处理对比

#### MTGS方法

```python
# MTGS: 分离的子模型
class MTGSSceneModel:
    def __init__(self):
        self.gaussian_models = torch.nn.ModuleDict()
        # 每个物体是一个独立的模型实例
        self.gaussian_models['vehicle_001'] = RigidSubModel(...)
        self.gaussian_models['vehicle_002'] = RigidSubModel(...)
        self.gaussian_models['pedestrian_001'] = DeformableSubModel(...)

# 每个子模型维护独立的姿态
class RigidSubModel:
    def __init__(self):
        self.instance_quats = Parameter(...)  # (num_frames, 4)
        self.instance_trans = Parameter(...)  # (num_frames, 3)

# 渲染时需要查询每个子模型
def get_outputs(self, camera, frame_idx):
    for model_name, model in self.gaussian_models.items():
        gaussians = model.get_gaussians(camera, frame_idx)
        # 渲染...
```

**特点**:
- ✅ 模块化清晰，易于扩展
- ✅ 每个物体独立优化，灵活性高
- ❌ 内存占用大（O(N×T)）
- ❌ 帧间不连续（离散姿态）

#### BezierGS方法

```python
# BezierGS: 统一模型 + 分组
class GaussianModel:
    def __init__(self):
        self._control_points = Parameter(...)  # (N, num_cp, 3)
        self._group = torch.tensor(...)        # (N,)

        # 轨迹控制点（每个物体一组）
        self.trajectory_cp_tensor = [...]      # (obj_num, num_cp, 3)

# 渲染时统一计算
def get_xyz_with_offset(self, world_time):
    xyz = self._control_points[:, 0, :].clone()

    # 一次性处理所有物体
    for idx in range(1, self.group_num + 1):
        mask = self.get_group == idx
        xyz[mask] = self.compute_bezier_position(idx, world_time)

    return xyz
```

**特点**:
- ✅ 内存高效（O(N×P), P≪T）
- ✅ 时间连续（平滑曲线）
- ✅ 统一渲染，批处理友好
- ❌ 灵活性稍低（共享运动模型）

### 3.3 内存复杂度对比

假设场景有：
- N个动态物体
- 每个物体平均M个高斯点
- T个时间帧

#### MTGS
```
内存 = Σ (物体i的高斯点数 × 帧数)
     = N × M × T × sizeof(Pose)

对于 N=100, M=1000, T=200:
内存 = 100 × 1000 × 200 × (4×4 + 3×4) bytes
     ≈ 560 MB (仅姿态参数)
```

#### BezierGS
```
内存 = Σ (物体i的高斯点数 × 控制点数) + 轨迹控制点
     = N × M × P + N × P × sizeof(Point)

对于 N=100, M=1000, P=4:
内存 = 100 × 1000 × 4 × 3×4 + 100 × 4 × 3×4 bytes
     ≈ 19.2 MB + 4.8 KB
```

**内存节省**: ~30倍

### 3.4 时间连续性对比

#### MTGS
```python
# 离散姿态，帧间可能跳跃
frame_100_pose = (quat[100], trans[100])
frame_101_pose = (quat[101], trans[101])

# 需要显式插值
interpolated_pose = interpolate(frame_100_pose, frame_101_pose, t=0.5)
```

#### BezierGS
```python
# 连续曲线，任意时刻可查询
pose_at_any_time = bezier_curve(evaluate_at=t)

# 自然平滑，一阶导数连续
velocity = bezier_derivative(t)
```

### 3.5 多遍历支持对比

#### MTGS
```python
# 专门的多遍历支持
class MultiColorGaussianSplattingModel:
    def __init__(self):
        # 每个遍历独立的颜色特征
        self.features_adapters = Parameter(...)  # (N, num_traversals, 3)

    def get_features(self, traversal_id):
        # 根据遍历ID选择颜色
        return self.features_dc + self.features_adapters[:, traversal_id, :]
```

**优势**: 专门设计用于多遍历场景
**劣势**: 颜色特征随遍历数线性增长

#### BezierGS
```python
# 时序颜色特征（4D Gaussian）
def get_features4d(self, time):
    features = []
    for idx in range(self.group_num + 1):
        # 归一化时间
        norm_time = (time - self.exist_range[idx][0]) / \
                    (self.exist_range[idx][1] - self.exist_range[idx][0])

        # Fourier基函数
        idft_base = IDFT(norm_time, self.fourier_dim)

        # 时变颜色
        feature = torch.sum(self._features_dc * idft_base, dim=1)
        features.append(feature)

    return torch.cat(features, dim=0)
```

**优势**: 连续时间表示，支持任意时间查询
**劣势**: 频域表示可能难以捕捉突变

---

## 四、动态模式对比

### MTGS的动态处理

| 物体类型 | 模型 | 运动表示 | 适用场景 |
|---------|------|---------|---------|
| 车辆 | RigidSubModel | 6DoF姿态优化 | 刚体运动 |
| 行人/自行车 | DeformableSubModel | MLP变形网络 | 非刚体形变 |
| 背景 | MultiColorGaussianSplatting | 静态 + 颜色适配 | 多遍历颜色 |

### BezierGS的动态模式

```python
self.dynamic_mode = args.dynamic_mode

# 1. Bezier模式（默认）
# - 参数化轨迹
# - 适用于: 车辆、规则的移动物体
if self.dynamic_mode == "Bezier":
    xyz = self.get_xyz_with_offset(world_time)

# 2. PVG模式（Periodic Volumetric Gaussian）
# - 正弦波表示
# - 适用于: 周期性运动（如行人摆臂）
elif self.dynamic_mode == "PVG":
    xyz = self.pvg_motion(world_time)

# 3. DeformableGS模式
# - MLP变形网络
# - 适用于: 复杂非刚体变形
elif self.dynamic_mode == "DeformableGS":
    d_xyz = self.deform.step(xyz, world_time)
    xyz = xyz + d_xyz
```

**对比**:
- MTGS: 根据物体类型自动选择模型（硬编码）
- BezierGS: 全局统一模式，但可配置切换

---

## 五、训练流程对比

### MTGS训练流程

```python
# 1. 数据加载
datamanager = CustomFullImageDatamanagerConfig(
    dataparser=NuplanDataParserConfig(...)
)

# 2. 场景初始化
model = MTGSSceneModel(
    model_config={
        'background': MultiColorGaussianSplattingModelConfig(...),
        'rigid_object': RigidSubModelConfig(...),
        'deformable_node': DeformableSubModelConfig(...),
    }
)

# 3. 多优化器
optimizers = {
    "means": Adam(lr=8e-4),
    "ins_rotation": Adam(lr=1e-5),
    "ins_translation": Adam(lr=5e-4),
    # ...
}

# 4. 训练循环
for iteration in range(30000):
    # 获取相机和帧索引
    camera = datamanager.next_camera()
    frame_idx = camera.frame_idx

    # 渲染（多子模型融合）
    outputs = model.get_outputs(camera, frame_idx)

    # 损失计算
    loss = L1 + SSIM + depth + ...

    # 反向传播（各子模型独立）
    loss.backward()
    for opt in optimizers.values():
        opt.step()
```

### BezierGS训练流程

```python
# 1. 数据加载
scene = Scene(args, gaussians)  # 支持多种数据集

# 2. 模型初始化
gaussians = GaussianModel(args)
gaussians.create_from_ply_dict(ply_dict)

# 3. 单一优化器
gaussians.training_setup(args)
# 内部参数组：
# - control_points
# - features_dc/rest
# - trajectory_cp_tensor（轨迹控制点）

# 4. 训练循环
for iteration in range(30000):
    # 获取相机和时间
    viewpoint_cam = scene.getRandomCamera()
    world_time = viewpoint_cam.time

    # 渲染（统一计算）
    xyz, _, valid_mask = gaussians.get_xyz_with_offset(world_time)
    render_pkg = render(viewpoint_cam, gaussians, pipe, xyz)

    # 损失计算（包含速度损失）
    loss = L1_loss + lambda_velocity * velocity_loss + ...

    # 反向传播（统一优化器）
    loss.backward()
    gaussians.optimizer.step()
```

**关键差异**:
- **优化器**: MTGS多优化器 vs BezierGS单一优化器
- **损失**: BezierGS有独特的速度一致性损失
- **时间**: MTGS用frame_idx，BezierGS用连续world_time

---

## 六、配置文件对比

### MTGS配置 (`mtgs/config/MTGS.py`)

```python
config = CustomTrainerConfig(
    pipeline=MultiTravelEvalPipielineConfig(
        datamanager=CustomFullImageDatamanagerConfig(
            dataparser=NuplanDataParserConfig(...)
        ),
        model=MTGSSceneModelConfig(
            control=GaussianSplattingControlConfig(
                densify_from_iter=500,
                cull_alpha_thresh=0.005,
                sh_degree=3,
            ),
            # 模块化配置
            model_config=dict(
                background=MultiColorGaussianSplattingModelConfig(...),
                skybox=SkyboxGaussianSplattingModelConfig(...),
                rigid_object=RigidSubModelConfig(...),
                deformable_node=DeformableSubModelConfig(...),
            ),
        ),
    ),
    # 多优化器配置
    optimizers={
        "means": {...},
        "features_dc": {...},
        "ins_rotation": {...},
        "ins_translation": {...},
        # ...
    },
)
```

### BezierGS配置 (`configs/waymo/017.yaml`)

```yaml
# 场景配置
scene_type: "Waymo"
start_time: 61
end_time: 109

# Bézier参数
order: 3  # 3阶Bézier曲线（4个控制点）

# 动态模式
dynamic_mode: "Bezier"
render_type: "bezier"

# 优化参数
iterations: 30000
densify_until_iter: 15000
lambda_velocity: 1.0  # 速度损失权重

# 数据加载
load_sky_mask: true
load_dynamic_mask: true
load_bbox_mask: true
```

**对比**:
- MTGS: Python配置类，复杂但灵活
- BezierGS: YAML配置，简洁直观
- 两者都支持数据集特定配置

---

## 七、优劣分析与适用场景

### MTGS优势

1. **模块化设计**: 易于添加新的物体类型或处理方法
2. **多遍历专用**: 专门针对多遍历场景优化
3. **灵活性高**: 每个物体独立配置和优化
4. **扩展性强**: 继承机制便于自定义

**适用场景**:
- 多遍历城市重建
- 需要精细控制每个物体的场景
- 研究和原型开发
- 需要处理复杂物体交互

### BezierGS优势

1. **内存高效**: 参数化表示大幅减少内存占用
2. **时间连续**: 平滑的轨迹表示
3. **推理快速**: O(1)时间查询任意时刻
4. **简洁优雅**: 统一的建模框架

**适用场景**:
- 长序列动态场景
- 资源受限环境
- 需要时序插值的场景
- 生产环境部署

---

## 八、如何在MTGS中借鉴BezierGS的思想

### 方案1: 添加Bézier运动模型作为新的子模型

```python
# 创建新模块: mtgs/scene_model/gaussian_model/bezier_node.py

from mtgs.scene_model.gaussian_model.vanilla_gaussian_splatting import (
    VanillaGaussianSplattingModel,
    VanillaGaussianSplattingModelConfig
)

@dataclass
class BezierSubModelConfig(VanillaGaussianSplattingModelConfig):
    _target: Type = field(default_factory=lambda: BezierSubModel)

    # Bézier参数
    bezier_order: int = 3  # 3阶Bézier
    use_velocity_loss: bool = True
    velocity_loss_weight: float = 1.0

class BezierSubModel(VanillaGaussianSplattingModel):
    """使用Bézier曲线的动态物体模型"""

    def populate_modules(self, instance_dict, data_frame_dict, **kwargs):
        # 初始化基础高斯
        super().populate_modules(points_3d=points_dict)

        # Bézier控制点
        num_control_points = self.config.bezier_order + 1
        trajectory = instance_dict["trajectory"]  # (T, 3)

        # 拟合Bézier曲线
        self.trajectory_cp = self._fit_bezier_curve(trajectory)

        # 局部偏移控制点
        self.offset_cp = Parameter(
            torch.zeros(self.get_xyz.shape[0], num_control_points, 3)
        )

    def _fit_bezier_curve(self, trajectory):
        """拟合轨迹到Bézier曲线"""
        # 使用最小二乘法拟合
        # 返回 (num_cp, 3) 控制点
        pass

    def get_means(self, camera, frame_idx):
        """计算Bézier插值位置"""
        world_time = camera.timestamps[frame_idx]
        bezier_t = self._world_time_to_bezier_t(world_time)

        # 轨迹中心
        traj_center = self._evaluate_bezier(
            self.trajectory_cp, bezier_t
        )

        # 局部偏移
        offset = self._evaluate_bezier(
            self.offset_cp, bezier_t
        )

        return self.get_xyz + offset + traj_center
```

### 方案2: 在MTGS中集成BezierGS的轨迹拟合

```python
# 在 mtgs/tools/ 中添加工具脚本

def fit_bezier_trajectories(
    instance_trajectories: Dict[str, torch.Tensor],
    order: int = 3
) -> Dict[str, torch.Tensor]:
    """
    将MTGS的离散姿态拟合为Bézier曲线

    Args:
        instance_trajectories: {token: (T, 3) trajectory}
        order: Bézier曲线阶数

    Returns:
        {token: (order+1, 3) control_points}
    """
    control_points_dict = {}

    for token, trajectory in instance_trajectories.items():
        # 弦长参数化
        t = _chord_length_parametrization(trajectory)

        # 最小二乘拟合
        cp = _least_squares_bezier_fit(trajectory, t, order)
        control_points_dict[token] = cp

    return control_points_dict
```

### 方案3: 混合架构

```python
# 在 MTGSSceneModel 中同时支持两种表示

class MTGSSceneModel:
    def __init__(self, config):
        self.use_bezier_for_vehicles = config.use_bezier_for_vehicles

    def _init_gaussian_models(self):
        for config_name, model_config in self.config.model_config.items():
            if config_name == 'rigid_object':
                for instance_token, instance_dict in instances_info.items():
                    # 根据配置选择模型
                    if self.use_bezier_for_vehicles:
                        model = BezierSubModel(...)
                    else:
                        model = RigidSubModel(...)

                    self.gaussian_models[instance_token] = model
```

---

## 九、实施建议

### 如果要在MTGS中集成BezierGS：

1. **渐进式集成**
   - 先实现独立的BezierSubModel
   - 在小规模场景验证
   - 逐步替换现有RigidSubModel

2. **保持兼容性**
   - 保留现有的离散姿态接口
   - 添加配置选项切换模式
   - 确保数据格式兼容

3. **性能优化**
   - 批量Bézier计算
   - CUDA kernel优化
   - 缓存Bézier系数矩阵

4. **评估指标**
   - 内存占用对比
   - 渲染质量（PSNR/SSIM）
   - 时序平滑性（速度一致性）
   - 训练时间

---

## 十、总结

### 核心差异总结

| 维度 | MTGS | BezierGS |
|------|------|----------|
| **设计哲学** | 模块化、分离式 | 统一、参数化 |
| **动态表示** | 离散姿态 | Bézier曲线 |
| **内存效率** | 中等 | 高 |
| **时间连续性** | 需插值 | 天然连续 |
| **扩展性** | 高（继承） | 中（配置） |
| **适用场景** | 多遍历、复杂交互 | 长序列、资源受限 |

### 选择建议

**选择MTGS如果**:
- 需要处理多遍历场景
- 物体类型多样，需要不同处理方式
- 研究新方法，需要高灵活性
- 场景复杂，物体交互多

**选择BezierGS如果**:
- 时间序列长，内存受限
- 需要时序插值和连续查询
- 追求简洁和效率
- 场景以车辆等规则运动为主

**混合使用**:
- 背景/静态物：MTGS的MultiColorGaussianSplatting
- 车辆：BezierGS的Bézier曲线
- 行人：MTGS的DeformableSubModel
- 天空：MTGS的SkyboxGaussianSplatting

这种组合可以发挥两者的优势，实现最优的动态场景重建效果。

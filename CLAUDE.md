1. 使用中文回答，可以使用英文思考
2. 这是一个用于自动驾驶高保真场景生成的多遍历3dgs模型
3. Always use context7 when I need code generation, setup or configuration steps, or library/API documentation.
## 项目概述

**MTGS (Multi-Traversal Gaussian Splatting)** - 基于高斯溅射的3D场景重建项目，专为自动驾驶设计

### 核心特性
- 多次遍历场景重建（利用同一场景的多次通过数据）
- nuPlan数据集集成
- Web可视化查看器（支持多次遍历和时间切片）
- 动态/静态元素分离处理
- 高质量渲染（60Hz姿态插值）
- 深度和法向量监督
- 多GPU训练支持

### 项目结构
```
MTGS/
├── mtgs/                      # 主包
│   ├── config/               # 配置文件
│   ├── dataset/              # 数据加载
│   ├── scene_model/          # 场景模型
│   ├── custom_viewer/        # Web查看器
│   ├── tools/                # 工具脚本
│   └── utils/                # 辅助函数
├── nuplan_scripts/           # nuPlan脚本
├── thirdparty/               # 第三方依赖
│   ├── UniDepth/             # 深度估计
│   ├── kiss-icp/             # 点云处理
│   └── lama/                 # 图像修复
└── docs/                     # 文档
```

### 主要组件
- **MTGSSceneModel** - 主场景模型
- **VanillaGaussianSplatting** - 基础高斯溅射
- **RigidSubModel** - 刚性物体（车辆、建筑）
- **MultiColorGaussianSplatting** - 多颜色背景
- **NuplanDataParser** - 数据集解析器
- **CustomDataManager** - 异步数据加载

### 主要入口
```bash
# 训练
ns-train mtgs [options] nuplan [dataset_config]

# 开发模式
source dev.sh
mtgs_setup mtgs/config/MTGS.py
mtgs_train [args]

# 查看器
python mtgs/tools/run_viewer.py

# 渲染
python mtgs/tools/render.py
```

### 配置文件
- [MTGS.py](mtgs/config/MTGS.py) - 完整多次遍历重建（车辆+背景）
- [MTGS_static.py](mtgs/config/MTGS_static.py) - 仅静态重建
- [3DGS.py](mtgs/config/3DGS.py) - 基础3D高斯溅射基线

### 核心依赖
- nerfstudio==1.1.5
- gsplat
- torch
- UniDepth（深度估计）
- kiss-icp（点云配准）
- lama（图像修复）
#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Hybrid Integration with BezierGS                                                 #
#-------------------------------------------------------------------------------#
"""
BezierRigidSubModel: 结合MTGS和BezierGS的优势

核心思想：
1. 使用Bézier曲线参数化车辆轨迹（节省内存、保证连续性）
2. 保留MTGS的模块化设计和多遍历支持
3. 统一的接口与现有MTGS架构兼容

相比传统RigidSubModel的优势：
- 内存效率：O(N×P) vs O(N×T)，P=4控制点，T=帧数
- 时间连续性：Bézier曲线保证平滑
- 可微性：支持端到端训练
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type, Union, Any, Literal
import warnings
import math

import numpy as np
import torch

from torch import Tensor
from torch.nn import Parameter

try:
    from gsplat.cuda._wrapper import spherical_harmonics
except ImportError:
    print("Please install gsplat>=1.0.0")

from nerfstudio.utils.rich_utils import CONSOLE

from .vanilla_gaussian_splatting import VanillaGaussianSplattingModel, VanillaGaussianSplattingModelConfig
from .rigid_node import RigidSubModel


@dataclass
class BezierRigidSubModelConfig(VanillaGaussianSplattingModelConfig):
    """Bézier刚体模型配置"""

    _target: Type = field(default_factory=lambda: BezierRigidSubModel)

    # Bézier曲线参数
    bezier_order: int = 3
    """Bézier曲线阶数，默认3阶（4个控制点）"""

    use_velocity_loss: bool = True
    """是否使用速度一致性损失"""

    velocity_loss_weight: float = 0.1
    """速度损失权重"""

    use_trajectory_fitting: bool = True
    """是否在初始化时拟合轨迹到Bézier曲线"""

    trajectory_fitting_iterations: int = 100
    """轨迹拟合迭代次数"""

    # 兼容性参数（与RigidSubModel保持一致）
    fourier_features_dim: Optional[int] = None
    fourier_features_scale: Optional[int] = 1
    is_static: Optional[bool] = False
    fourier_in_space: Optional[Literal['spatial', 'temporal']] = 'temporal'
    ground_filter_height: float = 0.0
    """remove points within this height above the lowest z; set to 0 to disable."""
    min_points_after_filter: int = 50
    """minimum points required after filtering to accept the crop."""


class BezierRigidSubModel(VanillaGaussianSplattingModel):
    """
    使用Bézier曲线的刚体动态物体模型

    适用于：车辆、机器人等刚体运动

    核心创新：
    - 轨迹中心：使用Bézier曲线表示全局运动
    - 局部偏移：使用Bézier曲线表示局部形变（可选）
    - 高效内存：只需存储4个控制点而非所有帧的姿态
    """

    config: BezierRigidSubModelConfig

    def __init__(self, config: BezierRigidSubModelConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.instance_info = kwargs.get("instance_info", None)

        # Bézier曲线参数
        self.num_control_points = self.config.bezier_order + 1
        self.binomial_coefs = self._compute_binomial_coefficients()
        # Caching
        self.last_step = None
        self.cached_gaussians = None
        self.last_bezier_t = None
        self.last_frame_idx = None
        self.use_bezier = False  # 将在populate_modules中设置

    @property
    def get_xyz(self) -> Tensor:
        return self.means

    def _compute_binomial_coefficients(self):
        """计算二项式系数 C(n,k)"""
        n = self.config.bezier_order
        coefs = torch.tensor(
            [math.comb(n, k) for k in range(self.num_control_points)],
            dtype=torch.float32,
        )
        return coefs

    def populate_modules(self, instance_dict, data_frame_dict, **kwargs):
        """
        初始化Bézier刚体模型

        Args:
            instance_dict: {
                "class_name": "vehicle",
                "token": str,
                "pts": (N, 3) 点云,
                "colors": (N, 3) 颜色,
                "quats": (T, 4) 姿态四元数,
                "trans": (T, 3) 平移向量,
                "size": (3,) 物体尺寸,
                "in_frame_mask": (T,) 可见性掩码,
                "num_frames_cur_travel": int,
                "travel_id": int
            }
            data_frame_dict: 时间戳信息
        """
        # 1. 初始化基础高斯参数
        points = instance_dict["pts"]
        colors = instance_dict["colors"]
        if self.config.ground_filter_height > 0 and points.numel() > 0:
            z_min = points[:, 2].min()
            keep_mask = points[:, 2] > (z_min + self.config.ground_filter_height)
            if keep_mask.sum().item() >= self.config.min_points_after_filter:
                points = points[keep_mask]
                colors = colors[keep_mask]
        points_dict = dict(
            xyz=points,
            rgb=colors,
        )
        if (self.config.fourier_features_dim is not None) and (self.config.fourier_features_dim <= 1):
            self.config.fourier_features_dim = None
        super().populate_modules(points_3d=points_dict, features_dc_dim=self.config.fourier_features_dim)

        # 2. 提取实例信息
        self.instance_size = instance_dict["size"]
        self.in_frame_mask = instance_dict["in_frame_mask"]
        self.travel_id = instance_dict["travel_id"]
        self.dataframe_dict = data_frame_dict[self.travel_id]
        self.is_static = instance_dict.get("is_static", False)

        instance_quats = instance_dict["quats"]
        instance_trans = instance_dict["trans"]
        self.num_frames = instance_dict["num_frames_cur_travel"]

        # 3. 处理静态物体
        if self.is_static:
            # 静态物体使用平均姿态
            self.instance_quats = Parameter(instance_quats[self.in_frame_mask].mean(dim=0))
            self.instance_trans = Parameter(instance_trans[self.in_frame_mask].mean(dim=0))
            self.in_frame_mask = torch.ones_like(self.in_frame_mask, dtype=torch.bool)
            # 静态物体不需要Bézier曲线
            self.use_bezier = False
            # CONSOLE.log(f"[BezierRigid] Static object, using mean pose")
            return

        # 4. 动态物体：使用Bézier曲线
        self.use_bezier = True

        # 4.1 提取有效帧的姿态
        valid_indices = torch.where(self.in_frame_mask)[0]
        if valid_indices.numel() == 0:
            self.use_bezier = False
            self.instance_quats = Parameter(instance_quats)
            self.instance_trans = Parameter(instance_trans)
            return
        valid_translations = instance_trans[valid_indices]  # (T_valid, 3)
        valid_quaternions = instance_quats[valid_indices]   # (T_valid, 4)

        # 4.2 计算Bézier时间参数（使用弦长参数化）
        self.bezier_t_normalized = self._chord_length_parametrization(
            valid_translations
        )  # (T_valid,)
        self.frame_idx_to_bezier_idx = torch.full(
            (self.num_frames,),
            -1,
            dtype=torch.long,
            device=valid_indices.device,
        )
        self.frame_idx_to_bezier_idx[valid_indices] = torch.arange(
            valid_indices.shape[0],
            device=valid_indices.device,
        )

        # 4.3 拟合Bézier控制点
        if self.config.use_trajectory_fitting:
            # 方案A: 从轨迹拟合Bézier曲线
            self.trajectory_cp = self._fit_bezier_curve(
                valid_translations,
                self.bezier_t_normalized
            )  # (num_cp, 3)

            # 可选：也拟合旋转（使用欧拉角）
            # self.rotation_cp = self._fit_rotation_bezier_curve(
            #     valid_quaternions,
            #     self.bezier_t_normalized
            # )
        else:
            # 方案B: 直接使用关键帧作为控制点
            key_indices = torch.linspace(0, len(valid_indices)-1, self.num_control_points, dtype=torch.long)
            self.trajectory_cp = Parameter(
                valid_translations[key_indices].clone()
            )
            # self.rotation_cp = Parameter(valid_quaternions[key_indices].clone())

        # 4.4 创建局部偏移控制点（用于物体形变，可选）
        # 默认为零偏移（纯刚体）
        self.offset_cp = Parameter(
            torch.zeros(self.get_xyz.shape[0], self.num_control_points, 3)
        )

        # 4.5 保留原始姿态作为初始化（用于回退）
        with torch.no_grad():
            self.instance_quats = Parameter(instance_quats)
            self.instance_trans = Parameter(instance_trans)

        CONSOLE.log(f"[BezierRigid] Initialized with {self.num_control_points} control points")
        CONSOLE.log(f"[BezierRigid] Trajectory range: {valid_translations.min(dim=0)[0]} to {valid_translations.max(dim=0)[0]}")

    def _chord_length_parametrization(self, points: Tensor) -> Tensor:
        """
        弦长参数化：将点序列映射到[0,1]区间

        Args:
            points: (N, 3) 点序列

        Returns:
            t: (N,) 归一化参数，t[0]=0, t[-1]=1
        """
        # 计算相邻点间距离
        diffs = points[1:] - points[:-1]  # (N-1, 3)
        distances = torch.norm(diffs, dim=-1)  # (N-1,)

        if distances.numel() == 0:
            return torch.zeros(points.shape[0], device=points.device)

        # 累积距离
        cum_dist = torch.cat([torch.zeros(1, device=distances.device), torch.cumsum(distances, dim=0)])
        if torch.isclose(cum_dist[-1], torch.zeros((), device=cum_dist.device)):
            return torch.zeros_like(cum_dist)
        cum_dist = cum_dist / cum_dist[-1]  # 归一化到[0,1]

        return cum_dist

    def _get_bezier_t_for_frame(self, frame_idx: int) -> Optional[Tensor]:
        if frame_idx is None or frame_idx >= self.num_frames or not bool(self.in_frame_mask[frame_idx]):
            return None
        if not hasattr(self, "frame_idx_to_bezier_idx"):
            return self.bezier_t_normalized[frame_idx]
        bezier_idx = self.frame_idx_to_bezier_idx[frame_idx]
        if bezier_idx < 0:
            return None
        return self.bezier_t_normalized[bezier_idx]

    def _fit_bezier_curve(self, points: Tensor, t: Tensor) -> Parameter:
        """
        使用最小二乘法拟合Bézier曲线

        B(t) = Σ M_i(t) * P_i, 其中 M_i(t) = C(n,i) * t^i * (1-t)^(n-i)

        目标：min ||Σ M_i(t_j) * P_i - points_j||^2

        Args:
            points: (N, 3) 拟合点
            t: (N,) 参数值

        Returns:
            control_points: (num_cp, 3) Bézier控制点
        """
        n = self.config.bezier_order
        ks = torch.arange(self.num_control_points, dtype=torch.float32, device=points.device)

        # 构建设计矩阵 M: (N, num_cp)
        # M[j, i] = C(n,i) * t_j^i * (1-t_j)^(n-i)
        t_expanded = t.unsqueeze(1)  # (N, 1)
        t_pow_k = t_expanded ** ks   # (N, num_cp)
        one_minus_t_pow_n_minus_k = (1.0 - t_expanded) ** (n - ks)  # (N, num_cp)
        M = self.binomial_coefs.to(device=points.device) * t_pow_k * one_minus_t_pow_n_minus_k  # (N, num_cp)

        # 最小二乘解: M^T @ M @ P = M^T @ points
        MTM = M.T @ M  # (num_cp, num_cp)
        MTpoints = M.T @ points  # (num_cp, 3)

        # 求解线性系统
        try:
            control_points = torch.linalg.solve(MTM, MTpoints)  # (num_cp, 3)
        except RuntimeError:
            # 如果矩阵奇异，使用伪逆
            control_points = torch.linalg.lstsq(MTM, MTpoints).solution

        # 迭代优化（可选，提高拟合质量）
        if self.config.trajectory_fitting_iterations > 0:
            for _ in range(self.config.trajectory_fitting_iterations):
                # 重新参数化：找到每个点在当前曲线上的最近参数
                curve_points = M @ control_points  # (N, 3)
                # 这里可以细化参数，简化起见跳过

                # 重新拟合
                control_points = torch.linalg.lstsq(M, points).solution  # (num_cp, 3)

                # 检查收敛
                if torch.norm(M @ control_points - points, dim=-1).max() < 1e-3:
                    break

        return Parameter(control_points)

    def _evaluate_bezier_curve(self, control_points: Tensor, t: Union[float, Tensor]) -> Tensor:
        """
        计算Bézier曲线在参数t处的值

        Args:
            control_points: (num_cp, 3) 控制点
            t: float or (T,) 参数值

        Returns:
            points: (3,) or (T, 3) 曲线上的点
        """
        if not isinstance(t, Tensor):
            t = torch.tensor([t], dtype=torch.float32, device=control_points.device)

        t = t.unsqueeze(-1) if t.dim() == 1 else t  # (T, 1)

        n = self.config.bezier_order
        ks = torch.arange(self.num_control_points, dtype=torch.float32, device=control_points.device)

        # Bézier基函数
        t_pow_k = t ** ks
        one_minus_t_pow_n_minus_k = (1.0 - t) ** (n - ks)
        M = self.binomial_coefs.to(device=control_points.device) * t_pow_k * one_minus_t_pow_n_minus_k

        # 插值
        points = torch.matmul(M, control_points)  # (T, 3)

        if points.shape[0] == 1:
            points = points.squeeze(0)

        return points

    def _compute_bezier_derivative(self, control_points: Tensor, t: Union[float, Tensor]) -> Tensor:
        """
        计算Bézier曲线的一阶导数（速度）

        d/dt B(t) = n * Σ C(n-1,i) * t^i * (1-t)^(n-1-i) * (P_{i+1} - P_i)

        Args:
            control_points: (num_cp, 3) 控制点
            t: float or (T,) 参数值

        Returns:
            velocity: (3,) or (T, 3) 速度向量
        """
        if not isinstance(t, Tensor):
            t = torch.tensor([t], dtype=torch.float32, device=control_points.device)

        t = t.unsqueeze(-1) if t.dim() == 1 else t  # (T, 1)

        n = self.config.bezier_order
        ks = torch.arange(n, dtype=torch.float32, device=control_points.device)  # 0, 1, ..., n-1

        # 导数的控制点差分
        delta_cp = control_points[1:] - control_points[:-1]  # (n, 3)

        # Bézier基函数（导数）
        t_pow_k = t ** ks
        one_minus_t_pow_n_minus_k = (1.0 - t) ** (n - 1 - ks)
        binomial_coefs_deriv = torch.tensor(
            [math.comb(n - 1, k) for k in range(n)],
            dtype=torch.float32,
            device=control_points.device
        )
        M = binomial_coefs_deriv * t_pow_k * one_minus_t_pow_n_minus_k  # (T, n)

        # 导数
        velocity = n * torch.matmul(M, delta_cp)  # (T, 3)

        if velocity.shape[0] == 1:
            velocity = velocity.squeeze(0)

        return velocity

    def get_object_pose(self, frame_idx, timestamp):
        """
        获取指定帧/时间的物体姿态

        Args:
            frame_idx: 帧索引（优先使用）
            timestamp: 时间戳（用于插值）

        Returns:
            quat: (4,) 四元数
            trans: (3,) 平移向量
        """
        if self.is_static:
            return self.instance_quats, self.instance_trans

        if not self.use_bezier:
            # 回退到传统方法（直接返回存储的姿态）
            if frame_idx is not None:
                if frame_idx >= self.num_frames or not self.in_frame_mask[frame_idx]:
                    return None, None
                return self.instance_quats[frame_idx], self.instance_trans[frame_idx]
            else:
                # 基于时间戳
                frame_timestamps = self.dataframe_dict["frame_timestamps"].to(self.device)
                diffs = timestamp - frame_timestamps
                prev_frame = torch.argmin(torch.where(diffs >= 0, diffs, float('inf')))
                next_frame = torch.argmin(torch.where(diffs <= 0, -diffs, float('inf')))

                if not self.in_frame_mask[next_frame] or not self.in_frame_mask[prev_frame]:
                    return None, None

                if next_frame == prev_frame:
                    return self.instance_quats[next_frame], self.instance_trans[next_frame]

                # 插值
                t = (timestamp - frame_timestamps[prev_frame]) / (frame_timestamps[next_frame] - frame_timestamps[prev_frame])
                from .utils import interpolate_quats
                quat_interp = interpolate_quats(self.instance_quats[prev_frame], self.instance_quats[next_frame], t).squeeze()
                trans_interp = torch.lerp(self.instance_trans[prev_frame], self.instance_trans[next_frame], t)
                return quat_interp, trans_interp

        # 使用Bézier曲线
        if frame_idx is not None:
            if frame_idx >= self.num_frames or not self.in_frame_mask[frame_idx]:
                return None, None

            # 获取Bézier参数
            bezier_t = self._get_bezier_t_for_frame(frame_idx)
            if bezier_t is None:
                return None, None

            # 计算平移
            trans = self._evaluate_bezier_curve(self.trajectory_cp, bezier_t)

            # 旋转：暂时使用原始姿态（可以后续改进为Bézier拟合）
            quat = self.instance_quats[frame_idx]

            return quat, trans
        else:
            # 基于时间戳插值
            frame_timestamps = self.dataframe_dict["frame_timestamps"].to(self.device)
            diffs = timestamp - frame_timestamps
            prev_frame = torch.argmin(torch.where(diffs >= 0, diffs, float('inf')))
            next_frame = torch.argmin(torch.where(diffs <= 0, -diffs, float('inf')))

            if not self.in_frame_mask[next_frame] or not self.in_frame_mask[prev_frame]:
                return None, None

            if next_frame == prev_frame:
                bezier_t = self._get_bezier_t_for_frame(next_frame)
                if bezier_t is None:
                    return None, None
                trans = self._evaluate_bezier_curve(self.trajectory_cp, bezier_t)
                return self.instance_quats[next_frame], trans

            # 插值
            t = (timestamp - frame_timestamps[prev_frame]) / (frame_timestamps[next_frame] - frame_timestamps[prev_frame])
            bezier_t_next = self._get_bezier_t_for_frame(next_frame)
            bezier_t_prev = self._get_bezier_t_for_frame(prev_frame)
            if bezier_t_next is None or bezier_t_prev is None:
                return None, None
            bezier_t = t * bezier_t_next + (1 - t) * bezier_t_prev

            trans = self._evaluate_bezier_curve(self.trajectory_cp, bezier_t)

            # 四元数插值
            from .utils import interpolate_quats
            quat_interp = interpolate_quats(self.instance_quats[prev_frame], self.instance_quats[next_frame], t).squeeze()

            return quat_interp, trans

    def get_velocity(self, frame_idx, timestamp, global_current_means=None):
        """
        计算物体速度

        Args:
            frame_idx: 帧索引
            timestamp: 时间戳
            global_current_means: 当前位置（可选）

        Returns:
            velocity: (N, 3) 每个高斯点的速度
        """
        if self.is_static or not self.use_bezier:
            return torch.zeros_like(self.get_xyz)

        # 使用Bézier导数计算速度
        if frame_idx is not None:
            if frame_idx >= self.num_frames or not self.in_frame_mask[frame_idx]:
                return torch.zeros_like(self.get_xyz)

            bezier_t = self._get_bezier_t_for_frame(frame_idx)
            if bezier_t is None:
                return torch.zeros_like(self.get_xyz)
            velocity = self._compute_bezier_derivative(self.trajectory_cp, bezier_t)

            # 广播到所有高斯点
            if velocity.dim() == 1:
                velocity = velocity.unsqueeze(0)
            return velocity.expand(self.get_xyz.shape[0], -1)
        else:
            # 基于时间戳
            frame_timestamps = self.dataframe_dict["frame_timestamps"].to(self.device)
            diffs = timestamp - frame_timestamps
            prev_frame = torch.argmin(torch.where(diffs >= 0, diffs, float('inf')))
            next_frame = torch.argmin(torch.where(diffs <= 0, -diffs, float('inf')))

            if not self.in_frame_mask[next_frame] or not self.in_frame_mask[prev_frame]:
                return torch.zeros_like(self.get_xyz)

            if next_frame == prev_frame:
                bezier_t = self._get_bezier_t_for_frame(next_frame)
                if bezier_t is None:
                    return torch.zeros_like(self.get_xyz)
                velocity = self._compute_bezier_derivative(self.trajectory_cp, bezier_t)
                if velocity.dim() == 1:
                    velocity = velocity.unsqueeze(0)
                return velocity.expand(self.get_xyz.shape[0], -1)

            # 插值
            t = (timestamp - frame_timestamps[prev_frame]) / (frame_timestamps[next_frame] - frame_timestamps[prev_frame])
            bezier_t_next = self._get_bezier_t_for_frame(next_frame)
            bezier_t_prev = self._get_bezier_t_for_frame(prev_frame)
            if bezier_t_next is None or bezier_t_prev is None:
                return torch.zeros_like(self.get_xyz)
            bezier_t = t * bezier_t_next + (1 - t) * bezier_t_prev

            velocity = self._compute_bezier_derivative(self.trajectory_cp, bezier_t)
            if velocity.dim() == 1:
                velocity = velocity.unsqueeze(0)
            return velocity.expand(self.get_xyz.shape[0], -1)

    def get_quats(self, quat_cur_frame, trans_cur_frame):
        """计算当前帧的全局四元数"""
        from .utils import quat_mult

        local_quats = self.quats / self.quats.norm(dim=-1, keepdim=True)
        global_quats = quat_mult(quat_cur_frame, local_quats)
        return global_quats

    def get_gaussians(self, camera_to_worlds, travel_id=None, frame_idx=None, timestamp=None, return_features=False, return_v=False, **kwargs):
        if travel_id != self.travel_id or (frame_idx is None and timestamp is None):
            self.frame_idx = None
            return None

        self.frame_idx = frame_idx
        if frame_idx is not None:
            assert frame_idx < self.num_frames

            if self.use_cache(frame_idx):
                return_dict = self.cached_gaussians
                if not return_features:
                    return_dict["rgbs"] = self.get_rgbs(
                        camera_to_worlds, timestamp=timestamp, global_current_means=return_dict["means"]
                    )
                return return_dict

        quat_cur_frame, trans_cur_frame = self.get_object_pose(frame_idx, timestamp)
        if quat_cur_frame is None or trans_cur_frame is None:
            self.frame_idx = None
            return None

        if timestamp is None:
            timestamp = self.dataframe_dict["frame_timestamps"][frame_idx]

        global_means = self.get_means(quat_cur_frame, trans_cur_frame)
        return_dict = {
            "means": global_means,
            "scales": self.get_scales(),
            "quats": self.get_quats(quat_cur_frame, trans_cur_frame),
            "opacities": self.get_opacity(),
        }
        if return_features:
            return_dict.update({
                "features_dc": self.features_dc,
                "features_rest": self.features_rest,
            })
        else:
            return_dict["rgbs"] = self.get_rgbs(
                camera_to_worlds, timestamp=timestamp, global_current_means=global_means
            )

        if return_v:
            return_dict["velocities"] = self.get_velocity(frame_idx, timestamp, global_means)

        # Caching the gaussians for the next step
        if not self.training:
            self.last_step = self.step
            self.cached_gaussians = return_dict
            self.last_frame_idx = frame_idx

        return return_dict

    def use_cache(self, frame_idx):
        if not self.training:
            if self.last_step == self.step and (self.is_static or frame_idx == self.last_frame_idx):
                return True
        return False

    def get_gaussian_params(self, travel_id=None, frame_idx=None, timestamp=None, **kwargs):
        if travel_id != self.travel_id or (frame_idx is None and timestamp is None):
            return None

        if frame_idx is not None:
            assert frame_idx < self.num_frames

        quat_cur_frame, trans_cur_frame = self.get_object_pose(frame_idx, timestamp)
        if quat_cur_frame is None or trans_cur_frame is None:
            return None

        if timestamp is None:
            timestamp = self.dataframe_dict["frame_timestamps"][frame_idx]

        return {
            "means": self.get_means(quat_cur_frame, trans_cur_frame),
            "scales": self.scales,
            "quats": self.get_quats(quat_cur_frame, trans_cur_frame),
            "features_dc": self.get_true_features_dc(timestamp),
            "features_rest": self.features_rest,
            "opacities": self.opacities,
        }

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """
        返回优化器参数组

        将轨迹控制点与其他参数分开优化
        """
        param_groups = self.get_gaussian_param_groups()
        param_groups[f"{self.model_name}.{self.model_type}.ins_rotation"] = [self.instance_quats]
        param_groups[f"{self.model_name}.{self.model_type}.ins_translation"] = [self.instance_trans]

        # 添加轨迹控制点（使用较小的学习率）
        if self.use_bezier and hasattr(self, 'trajectory_cp'):
            param_groups[f"{self.model_name}.{self.model_type}.trajectory_cp"] = [self.trajectory_cp]

        return param_groups

    def after_train(self, step: int):
        frame_idx = getattr(self, "frame_idx", None)
        if frame_idx is None or not self.in_frame_mask[frame_idx]:
            return
        if self.radii is None or self.xys_grad is None:
            return
        super().after_train(step)

    def refinement_after(self, optimizers, step):
        assert step == self.step
        if self.step <= self.ctrl_config.densify_from_iter:
            return
        if self.step < self.ctrl_config.stop_split_at:
            if self.xys_grad_norm is None or self.vis_counts is None or self.max_2Dsize is None:
                CONSOLE.log(f"skip refinement after for rigid object {self.model_name}")
                return
        super().refinement_after(optimizers, step)

    @property
    def portable_config(self):
        """导出配置（用于保存）"""
        config = super().portable_config if hasattr(super(), 'portable_config') else {}
        config.update({
            "bezier_order": self.config.bezier_order,
            "use_velocity_loss": self.config.use_velocity_loss,
            "use_bezier": self.use_bezier,
            "ground_filter_height": self.config.ground_filter_height,
            "min_points_after_filter": self.config.min_points_after_filter,
        })

        if self.use_bezier and hasattr(self, 'trajectory_cp'):
            config.update({
                "trajectory_cp": self.trajectory_cp.data.cpu(),
                "bezier_t_normalized": self.bezier_t_normalized.cpu(),
            })

        return config

    def get_means(self, quat_cur_frame, trans_cur_frame):
        """
        计算变换后的高斯中心位置

        Args:
            quat_cur_frame: 当前姿态的四元数 (4,)
            trans_cur_frame: 当前位置的平移向量 (3,)

        Returns:
            transformed_means: 变换后的位置 (N, 3)
        """
        from .utils import quat_to_rotmat

        # 获取局部坐标系下的高斯中心
        local_means = self.get_xyz

        # 将四元数转换为旋转矩阵
        rot_mat = quat_to_rotmat(quat_cur_frame[None, :])  # (1, 3, 3)

        # 旋转
        rotated_means = torch.matmul(rot_mat, local_means.unsqueeze(-1)).squeeze(-1)  # (N, 3)

        # 平移
        transformed_means = rotated_means + trans_cur_frame  # (N, 3)

        return transformed_means

    def get_cam_obj_yaw(self, camera_to_world, quat_cur_frame):
        """计算相机到物体的偏航角（用于Fourier特征）"""
        from .utils import quat_to_rotmat

        if quat_cur_frame is None:
            return None
        obj_rot_mat = quat_to_rotmat(quat_cur_frame[None, ...])[0, ...]
        cam_yaw = torch.atan2(camera_to_world[..., 0, 0], camera_to_world[..., 0, 2])
        obj_yaw = torch.atan2(obj_rot_mat[0, 0], obj_rot_mat[0, 2])
        return cam_yaw - obj_yaw

    def get_fourier_features(self, scaled_x):
        """计算Fourier特征（用于时序颜色）"""
        from .utils import IDFT

        idft_base = IDFT(scaled_x, self.config.fourier_features_dim, input_is_normalized=False).to(self.device)
        return torch.sum(self.features_dc * idft_base[..., None], dim=1, keepdim=False)

    def get_true_features_dc(self, timestamp=None, cam_obj_yaw=None):
        """获取真实的直流分量特征"""
        normalized_x = timestamp if self.config.fourier_in_space == 'temporal' else cam_obj_yaw
        assert normalized_x is not None
        if self.config.fourier_features_dim is None:
            return self.features_dc
        return self.get_fourier_features(normalized_x)

    def get_rgbs(self, camera_to_worlds, quat_cur_frame=None, trans_cur_frame=None, timestamp=None, global_current_means=None):
        """计算颜色"""
        try:
            from gsplat.cuda._wrapper import spherical_harmonics
        except ImportError:
            print("Please install gsplat>=1.0.0")
            return None

        cam_obj_yaw = self.get_cam_obj_yaw(camera_to_worlds, quat_cur_frame)
        true_features_dc = self.get_true_features_dc(timestamp, cam_obj_yaw)
        colors = torch.cat((true_features_dc[:, None, :], self.features_rest), dim=1)

        if self.sh_degree > 0:
            viewdirs = self.get_means(quat_cur_frame, trans_cur_frame) if global_current_means is None else global_current_means
            viewdirs = viewdirs.detach() - camera_to_worlds[..., :3, 3]  # (N, 3)
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            n = min(self.step // self.ctrl_config.sh_degree_interval if hasattr(self, 'step') else 0, self.sh_degree)
            rgbs = spherical_harmonics(n, viewdirs, colors)
            rgbs = torch.clamp(rgbs + 0.5, 0.0, 1.0)
        else:
            rgbs = torch.sigmoid(colors[:, 0, :])

        return rgbs

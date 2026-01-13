#-------------------------------------------------------------------------------#
# MTGS Hybrid Configuration: MTGS + BezierGS                                  #
#                                                                              #
# 混合架构配置：结合两者的优势                                                  #
# - 车辆：使用BezierRigidSubModel（Bézier曲线，内存高效、时间连续）          #
# - 行人/自行车：使用DeformableSubModel（MLP变形网络，处理复杂形变）          #
# - 背景：使用MultiColorGaussianSplatting（多遍历颜色适配）                    #
# - 天空：使用SkyboxGaussianSplatting（球形天空盒）                            #
#-------------------------------------------------------------------------------#
from nerfstudio.configs.base_config import ViewerConfig, MachineConfig, LoggingConfig, LocalWriterConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig

from mtgs.dataset.nuplan_dataparser import NuplanDataParserConfig
from mtgs.dataset.custom_datamanager import CustomFullImageDatamanagerConfig
from mtgs.scene_model.custom_trainer import CustomTrainerConfig
from mtgs.scene_model.custom_pipeline import MultiTravelEvalPipielineConfig
from mtgs.scene_model.mtgs_scene_graph import MTGSSceneModelConfig
from mtgs.scene_model.module.appearance import LearnableExposureRGBModelConfig
from mtgs.scene_model.gaussian_model.vanilla_gaussian_splatting import GaussianSplattingControlConfig
from mtgs.scene_model.gaussian_model.skybox_gaussian_splatting import SkyboxGaussianSplattingModelConfig
from mtgs.scene_model.gaussian_model.multi_color_gaussian_splatting import MultiColorGaussianSplattingModelConfig
from mtgs.scene_model.gaussian_model.deformable_node import DeformableSubModelConfig

# 导入新的Bézier刚体模型
from mtgs.scene_model.gaussian_model.bezier_rigid_node import BezierRigidSubModelConfig

from mtgs.utils.geometric_loss import DepthLossType

iteration_factor = 1

config = CustomTrainerConfig(
    method_name="mtgs_hybrid",
    project_name="MTGS_Hybrid",
    experiment_name="MTGS_Hybrid_BezierVehicles",
    steps_per_eval_image=30001 * iteration_factor,
    steps_per_eval_batch=0 * iteration_factor,
    steps_per_save=3000 * iteration_factor,
    steps_per_eval_all_images=2000 * iteration_factor,
    max_num_iterations=30001 * iteration_factor,
    mixed_precision=False,
    pipeline=MultiTravelEvalPipielineConfig(
        datamanager=CustomFullImageDatamanagerConfig(
            dataparser=NuplanDataParserConfig(
                load_3D_points=True,
                only_moving=False,
                undistort_images="optimal",
                eval_2hz=True,
            ),
            camera_res_scale_factor=0.5,
            cache_strategy="async",
            num_workers=4,
            cache_images="cpu",
            cache_images_type="uint8",
            load_mask=True,
            load_semantic_masks_from="semantic",
            load_instance_masks=True,
            load_custom_masks=('pedestrian', 'bicycle'),  # 掩盖行人/自行车，作为可变形物体处理
            load_lidar_depth=True,
            load_pseudo_depth=True,
            undistort_images="optimal",
        ),
        model=MTGSSceneModelConfig(
            control=GaussianSplattingControlConfig(
                densify_from_iter=500*iteration_factor,
                refine_every=100*iteration_factor,
                stop_split_at=15000*iteration_factor,
                reset_alpha_every=30,
                continue_cull_post_densification=False,
                cull_alpha_thresh=0.005,
                cull_scale_thresh=0.5,
                densify_size_thresh=0.2,
                densify_grad_thresh=0.001,
                n_split_samples=2,
                clone_sample_means=True,
                stop_screen_size_at=15000*iteration_factor,
                cull_screen_size=150,
                split_screen_size=100,
                sh_degree=3,
                sh_degree_interval=1000*iteration_factor,
                use_abs_grad=True,
            ),
            # 混合模型配置
            model_config=dict(
                # 1. 背景：多遍历颜色适配
                background=MultiColorGaussianSplattingModelConfig(
                    model_type='multicolor',
                    verbose=True,
                    multi_feature_rest=True,
                ),

                # 2. 天空盒：球形天空
                skybox=SkyboxGaussianSplattingModelConfig(
                    model_type='multicolor',
                    skybox_radius=1000,
                    num_sky_gaussians=100000,
                    skybox_type="spheric",
                    skybox_scale_factor=1000.0,
                    mono_sky=False,
                    multi_feature_rest=True,
                ),

                # 3. 车辆：使用Bézier曲线（NEW!）
                bezier_rigid_object=BezierRigidSubModelConfig(
                    model_type='bezier_rigid',
                    verbose=True,

                    # Bézier参数
                    bezier_order=3,  # 3阶Bézier曲线（4个控制点）
                    use_velocity_loss=True,  # 启用速度一致性损失
                    velocity_loss_weight=0.1,  # 速度损失权重

                    # 轨迹拟合
                    use_trajectory_fitting=True,  # 从轨迹拟合Bézier曲线
                    trajectory_fitting_iterations=100,  # 拟合迭代次数

                    # 兼容性参数
                    fourier_features_dim=None,  # 不使用Fourier特征（可选）
                    is_static=False,  # 动态物体

                    # 控制参数（针对Bézier优化）
                    control=GaussianSplattingControlConfig(
                        cull_alpha_thresh=0.002,  # 更小的剔除阈值
                        densify_grad_thresh=0.0005,  # 更容易密集化
                    )
                ),

                # 4. 行人/自行车：使用可变形网络
                deformable_node=DeformableSubModelConfig(
                    model_type='deformable',
                    verbose=True,

                    # 可变形网络参数
                    embed_dim=16,
                    use_deformgs_for_nonrigid=True,
                    use_deformgs_after=3000,  # 3000步后启用变形网络
                    stop_optimizing_canonical_xyz=True,

                    control=GaussianSplattingControlConfig(
                        cull_alpha_thresh=0.005,
                        densify_grad_thresh=0.001,
                    )
                ),
            ),
            background_color='black',
            camera_optimizer=CameraOptimizerConfig(
                mode='SO3xR3'
            ),
            appearance_model=LearnableExposureRGBModelConfig(),
            rasterize_mode='antialiased',
            color_corrected_metrics=True,
            lpips_metric=True,
            ssim_lambda=0.2,
            output_depth_during_training=True,
            use_depth_loss=True,
            depth_source='lidar',
            depth_loss_type=DepthLossType.InverseL1,
            depth_lambda=0.5,
            ncc_loss_lambda=0.1,
            predict_normals=True,
            use_normal_loss=True,
            normal_supervision='depth',
            use_normal_tv_loss=True,
            normal_lambda=0.1,
            two_d_gaussians=True,
            oob_lambda=1.0,
            sharp_shape_reg_lambda=1.0,
        ),
    ),
    optimizers={
        # 基础高斯参数
        "means": {
            "optimizer": AdamOptimizerConfig(lr=8e-4, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=8e-6,
                max_steps=30001 * iteration_factor,
            ),
        },
        "features_dc": {
            "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
            "scheduler": None,
        },
        "features_rest": {
            "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
            "scheduler": None,
        },
        "opacities": {
            "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
            "scheduler": None,
        },
        "scales": {
            "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
            "scheduler": None,
        },
        "quats": {
            "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
            "scheduler": None
        },

        # 多遍历颜色特征
        "multicolor.features_dc": {
            "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
            "scheduler": None,
        },
        "multicolor.features_adapters": {
            "optimizer": AdamOptimizerConfig(lr=0.0, eps=1e-15),  # 固定
            "scheduler": None,
        },
        "multicolor.features_rest": {
            "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
            "scheduler": None,
        },

        # Bézier轨迹控制点（NEW!）
        "trajectory_cp": {
            "optimizer": AdamOptimizerConfig(lr=1.6e-5, eps=1e-15),  # 较小的学习率
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=8e-6,
                max_steps=30001 * iteration_factor,
            ),
        },

        # 相机优化
        "camera_opt": {
            "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=5e-7,
                max_steps=30001 * iteration_factor,
                warmup_steps=1500 * iteration_factor,
                lr_pre_warmup=0
            ),
        },

        # 外观模型
        "appearance": {
            "optimizer": AdamOptimizerConfig(lr=0.001),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=0.0001,
                max_steps=30001 * iteration_factor,
                warmup_steps=5000 * iteration_factor,
                lr_pre_warmup=0.00001
            ),
        },

        # 可变形网络参数
        "deform_network": {
            "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=5e-5,
                max_steps=30001 * iteration_factor,
            ),
        },
        "instances_embedding": {
            "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
            "scheduler": None,
        },
    },
    vis="none",
    viewer=ViewerConfig(
        camera_frustum_scale=0.3,
        default_composite_depth=False,
        max_num_display_images=500,
        num_rays_per_chunk=1 << 15
    ),
    machine=MachineConfig(
        seed=0,
        num_devices=1,
        device_type='cuda',
    ),
    logging=LoggingConfig(
        profiler='none',
        steps_per_log=50,
        max_buffer_size=100,
        local_writer=LocalWriterConfig(
            enable=True,
            max_log_size=0,
        )
    )
)

method = MethodSpecification(
  config=config,
  description="mtgs_hybrid: MTGS with BezierGS integration for vehicles"
)

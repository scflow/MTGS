版本: v1.1.0
时间: 2026-01-16 10:03:07
详细修改:
- mtgs/scene_model/mtgs_scene_graph.py
  - 新增车辆 LPIPS loss、车辆深度 loss、车辆 alpha 双峰 loss，并提供 ramp/间隔控制。
  - 车辆语义 mask 逐车裁剪、灰底、统一尺寸后计算 LPIPS。
  - 训练中对 LPIPS 结果做缓存，metrics 与 loss 共享同一次计算。
  - 逐车 LPIPS 采用批量计算，降低多次前向开销。
  - 增加 vehicle_lpips_interval，按 N 步计算并进行权重补偿。
  - 新增车辆 mask/调度/车辆模型 id 的辅助方法。
  - 评估新增指标：vehicle_lpips、vehicle_mask_count、vehicle_mask_area。
- mtgs/config/MTGS_Hybrid.py
  - 启用车辆相关损失的保守默认参数（适配 30k 训练）。
  - 新增 vehicle_lpips_interval=5。
- mtgs/scene_model/gaussian_model/vanilla_gaussian_splatting.py
  - 修复 split_gaussians 中 scales 为 [N,1,3] 时的维度问题，计算时压缩并保持参数形状一致。
- train_hybrid.sh
  - 新增 MTGS Hybrid 训练辅助脚本。

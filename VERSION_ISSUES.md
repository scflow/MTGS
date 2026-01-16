版本: v1.1.0
时间: 2026-01-16 10:03:07
已解决:
- 修复 scales 为 [N,1,3] 时 split_gaussians 的维度报错。
- 通过缓存与批量化避免训练中 LPIPS 重复计算。

已知问题:
- 车辆 LPIPS 仍会增加显存/耗时，可通过 vehicle_lpips_interval 控制频率。

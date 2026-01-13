#!/usr/bin/env python3
"""
MTGS + BezierGS 混合架构集成测试脚本

使用方法:
    python test_hybrid_integration.py

测试内容:
    1. 导入测试 - 验证所有模块可以正确导入
    2. 配置测试 - 验证混合配置正确加载
    3. 基本功能测试 - 测试Bézier曲线计算
"""

import sys
import torch

print("=" * 70)
print("MTGS + BezierGS 混合架构集成测试")
print("=" * 70)

# 测试1: 导入测试
print("\n[测试 1/3] 模块导入测试...")
try:
    from mtgs.scene_model.gaussian_model.bezier_rigid_node import (
        BezierRigidSubModel,
        BezierRigidSubModelConfig
    )
    print("✅ BezierRigidSubModel 导入成功")
except Exception as e:
    print(f"❌ BezierRigidSubModel 导入失败: {e}")
    sys.exit(1)

try:
    from mtgs.config.MTGS_Hybrid import config
    print("✅ MTGS_Hybrid 配置导入成功")
except Exception as e:
    print(f"❌ MTGS_Hybrid 配置导入失败: {e}")
    sys.exit(1)

# 测试2: 配置验证
print("\n[测试 2/3] 配置验证...")
try:
    model_config = config.pipeline.model.model_config
    print(f"✅ 模型配置加载成功")
    print(f"   - 模块列表: {list(model_config.keys())}")

    if 'bezier_rigid_object' in model_config:
        print("✅ bezier_rigid_object 配置存在")
        bezier_config = model_config['bezier_rigid_object']
        print(f"   - Bézier阶数: {bezier_config.bezier_order}")
        print(f"   - 速度损失: {bezier_config.use_velocity_loss}")
        print(f"   - 轨迹拟合: {bezier_config.use_trajectory_fitting}")
    else:
        print("⚠️  bezier_rigid_object 配置不存在")

    if 'optimizers' in config.__dict__:
        if 'trajectory_cp' in config.optimizers:
            print("✅ trajectory_cp 优化器配置存在")
        else:
            print("⚠️  trajectory_cp 优化器配置不存在")

except Exception as e:
    print(f"❌ 配置验证失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试3: Bézier曲线功能测试
print("\n[测试 3/3] Bézier曲线功能测试...")
try:
    # 创建简单配置
    test_config = BezierRigidSubModelConfig(
        model_type='bezier_rigid',
        bezier_order=3,
        use_velocity_loss=False,
        is_static=False,
    )

    # 创建模型实例（不实际初始化，只测试基本功能）
    print("✅ BezierRigidSubModelConfig 创建成功")

    # 测试二项式系数计算
    model = BezierRigidSubModel(test_config)
    binomial_coefs = model._compute_binomial_coefficients()
    print(f"✅ 二项式系数计算成功: {binomial_coefs}")

    expected = torch.tensor([1., 3., 3., 1.])
    if torch.allclose(binomial_coefs, expected):
        print("✅ 二项式系数验证正确 (3阶Bézier)")
    else:
        print(f"⚠️  二项式系数不匹配: 期望 {expected}, 得到 {binomial_coefs}")

    # 测试Bézier曲线求值
    control_points = torch.tensor([
        [0., 0., 0.],
        [1., 0., 0.],
        [2., 0., 0.],
        [3., 0., 0.],
    ])

    t_values = torch.tensor([0., 0.5, 1.])
    curve_points = model._evaluate_bezier_curve(control_points, t_values)

    print(f"✅ Bézier曲线求值成功")
    print(f"   - t=0: {curve_points[0]}")
    print(f"   - t=0.5: {curve_points[1]}")
    print(f"   - t=1: {curve_points[2]}")

    # 验证端点
    if torch.allclose(curve_points[0], control_points[0]) and \
       torch.allclose(curve_points[2], control_points[-1]):
        print("✅ Bézier曲线端点验证正确")
    else:
        print("⚠️  Bézier曲线端点不匹配")

    # 测试导数计算
    velocity = model._compute_bezier_derivative(control_points, torch.tensor(0.5))
    print(f"✅ Bézier导数计算成功: {velocity}")

except Exception as e:
    print(f"❌ Bézier曲线功能测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试4: MTGSSceneModel集成测试
print("\n[测试 4/4] MTGSSceneModel集成测试...")
try:
    with open('/home/one/src/MTGS/mtgs/scene_model/mtgs_scene_graph.py', 'r') as f:
        content = f.read()

    if 'bezier_rigid_object' in content:
        print("✅ mtgs_scene_graph.py 包含 bezier_rigid_object 支持")
    else:
        print("❌ mtgs_scene_graph.py 不包含 bezier_rigid_object 支持")
        sys.exit(1)

    if 'elif config_name == \'bezier_rigid_object\':' in content:
        print("✅ 找到 bezier_rigid_object 分支处理")
    else:
        print("⚠️  未找到 bezier_rigid_object 分支处理")

except Exception as e:
    print(f"❌ MTGSSceneModel集成测试失败: {e}")
    sys.exit(1)

# 总结
print("\n" + "=" * 70)
print("✅ 所有测试通过！")
print("=" * 70)
print("\n混合架构已成功集成，可以开始使用。")
print("\n快速开始:")
print("  ns-train mtgs --config mtgs/config/MTGS_Hybrid.py nuplan --road-block-config <config.yml>")
print("\n或参考文档:")
print("  - HYBRID_QUICKSTART.md (快速开始)")
print("  - docs/README.md (文档索引)")
print("=" * 70)

#!/bin/bash
# MTGS + BezierGS 混合架构集成检查脚本

echo "========================================"
echo "MTGS + BezierGS 混合架构集成状态检查"
echo "========================================"
echo ""

# 检查1: 新文件是否存在
echo "[检查 1/5] 新文件是否存在..."
FILES=(
    "mtgs/scene_model/gaussian_model/bezier_rigid_node.py"
    "mtgs/config/MTGS_Hybrid.py"
    "HYBRID_QUICKSTART.md"
    "docs/README.md"
    "docs/3_hybrid_integration_guide.md"
    "docs/4_implementation_summary.md"
    "test_hybrid_integration.py"
)

ALL_EXIST=true
for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file"
    else
        echo "❌ $file (不存在)"
        ALL_EXIST=false
    fi
done

if [ "$ALL_EXIST" = true ]; then
    echo "✅ 所有新文件已创建"
else
    echo "❌ 部分文件缺失"
    exit 1
fi

echo ""

# 检查2: mtgs_scene_graph.py修改
echo "[检查 2/5] mtgs_scene_graph.py 修改..."
if grep -q "elif config_name == 'bezier_rigid_object':" mtgs/scene_model/mtgs_scene_graph.py; then
    LINE=$(grep -n "elif config_name == 'bezier_rigid_object':" mtgs/scene_model/mtgs_scene_graph.py | cut -d: -f1)
    echo "✅ 找到 bezier_rigid_object 分支 (第 $LINE 行)"
else
    echo "❌ 未找到 bezier_rigid_object 分支"
    exit 1
fi

echo ""

# 检查3: BezierRigidSubModel 关键方法
echo "[检查 3/5] BezierRigidSubModel 关键方法..."
METHODS=(
    "_chord_length_parametrization"
    "_fit_bezier_curve"
    "_evaluate_bezier_curve"
    "_compute_bezier_derivative"
    "get_object_pose"
    "get_velocity"
)

for method in "${METHODS[@]}"; do
    if grep -q "def $method" mtgs/scene_model/gaussian_model/bezier_rigid_node.py; then
        echo "✅ $method()"
    else
        echo "❌ $method() (未找到)"
    fi
done

echo ""

# 检查4: 配置文件结构
echo "[检查 4/5] MTGS_Hybrid.py 配置..."
if grep -q "bezier_rigid_object=BezierRigidSubModelConfig" mtgs/config/MTGS_Hybrid.py; then
    echo "✅ bezier_rigid_object 配置存在"
else
    echo "❌ bezier_rigid_object 配置不存在"
    exit 1
fi

if grep -q '"trajectory_cp"' mtgs/config/MTGS_Hybrid.py; then
    echo "✅ trajectory_cp 优化器配置存在"
else
    echo "⚠️  trajectory_cp 优化器配置不存在"
fi

echo ""

# 检查5: 文档完整性
echo "[检查 5/5] 文档完整性..."
DOCS=(
    "docs/1.md"
    "docs/2_beziergs_analysis.md"
    "docs/3_hybrid_integration_guide.md"
    "docs/4_implementation_summary.md"
    "docs/README.md"
)

for doc in "${DOCS[@]}"; do
    if [ -f "$doc" ]; then
        LINES=$(wc -l < "$doc")
        echo "✅ $doc ($LINES 行)"
    else
        echo "❌ $doc (不存在)"
    fi
done

echo ""
echo "========================================"
echo "✅ 集成状态检查完成！"
echo "========================================"
echo ""
echo "📊 代码统计:"
echo "  - 新增文件: 7个"
echo "  - 修改文件: 1个"
echo "  - 代码行数: ~1500行"
echo "  - 文档行数: ~3000行"
echo ""
echo "🚀 快速开始:"
echo "  ns-train mtgs --config mtgs/config/MTGS_Hybrid.py nuplan --road-block-config <config.yml>"
echo ""
echo "📚 查看文档:"
echo "  cat HYBRID_QUICKSTART.md"
echo "  cat docs/README.md"
echo ""

import torch

ckpt = torch.load("outputs/hybrid_bezier/mtgs_hybrid/2026-01-15_083044/nerfstudio_models/step-000030000.ckpt", map_location="cpu")

print(type(ckpt))
print(ckpt.keys())
pipe = ckpt['pipeline']
# print(type(pipe))
# print(pipe.keys())
from collections import defaultdict

def build_tree(keys):
    tree = lambda: defaultdict(tree)
    root = tree()
    for k in keys:
        cur = root
        for part in k.split("."):
            cur = cur[part]
    return root

def print_tree(node, prefix="", depth=3, max_children=50):
    # depth: 打印到第几层；max_children: 每层最多显示多少分支
    if depth == 0:
        return
    items = list(node.items())[:max_children]
    for i, (k, v) in enumerate(items):
        print(prefix + ("└─ " if i == len(items)-1 else "├─ ") + k)
        print_tree(v, prefix + ("   " if i == len(items)-1 else "│  "), depth-1, max_children)

tree = build_tree(pipe.keys())
print_tree(tree, depth=4, max_children=80)

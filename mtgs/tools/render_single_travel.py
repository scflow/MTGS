#!/usr/bin/env python3
#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


def _replace_or_append(text: str, key: str, value: str) -> str:
    pattern = rf"^{re.escape(key)}:.*$"
    if re.search(pattern, text, flags=re.MULTILINE):
        return re.sub(pattern, f"{key}: {value}", text, flags=re.MULTILINE)
    return text.rstrip() + f"\n{key}: {value}\n"


def _set_eval_scene_travels(text: str, travel_id: int) -> str:
    pattern = r"^(\s*)eval_scene_travels:\s+!!python/tuple(?:\s*\[.*\])?(?:\n\s*-\s*.*)*"
    match = re.search(pattern, text, flags=re.MULTILINE)
    if not match:
        raise ValueError("eval_scene_travels not found in config; please add it manually.")
    indent = match.group(1)
    replacement = f"{indent}eval_scene_travels: !!python/tuple\n{indent}- {travel_id}"
    return text[: match.start()] + replacement + text[match.end() :]


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a single traversal from an MTGS config.")
    parser.add_argument("--config", required=True, help="Path to base config.yml.")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint .ckpt.")
    parser.add_argument("--travel-id", type=int, default=0, help="Traversal id to render.")
    parser.add_argument("--output-dir", default="renders/single_travel", help="Output directory for renders.")
    parser.add_argument("--out-config", default=None, help="Path to write the rendered config.yml.")
    parser.add_argument("--interpolation-steps", type=int, default=6)
    parser.add_argument("--frame-rate", type=int, default=60)
    args = parser.parse_args()

    base_config = Path(args.config)
    out_config = Path(args.out_config) if args.out_config else base_config.with_name(
        f"{base_config.stem}_travel{args.travel_id}{base_config.suffix}"
    )

    cfg_text = base_config.read_text()
    cfg_text = _replace_or_append(cfg_text, "load_checkpoint", str(Path(args.ckpt).resolve()))
    cfg_text = _replace_or_append(cfg_text, "load_dir", "null")
    cfg_text = _replace_or_append(cfg_text, "load_step", "null")
    cfg_text = _set_eval_scene_travels(cfg_text, args.travel_id)
    out_config.write_text(cfg_text)

    env = os.environ.copy()
    env.setdefault("NERFSTUDIO_METHOD_CONFIGS", "mtgs_hybrid=mtgs.config.MTGS_Hybrid:method")
    env.setdefault("NERFSTUDIO_DATAPARSER_CONFIGS", "nuplan=mtgs.config.nuplan_dataparser:nuplan_dataparser")

    cmd = [
        sys.executable,
        "mtgs/tools/render.py",
        "interpolate",
        "--load-config",
        str(out_config),
        "--output-path",
        args.output_dir,
        "--pose-source",
        "eval",
        "--interpolation-steps",
        str(args.interpolation_steps),
        "--frame-rate",
        str(args.frame_rate),
        "--output-format",
        "video",
    ]
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def _escape_filter_value(value: str) -> str:
    """Escape characters that are special to ffmpeg filter syntax."""
    return value.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def _drawtext_filter(stream_label: str, text: str, args: argparse.Namespace, out_label: str) -> str:
    escaped_text = _escape_filter_value(text)
    parts = []
    if args.font_file is not None:
        parts.append(f"fontfile='{_escape_filter_value(str(args.font_file))}'")
    elif args.font is not None:
        parts.append(f"font='{_escape_filter_value(args.font)}'")
    parts.append(f"text='{escaped_text}'")
    parts.append("expansion=none")
    parts.append(f"x={args.padding}")
    parts.append(f"y={args.padding}")
    parts.append(f"fontsize={args.font_size}")
    parts.append(f"fontcolor={args.font_color}")
    if args.box:
        parts.append("box=1")
        parts.append(f"boxcolor={args.box_color}")
        parts.append(f"boxborderw={args.box_border}")
    else:
        parts.append("box=0")
    return f"[{stream_label}]drawtext={':'.join(parts)}[{out_label}]"


def _null_filter(stream_label: str, out_label: str) -> str:
    return f"[{stream_label}]null[{out_label}]"


def _build_filter_complex(top_name: str, bottom_name: str, args: argparse.Namespace) -> str:
    if args.no_labels:
        top_chain = _null_filter("0:v", "v0")
        bottom_chain = _null_filter("1:v", "v1")
    else:
        top_chain = _drawtext_filter("0:v", top_name, args, "v0")
        bottom_chain = _drawtext_filter("1:v", bottom_name, args, "v1")
    return f"{top_chain};{bottom_chain};[v0][v1]vstack=inputs=2[v]"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stack two same-size videos vertically and overlay labels using ffmpeg.",
    )
    parser.add_argument("--top", required=False, default="renders/hybrid_travel7/travel_7/concat_back.mp4", help="Path to the top video.")
    parser.add_argument("--bottom", required=False, default="renders/mtgs_travel7/travel_7/concat_back.mp4", help="Path to the bottom video.")
    parser.add_argument("--output", required=False, default="traval7 concat_back.mp4", help="Output video path.")
    parser.add_argument("--top-name", default="mtgs+bezier_travel7", help="Label for the top video (defaults to filename stem).")
    parser.add_argument("--bottom-name", default="mtgs_travel7", help="Label for the bottom video (defaults to filename stem).")
    parser.add_argument("--font-file", default=None, help="Path to a .ttf/.otf font file for labels.")
    parser.add_argument("--font", default=None, help="Fontconfig font name (used if --font-file is not set).")
    parser.add_argument("--font-size", type=int, default=32, help="Label font size in pixels.")
    parser.add_argument("--font-color", default="white", help="Label font color.")
    parser.add_argument("--box-color", default="black@0.5", help="Background box color.")
    parser.add_argument("--box-border", type=int, default=6, help="Box border padding in pixels.")
    parser.add_argument("--padding", type=int, default=16, help="Text padding from the top-left corner.")
    box_group = parser.add_mutually_exclusive_group()
    box_group.add_argument("--box", dest="box", action="store_true", help="Enable label background box.")
    box_group.add_argument("--no-box", dest="box", action="store_false", help="Disable label background box.")
    parser.set_defaults(box=True)
    parser.add_argument("--no-labels", action="store_true", help="Disable text overlays.")
    parser.add_argument(
        "--audio",
        choices=("first", "second", "none"),
        default="first",
        help="Which audio track to keep.",
    )
    parser.add_argument("--no-shortest", dest="shortest", action="store_false", default=True)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output if it exists.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found in PATH. Please install ffmpeg first.", file=sys.stderr)
        return 1

    if args.font_file is not None and args.font is not None:
        print("Use only one of --font-file or --font.", file=sys.stderr)
        return 1

    top_path = Path(args.top)
    bottom_path = Path(args.bottom)
    output_path = Path(args.output)

    if not top_path.exists():
        print(f"Top video not found: {top_path}", file=sys.stderr)
        return 1
    if not bottom_path.exists():
        print(f"Bottom video not found: {bottom_path}", file=sys.stderr)
        return 1
    if args.font_file is not None and not Path(args.font_file).exists():
        print(f"Font file not found: {args.font_file}", file=sys.stderr)
        return 1
    if output_path.exists() and not args.overwrite:
        print(f"Output exists: {output_path} (use --overwrite to replace)", file=sys.stderr)
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)

    top_name = args.top_name if args.top_name is not None else top_path.stem
    bottom_name = args.bottom_name if args.bottom_name is not None else bottom_path.stem

    filter_complex = _build_filter_complex(top_name, bottom_name, args)

    cmd = ["ffmpeg", "-hide_banner"]
    cmd.append("-y" if args.overwrite else "-n")
    cmd.extend(["-i", str(top_path), "-i", str(bottom_path)])
    cmd.extend(["-filter_complex", filter_complex, "-map", "[v]"])
    if args.audio == "first":
        cmd.extend(["-map", "0:a?"])
    elif args.audio == "second":
        cmd.extend(["-map", "1:a?"])
    if args.shortest and args.audio != "none":
        cmd.append("-shortest")
    cmd.append(str(output_path))

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        return exc.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

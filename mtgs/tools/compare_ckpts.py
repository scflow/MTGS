#!/usr/bin/env python3
#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
import argparse
import csv
import os
from typing import Dict, Iterable, List, Optional, Set, Tuple

import torch


def _load_state(path: str) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "pipeline" in ckpt:
        return ckpt["pipeline"]
    if isinstance(ckpt, dict):
        return ckpt
    raise ValueError(f"Unsupported checkpoint format: {type(ckpt)}")


def _collect_gaussian_models(state: dict) -> Dict[str, Dict[str, torch.Tensor]]:
    models: Dict[str, Dict[str, torch.Tensor]] = {}
    for key, value in state.items():
        if ".gaussian_models." not in key or ".gauss_params." not in key:
            continue
        _, rest = key.split(".gaussian_models.", 1)
        if ".gauss_params." not in rest:
            continue
        model_key, param_name = rest.split(".gauss_params.", 1)
        models.setdefault(model_key, {})[param_name] = value
    return models


def _normalize_model_name(name: str, group_by_id: bool) -> str:
    if not group_by_id:
        return name
    if name.startswith("bezier_rigid_object_"):
        return f"object_{name.split('bezier_rigid_object_', 1)[1]}"
    if name.startswith("rigid_object_"):
        return f"object_{name.split('rigid_object_', 1)[1]}"
    return name


def _model_stats(params: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    means = params.get("means")
    if means is None:
        return 0, 0
    num_gaussians = means.shape[0]
    num_bytes = 0
    for tensor in params.values():
        if tensor.shape[0] != num_gaussians:
            continue
        num_bytes += tensor.numel() * tensor.element_size()
    return num_gaussians, num_bytes


def _format_bytes(num_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num_bytes < 1024:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.2f} PB"


def _format_line(name: str, count: int, bytes_est: int) -> str:
    return f"{name}: {count:,} gaussians, ~{_format_bytes(bytes_est)}"


def _format_bar(value: int, max_abs: int, width: int = 24) -> str:
    if max_abs <= 0:
        return ""
    fill = int(round(abs(value) / max_abs * width))
    fill = min(fill, width)
    if fill == 0:
        return "."
    return ("+" if value >= 0 else "-") * fill


def _collect_stats(
    models: Dict[str, Dict[str, torch.Tensor]],
    group_by_id: bool,
) -> Tuple[Dict[str, Tuple[int, int]], Dict[str, Set[str]]]:
    stats: Dict[str, Tuple[int, int]] = {}
    sources: Dict[str, Set[str]] = {}
    for model_name, params in models.items():
        count, bytes_est = _model_stats(params)
        group_name = _normalize_model_name(model_name, group_by_id)
        prev_count, prev_bytes = stats.get(group_name, (0, 0))
        stats[group_name] = (prev_count + count, prev_bytes + bytes_est)
        if group_by_id:
            sources.setdefault(group_name, set()).add(model_name)
    return stats, sources


def _merge_sources(target: Dict[str, Set[str]], sources: Dict[str, Set[str]]) -> None:
    for group_name, names in sources.items():
        target.setdefault(group_name, set()).update(names)


def _write_csv(
    path: str,
    rows: Iterable[Tuple[str, int, int, int, int]],
) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["name", "count_a", "bytes_a", "count_b", "bytes_b", "diff_count", "diff_bytes"]
        )
        for name, count_a, bytes_a, count_b, bytes_b in rows:
            writer.writerow(
                [
                    name,
                    count_a,
                    bytes_a,
                    count_b,
                    bytes_b,
                    count_b - count_a,
                    bytes_b - bytes_a,
                ]
            )


def _resolve_plot_paths(path: str) -> List[str]:
    base, ext = os.path.splitext(path)
    if ext.lower() in (".png", ".pdf"):
        base = base or path
    else:
        base = path
    return [f"{base}.png", f"{base}.pdf"]


def _plot_differences(
    paths: List[str],
    rows: List[Tuple[str, int, int, int, int]],
    top_k: int,
    title: Optional[str],
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required for plotting. Please install it first.")
        return False

    diffs = [(name, count_a, count_b, count_b - count_a) for name, count_a, _, count_b, _ in rows]
    nonzero = [item for item in diffs if item[3] != 0]
    if not nonzero:
        print("No non-zero differences found; skipping plot.")
        return False

    ranked = sorted(nonzero, key=lambda item: abs(item[3]), reverse=True)
    if top_k <= 0:
        top_k = min(20, len(ranked))
    ranked = ranked[:top_k]

    names = [name for name, _, _, _ in ranked]
    values_a = [count_a for _, count_a, _, _ in ranked]
    values_b = [count_b for _, _, count_b, _ in ranked]
    diff_values = [diff for _, _, _, diff in ranked]
    colors_a = "#64748b"
    colors_b = "#2563eb"

    total_a = sum(count_a for _, count_a, _, _ in diffs)
    total_b = sum(count_b for _, _, count_b, _ in diffs)
    total_diff = total_b - total_a
    inc = sum(1 for _, _, _, diff in diffs if diff > 0)
    dec = sum(1 for _, _, _, diff in diffs if diff < 0)
    total_abs = sum(abs(diff) for _, _, _, diff in diffs)
    top_abs = sum(abs(diff) for diff in diff_values)
    coverage = (top_abs / total_abs) if total_abs else 0.0

    height = max(4.0, 0.45 * len(names) + 2.6)
    fig, ax = plt.subplots(figsize=(12, height))
    y_pos = list(range(len(names)))
    bar_height = 0.38
    offset = bar_height / 2
    y_pos_a = [pos - offset for pos in y_pos]
    y_pos_b = [pos + offset for pos in y_pos]
    ax.barh(y_pos_a, values_a, height=bar_height, color=colors_a, label="A")
    ax.barh(y_pos_b, values_b, height=bar_height, color=colors_b, label="B")
    ax.axvline(0, color="#111827", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Gaussians (A vs B)")
    ax.set_title(f"Top {len(names)} by |diff| (gaussians)")
    ax.legend(loc="lower right", frameon=False)
    ax.ticklabel_format(style="plain", axis="x")

    max_val = max(values_a + values_b) if values_a or values_b else 1
    label_offset = max(1, int(max_val * 0.01))
    for idx, diff in enumerate(diff_values):
        anchor = max(values_a[idx], values_b[idx])
        ax.text(anchor + label_offset, idx, f"Δ {diff:+,}", va="center", ha="left", fontsize=9)

    summary = (
        f"Total A {total_a:,} | Total B {total_b:,} | Diff {total_diff:+,} | "
        f"Changed {inc + dec}/{len(rows)} | Top {len(names)} covers {coverage:.1%} of |diff|"
    )
    fig.suptitle(title or "Model Differences (B - A)")
    fig.text(0.5, 0.01, summary, ha="center", fontsize=9)
    fig.tight_layout(rect=[0.04, 0.05, 0.98, 0.92])

    for output in paths:
        output_path = os.path.abspath(output)
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        print(f"Plot saved to: {output_path}")
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two MTGS checkpoints (gaussian stats).")
    parser.add_argument("--ckpt-a", required=True, help="Path to checkpoint A.")
    parser.add_argument("--ckpt-b", required=True, help="Path to checkpoint B.")
    parser.add_argument("--output", default="ckpt_compare_report.txt", help="Path to write report.")
    group_by_id_group = parser.add_mutually_exclusive_group()
    group_by_id_group.add_argument(
        "--group-by-id",
        dest="group_by_id",
        action="store_true",
        help="Merge rigid/bezier object models by id (rigid_object_* and bezier_rigid_object_*).",
    )
    group_by_id_group.add_argument(
        "--no-group-by-id",
        dest="group_by_id",
        action="store_false",
        help="Do not merge object models; compare raw model names.",
    )
    parser.set_defaults(group_by_id=True)
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Show top +/- diff items with ASCII bars (0 disables). Also used for plot top-k.",
    )
    parser.add_argument("--csv", default=None, help="Optional CSV output path.")
    parser.add_argument(
        "--plot",
        default=None,
        help="Optional plot base path; writes both .png and .pdf.",
    )
    parser.add_argument("--plot-title", default=None, help="Optional plot title.")
    args = parser.parse_args()

    state_a = _load_state(args.ckpt_a)
    state_b = _load_state(args.ckpt_b)

    models_a = _collect_gaussian_models(state_a)
    models_b = _collect_gaussian_models(state_b)

    if not models_a or not models_b:
        raise ValueError("No gaussian model parameters found in one of the checkpoints.")

    stats_a, sources_a = _collect_stats(models_a, args.group_by_id)
    stats_b, sources_b = _collect_stats(models_b, args.group_by_id)
    merged_sources: Dict[str, Set[str]] = {}
    if args.group_by_id:
        _merge_sources(merged_sources, sources_a)
        _merge_sources(merged_sources, sources_b)

    report_lines = []
    report_lines.append(f"Checkpoint A: {args.ckpt_a}")
    report_lines.append(f"Checkpoint B: {args.ckpt_b}")
    report_lines.append("")

    size_a = os.path.getsize(args.ckpt_a)
    size_b = os.path.getsize(args.ckpt_b)
    report_lines.append(f"File size A: {_format_bytes(size_a)}")
    report_lines.append(f"File size B: {_format_bytes(size_b)}")
    if args.group_by_id:
        merged_groups = sum(1 for names in merged_sources.values() if len(names) > 1)
        report_lines.append(
            "Grouped by object id: "
            f"{merged_groups} merged groups "
            "(rigid_object_* + bezier_rigid_object_* -> object_*)"
        )
    report_lines.append("")

    total_gauss_a = 0
    total_gauss_b = 0
    total_bytes_a = 0
    total_bytes_b = 0

    all_models = sorted(set(stats_a.keys()) | set(stats_b.keys()))
    csv_rows: List[Tuple[str, int, int, int, int]] = []
    for model_name in all_models:
        count_a, bytes_a = stats_a.get(model_name, (0, 0))
        count_b, bytes_b = stats_b.get(model_name, (0, 0))

        total_gauss_a += count_a
        total_gauss_b += count_b
        total_bytes_a += bytes_a
        total_bytes_b += bytes_b

        report_lines.append(_format_line(f"{model_name} [A]", count_a, bytes_a))
        report_lines.append(_format_line(f"{model_name} [B]", count_b, bytes_b))
        report_lines.append(f"diff: {count_b - count_a:+,} gaussians")
        report_lines.append("")
        csv_rows.append((model_name, count_a, bytes_a, count_b, bytes_b))

    report_lines.append("Totals")
    report_lines.append(_format_line("A", total_gauss_a, total_bytes_a))
    report_lines.append(_format_line("B", total_gauss_b, total_bytes_b))
    report_lines.append(f"diff: {total_gauss_b - total_gauss_a:+,} gaussians")
    report_lines.append("")

    if args.top_k > 0:
        diffs = [
            (name, count_b - count_a)
            for name, count_a, _, count_b, _ in csv_rows
        ]
        max_abs = max((abs(diff) for _, diff in diffs), default=0)
        top_pos = sorted((item for item in diffs if item[1] > 0), key=lambda x: x[1], reverse=True)
        top_neg = sorted((item for item in diffs if item[1] < 0), key=lambda x: x[1])

        report_lines.append(f"Top +diff (gaussians), k={args.top_k}")
        for name, diff in top_pos[: args.top_k]:
            report_lines.append(f"{name}: {diff:+,} |{_format_bar(diff, max_abs)}")
        if not top_pos:
            report_lines.append("None")
        report_lines.append("")

        report_lines.append(f"Top -diff (gaussians), k={args.top_k}")
        for name, diff in top_neg[: args.top_k]:
            report_lines.append(f"{name}: {diff:+,} |{_format_bar(diff, max_abs)}")
        if not top_neg:
            report_lines.append("None")
        report_lines.append("")

    report = "\n".join(report_lines)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)

    if args.csv is not None:
        _write_csv(args.csv, csv_rows)
        print(f"CSV saved to: {args.csv}")

    if args.plot is not None:
        plot_paths = _resolve_plot_paths(args.plot)
        _plot_differences(plot_paths, csv_rows, args.top_k, args.plot_title)

    print(report)
    print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
import argparse
import os
from typing import Dict, Tuple, Optional

import numpy as np
import torch


def _load_checkpoint(path: str) -> Tuple[dict, dict, bool]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "pipeline" in ckpt:
        return ckpt, ckpt["pipeline"], True
    if isinstance(ckpt, dict):
        return ckpt, ckpt, False
    raise ValueError(f"Unsupported checkpoint format: {type(ckpt)}")


def _collect_models(state: dict) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[Tuple[str, str], str]]:
    models: Dict[str, Dict[str, torch.Tensor]] = {}
    key_map: Dict[Tuple[str, str], str] = {}
    for key, value in state.items():
        if ".gaussian_models." not in key or ".gauss_params." not in key:
            continue
        _, rest = key.split(".gaussian_models.", 1)
        if ".gauss_params." not in rest:
            continue
        model_key, param_name = rest.split(".gauss_params.", 1)
        models.setdefault(model_key, {})[param_name] = value
        key_map[(model_key, param_name)] = key
    return models, key_map


def _prune_params(
    params: Dict[str, torch.Tensor],
    opacity_thresh: float,
    scale_min: Optional[float],
    scale_max: Optional[float],
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    if "means" not in params or "opacities" not in params or "scales" not in params:
        return params, torch.ones(0, dtype=torch.bool)

    means = params["means"]
    opacities = params["opacities"]
    scales = params["scales"]

    num_points = means.shape[0]
    mask = torch.ones(num_points, dtype=torch.bool)

    alpha = torch.sigmoid(opacities).squeeze(-1)
    mask &= alpha >= opacity_thresh

    scale_vals = torch.exp(scales)
    if scale_vals.dim() == 1:
        scale_max_vals = scale_vals
    else:
        scale_max_vals = scale_vals.max(dim=-1).values

    if scale_min is not None:
        mask &= scale_max_vals >= scale_min
    if scale_max is not None:
        mask &= scale_max_vals <= scale_max

    if mask.sum() == 0:
        # keep the most opaque gaussian to avoid empty models
        keep_idx = torch.argmax(alpha).item()
        mask[keep_idx] = True

    pruned: Dict[str, torch.Tensor] = {}
    for name, tensor in params.items():
        if tensor.shape[0] == num_points:
            pruned[name] = tensor[mask]
        else:
            pruned[name] = tensor
    return pruned, mask


def _resolve_features(params: Dict[str, torch.Tensor], travel_mode: str) -> Tuple[torch.Tensor, torch.Tensor]:
    features_dc = params.get("features_dc")
    features_rest = params.get("features_rest")
    if features_dc is None or features_rest is None:
        raise ValueError("Missing features_dc/features_rest in gauss_params.")

    if features_dc.dim() == 3:
        features_dc = features_dc[:, 0, :]

    if "features_adapters" in params:
        adapters = params["features_adapters"]
        if travel_mode == "mean":
            features_dc = features_dc + adapters.mean(dim=1)
        elif travel_mode == "first":
            features_dc = features_dc + adapters[:, 0, :]

    if features_rest.dim() == 4:
        if travel_mode == "mean":
            features_rest = features_rest.mean(dim=1)
        else:
            features_rest = features_rest[:, 0, :, :]

    return features_dc, features_rest


def _write_ply(
    path: str,
    xyz: np.ndarray,
    features_dc: np.ndarray,
    features_rest: np.ndarray,
    opacities: np.ndarray,
    scales: np.ndarray,
    quats: np.ndarray,
) -> None:
    num_points = xyz.shape[0]
    if features_rest.ndim == 3:
        features_rest = features_rest.reshape(num_points, -1)

    rest_dim = features_rest.shape[1] if features_rest.size > 0 else 0

    fields = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("f_dc_0", "f4"),
        ("f_dc_1", "f4"),
        ("f_dc_2", "f4"),
    ]
    fields += [(f"f_rest_{i}", "f4") for i in range(rest_dim)]
    fields += [
        ("opacity", "f4"),
        ("scale_0", "f4"),
        ("scale_1", "f4"),
        ("scale_2", "f4"),
        ("rot_0", "f4"),
        ("rot_1", "f4"),
        ("rot_2", "f4"),
        ("rot_3", "f4"),
    ]

    data = np.empty(num_points, dtype=fields)
    data["x"] = xyz[:, 0]
    data["y"] = xyz[:, 1]
    data["z"] = xyz[:, 2]
    data["nx"] = 0.0
    data["ny"] = 0.0
    data["nz"] = 0.0
    data["f_dc_0"] = features_dc[:, 0]
    data["f_dc_1"] = features_dc[:, 1]
    data["f_dc_2"] = features_dc[:, 2]
    for i in range(rest_dim):
        data[f"f_rest_{i}"] = features_rest[:, i]
    data["opacity"] = opacities
    data["scale_0"] = scales[:, 0]
    data["scale_1"] = scales[:, 1]
    data["scale_2"] = scales[:, 2]
    data["rot_0"] = quats[:, 0]
    data["rot_1"] = quats[:, 1]
    data["rot_2"] = quats[:, 2]
    data["rot_3"] = quats[:, 3]

    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {num_points}",
        "property float x",
        "property float y",
        "property float z",
        "property float nx",
        "property float ny",
        "property float nz",
        "property float f_dc_0",
        "property float f_dc_1",
        "property float f_dc_2",
    ]
    header += [f"property float f_rest_{i}" for i in range(rest_dim)]
    header += [
        "property float opacity",
        "property float scale_0",
        "property float scale_1",
        "property float scale_2",
        "property float rot_0",
        "property float rot_1",
        "property float rot_2",
        "property float rot_3",
        "end_header",
    ]

    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "wb") as f:
        f.write("\n".join(header).encode("ascii") + b"\n")
        data.tofile(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prune MTGS gaussian parameters and export PLY.")
    parser.add_argument("--checkpoint", required=True, help="Path to .ckpt or state dict .pth.")
    parser.add_argument("--output-ckpt", default=None, help="Optional path to save pruned checkpoint.")
    parser.add_argument("--output-ply", required=True, help="Path to output PLY.")
    parser.add_argument("--opacity-thresh", type=float, default=0.005)
    parser.add_argument("--scale-min", type=float, default=None)
    parser.add_argument("--scale-max", type=float, default=None)
    parser.add_argument("--travel-mode", choices=["mean", "first", "raw"], default="mean")
    parser.add_argument("--strip-optim", action="store_true", help="Drop optimizer/scheduler states when saving ckpt.")
    parser.add_argument("--per-model", action="store_true", help="Also export per-model PLYs.")
    args = parser.parse_args()

    ckpt, state, is_full = _load_checkpoint(args.checkpoint)
    models, key_map = _collect_models(state)

    if len(models) == 0:
        sample_keys = list(state.keys())[:20]
        raise ValueError(
            "No gaussian model parameters found in checkpoint. "
            f"Sample keys: {sample_keys}"
        )

    pruned_state = {}
    masks = {}
    for model_name, params in models.items():
        num_points = params["means"].shape[0]
        pruned_params, mask = _prune_params(
            params,
            opacity_thresh=args.opacity_thresh,
            scale_min=args.scale_min,
            scale_max=args.scale_max,
        )
        masks[model_name] = mask
        print(f"[prune] {model_name}: kept {int(mask.sum())}/{num_points} gaussians")
        for param_name, tensor in pruned_params.items():
            key = key_map.get((model_name, param_name))
            if key is None:
                continue
            pruned_state[key] = tensor

    for key in list(state.keys()):
        if key in pruned_state:
            state[key] = pruned_state[key]

    # Export PLY
    all_xyz = []
    all_f_dc = []
    all_f_rest = []
    all_opacities = []
    all_scales = []
    all_quats = []
    max_rest_dim = 0

    for model_name, params in models.items():
        mask = masks[model_name]
        if mask.numel() == 0:
            continue
        means = params["means"][mask]
        opacities = params["opacities"][mask].squeeze(-1)
        scales = params["scales"][mask]
        if scales.dim() == 1 or scales.shape[1] == 1:
            scales = scales.repeat(1, 3)

        if "quats" in params:
            quats = params["quats"]
            if quats.shape[0] == params["means"].shape[0]:
                quats = quats[mask]
            if quats.shape[1] != 4:
                quats = torch.zeros((means.shape[0], 4))
        else:
            quats = torch.zeros((means.shape[0], 4))

        features_dc, features_rest = _resolve_features(params, args.travel_mode)
        features_dc = features_dc[mask]
        features_rest = features_rest[mask]

        rest_dim = features_rest.shape[1] * features_rest.shape[2]
        max_rest_dim = max(max_rest_dim, rest_dim)

        all_xyz.append(means)
        all_f_dc.append(features_dc)
        all_f_rest.append(features_rest.reshape(features_rest.shape[0], -1))
        all_opacities.append(opacities)
        all_scales.append(scales)
        all_quats.append(quats)

        if args.per_model:
            base, ext = os.path.splitext(args.output_ply)
            ply_path = f"{base}_{model_name}{ext or '.ply'}"
            _write_ply(
                ply_path,
                means.cpu().numpy(),
                features_dc.cpu().numpy(),
                features_rest.cpu().numpy(),
                opacities.cpu().numpy(),
                scales.cpu().numpy(),
                quats.cpu().numpy(),
            )

    if len(all_xyz) == 0:
        raise ValueError("No gaussians left after pruning.")

    xyz = torch.cat(all_xyz, dim=0).cpu().numpy()
    f_dc = torch.cat(all_f_dc, dim=0).cpu().numpy()
    f_rest = torch.cat(all_f_rest, dim=0).cpu().numpy()
    if f_rest.shape[1] < max_rest_dim:
        pad = np.zeros((f_rest.shape[0], max_rest_dim - f_rest.shape[1]), dtype=f_rest.dtype)
        f_rest = np.concatenate([f_rest, pad], axis=1)
    f_rest = f_rest.reshape(f_rest.shape[0], -1, 3)
    opacities = torch.cat(all_opacities, dim=0).cpu().numpy()
    scales = torch.cat(all_scales, dim=0).cpu().numpy()
    quats = torch.cat(all_quats, dim=0).cpu().numpy()

    _write_ply(args.output_ply, xyz, f_dc, f_rest, opacities, scales, quats)

    if args.output_ckpt:
        if is_full and args.strip_optim:
            ckpt.pop("optimizers", None)
            ckpt.pop("schedulers", None)
            ckpt.pop("scalers", None)
        torch.save(ckpt, args.output_ckpt)


if __name__ == "__main__":
    main()

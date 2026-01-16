#!/usr/bin/env python3
import torch

from mtgs.scene_model.gaussian_model.vanilla_gaussian_splatting import (
    GaussianSplattingControlConfig,
    VanillaGaussianSplattingModel,
    VanillaGaussianSplattingModelConfig,
)


def _build_model(scales: torch.Tensor, clone_means: bool = True) -> VanillaGaussianSplattingModel:
    control = GaussianSplattingControlConfig(scale_dim=3, sh_degree=0, clone_sample_means=clone_means)
    config = VanillaGaussianSplattingModelConfig(control=control, model_type="vanilla")
    model = VanillaGaussianSplattingModel(config, model_name="test", model_id=0)
    n = scales.shape[0]
    quats = torch.zeros(n, 4)
    quats[:, 0] = 1.0
    model.gauss_params = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(torch.zeros(n, 3)),
            "scales": torch.nn.Parameter(scales),
            "quats": torch.nn.Parameter(quats),
            "features_dc": torch.nn.Parameter(torch.zeros(n, 3)),
            "features_rest": torch.nn.Parameter(torch.zeros(n, 1, 3)),
            "opacities": torch.nn.Parameter(torch.zeros(n, 1)),
        }
    )
    return model


def _expected_scales_shape(src: torch.Tensor, n_out: int) -> torch.Size:
    if src.dim() == 2:
        return torch.Size([n_out, 3])
    if src.dim() == 3:
        return torch.Size([n_out, src.shape[1], src.shape[2]])
    raise AssertionError(f"Unexpected source scales dim: {src.dim()}")


def _run_split(model: VanillaGaussianSplattingModel, split_mask: torch.Tensor, samps: int) -> None:
    out = model.split_gaussians(split_mask, samps)
    expected = int(split_mask.sum().item()) * samps
    for name, param in model.gauss_params.items():
        assert out[name].shape[0] == expected
        if name == "scales":
            assert out[name].shape == _expected_scales_shape(param, expected)


def _run_dup(model: VanillaGaussianSplattingModel, dup_mask: torch.Tensor) -> None:
    out = model.dup_gaussians(dup_mask)
    expected = int(dup_mask.sum().item())
    for name, param in model.gauss_params.items():
        assert out[name].shape[0] == expected
        if name == "scales":
            assert out[name].shape == _expected_scales_shape(param, expected)


def main() -> None:
    shapes = [
        torch.zeros(8, 3),
        torch.zeros(8, 1, 3),
        torch.zeros(8, 3, 1),
    ]
    masks = [
        torch.tensor([True, False, True, False, True, False, True, False]),
        torch.zeros(8, dtype=torch.bool),
        torch.ones(8, dtype=torch.bool),
    ]

    for scales in shapes:
        for clone_means in (True, False):
            model = _build_model(scales.clone(), clone_means=clone_means)
            for mask in masks:
                _run_split(model, mask, samps=2)
                _run_dup(model, mask)
            with torch.enable_grad():
                _run_split(model, masks[0], samps=2)
                _run_dup(model, masks[0])

    print("OK: strict splatting tests passed (split/dup, shapes, empty masks, grad on).")


if __name__ == "__main__":
    main()

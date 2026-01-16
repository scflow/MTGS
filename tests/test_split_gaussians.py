#!/usr/bin/env python3
import torch

from mtgs.scene_model.gaussian_model.vanilla_gaussian_splatting import (
    GaussianSplattingControlConfig,
    VanillaGaussianSplattingModel,
    VanillaGaussianSplattingModelConfig,
)


def _build_model(scale_shape: str) -> VanillaGaussianSplattingModel:
    control = GaussianSplattingControlConfig(scale_dim=3, sh_degree=0)
    config = VanillaGaussianSplattingModelConfig(control=control, model_type="vanilla")
    model = VanillaGaussianSplattingModel(config, model_name="test", model_id=0)

    n = 4
    means = torch.zeros(n, 3)
    if scale_shape == "2d":
        scales = torch.zeros(n, 3)
    elif scale_shape == "n1x3":
        scales = torch.zeros(n, 1, 3)
    elif scale_shape == "n3x1":
        scales = torch.zeros(n, 3, 1)
    else:
        raise ValueError(f"Unknown scale_shape: {scale_shape}")

    quats = torch.zeros(n, 4)
    quats[:, 0] = 1.0
    features_dc = torch.zeros(n, 3)
    features_rest = torch.zeros(n, 1, 3)
    opacities = torch.zeros(n, 1)

    model.gauss_params = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(means),
            "scales": torch.nn.Parameter(scales),
            "quats": torch.nn.Parameter(quats),
            "features_dc": torch.nn.Parameter(features_dc),
            "features_rest": torch.nn.Parameter(features_rest),
            "opacities": torch.nn.Parameter(opacities),
        }
    )
    return model


def _assert_split_shapes(model: VanillaGaussianSplattingModel, scale_shape: str) -> None:
    split_mask = torch.tensor([True, False, True, False])
    samps = 2
    out = model.split_gaussians(split_mask, samps)
    expected = int(split_mask.sum().item()) * samps
    assert out["means"].shape[0] == expected

    scales_out = out["scales"]
    if scale_shape == "2d":
        assert scales_out.shape == (expected, 3)
    elif scale_shape == "n1x3":
        assert scales_out.shape == (expected, 1, 3)
    elif scale_shape == "n3x1":
        assert scales_out.shape == (expected, 3, 1)


def _assert_empty_split(model: VanillaGaussianSplattingModel) -> None:
    split_mask = torch.zeros(4, dtype=torch.bool)
    out = model.split_gaussians(split_mask, 2)
    for name, param in model.gauss_params.items():
        assert out[name].shape[0] == 0
        assert out[name].shape[1:] == param.shape[1:]


def main() -> None:
    for shape in ("2d", "n1x3", "n3x1"):
        model = _build_model(shape)
        _assert_split_shapes(model, shape)
        _assert_empty_split(model)
    print("OK: split_gaussians handles empty split and scales shapes (2d/n1x3/n3x1).")


if __name__ == "__main__":
    main()

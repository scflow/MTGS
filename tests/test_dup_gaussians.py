#!/usr/bin/env python3
import torch

from mtgs.scene_model.gaussian_model.vanilla_gaussian_splatting import (
    GaussianSplattingControlConfig,
    VanillaGaussianSplattingModel,
    VanillaGaussianSplattingModelConfig,
)


def _build_model(clone_means: bool) -> VanillaGaussianSplattingModel:
    control = GaussianSplattingControlConfig(scale_dim=3, sh_degree=0, clone_sample_means=clone_means)
    config = VanillaGaussianSplattingModelConfig(control=control, model_type="vanilla")
    model = VanillaGaussianSplattingModel(config, model_name="test", model_id=0)
    n = 4
    means = torch.zeros(n, 3)
    scales = torch.zeros(n, 3)
    quats = torch.zeros(n, 4)
    quats[:, 0] = 1.0
    model.gauss_params = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(means),
            "scales": torch.nn.Parameter(scales),
            "quats": torch.nn.Parameter(quats),
            "features_dc": torch.nn.Parameter(torch.zeros(n, 3)),
            "features_rest": torch.nn.Parameter(torch.zeros(n, 1, 3)),
            "opacities": torch.nn.Parameter(torch.zeros(n, 1)),
        }
    )
    return model


def _assert_dup(model: VanillaGaussianSplattingModel) -> None:
    dup_mask = torch.tensor([True, False, True, False])
    out = model.dup_gaussians(dup_mask)
    expected = int(dup_mask.sum().item())
    assert out["means"].shape[0] == expected
    assert out["scales"].shape[0] == expected
    assert out["quats"].shape[0] == expected


def _assert_empty_dup(model: VanillaGaussianSplattingModel) -> None:
    dup_mask = torch.zeros(4, dtype=torch.bool)
    out = model.dup_gaussians(dup_mask)
    for name, param in model.gauss_params.items():
        assert out[name].shape[0] == 0
        assert out[name].shape[1:] == param.shape[1:]


def main() -> None:
    for clone in (False, True):
        model = _build_model(clone)
        _assert_dup(model)
        _assert_empty_dup(model)
    print("OK: dup_gaussians handles clone_sample_means and empty masks.")


if __name__ == "__main__":
    main()

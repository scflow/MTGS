#!/usr/bin/env python3
import torch

from mtgs.scene_model.gaussian_model.vanilla_gaussian_splatting import (
    GaussianSplattingControlConfig,
    VanillaGaussianSplattingModel,
    VanillaGaussianSplattingModelConfig,
)


def _build_model(scales: torch.Tensor) -> VanillaGaussianSplattingModel:
    control = GaussianSplattingControlConfig(scale_dim=3, sh_degree=0)
    config = VanillaGaussianSplattingModelConfig(control=control, model_type="vanilla")
    model = VanillaGaussianSplattingModel(config, model_name="test", model_id=0)
    n = scales.shape[0]
    model.gauss_params = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(torch.zeros(n, 3)),
            "scales": torch.nn.Parameter(scales),
            "quats": torch.nn.Parameter(torch.zeros(n, 4)),
            "features_dc": torch.nn.Parameter(torch.zeros(n, 3)),
            "features_rest": torch.nn.Parameter(torch.zeros(n, 1, 3)),
            "opacities": torch.nn.Parameter(torch.zeros(n, 1)),
        }
    )
    return model


def main() -> None:
    n = 4
    for scales in (
        torch.zeros(n, 3),
        torch.zeros(n, 1, 3),
        torch.zeros(n, 3, 1),
    ):
        model = _build_model(scales)
        out = model.scales
        assert out.shape == (n, 3)

    invalid = torch.zeros(n, 2, 2)
    model = _build_model(invalid)
    try:
        _ = model.scales
        raise AssertionError("Expected ValueError for invalid scales shape")
    except ValueError:
        pass
    print("OK: scales property normalizes [N,3]/[N,1,3]/[N,3,1] and rejects invalid shapes.")


if __name__ == "__main__":
    main()

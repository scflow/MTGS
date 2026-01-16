#!/usr/bin/env python3
import types
import torch

from mtgs.scene_model.mtgs_scene_graph import MTGSSceneModel


class CountingLPIPS:
    def __init__(self):
        self.calls = 0

    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return (a - b).abs().mean(dim=(1, 2, 3))


def _build_stub_model() -> MTGSSceneModel:
    model = MTGSSceneModel.__new__(MTGSSceneModel)
    model.config = types.SimpleNamespace(
        vehicle_lpips_pad=0,
        vehicle_lpips_resize=4,
        vehicle_lpips_min_area=1,
    )
    model.lpips = CountingLPIPS()
    model.step = 10
    return model


def main() -> None:
    model = _build_stub_model()
    gt = torch.rand(1, 3, 8, 8)
    pred = torch.rand(1, 3, 8, 8)
    sem = torch.zeros(8, 8, 1, dtype=torch.uint8)
    sem[0:2, 0:2, 0] = 13
    sem[6:8, 6:8, 0] = 13
    mask = torch.ones(8, 8, dtype=torch.bool)
    batch = {"image_idx": torch.tensor(1)}
    key = model._vehicle_lpips_cache_key(batch)

    mean_tensor, count, _ = model._get_vehicle_lpips_cached(
        gt, pred, sem, valid_mask=mask, require_grad=True, cache_key=key
    )
    assert count == 2
    assert mean_tensor is not None
    first_calls = model.lpips.calls

    mean_value, count, _ = model._get_vehicle_lpips_cached(
        gt, pred, sem, valid_mask=mask, require_grad=False, cache_key=key
    )
    assert isinstance(mean_value, float)
    assert count == 2
    assert model.lpips.calls == first_calls

    model.step = 11
    key2 = model._vehicle_lpips_cache_key(batch)
    _ = model._get_vehicle_lpips_cached(
        gt, pred, sem, valid_mask=mask, require_grad=False, cache_key=key2
    )
    assert model.lpips.calls > first_calls
    print("OK: vehicle LPIPS cache shares results between loss/metrics.")


if __name__ == "__main__":
    main()

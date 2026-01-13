#!/usr/bin/env python3
import argparse
import os
import random
import subprocess
import sys
from pathlib import Path
import shutil

import cv2
import numpy as np
from PIL import Image
from pyquaternion import Quaternion

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from nuplan_scripts.utils.config import load_config
from nuplan_scripts.utils.video_scene_dict_tools import VideoScene

VEHICLE_LABELS = (13, 14, 15)  # car, truck, bus


def build_ego_to_cam(cam_info):
    rotation = Quaternion(cam_info["sensor2ego_rotation"]).rotation_matrix.T
    translation = -rotation @ np.array(cam_info["sensor2ego_translation"], dtype=np.float32)
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def box_corners_ego(box):
    x, y, z, length, width, height, yaw = box
    dx = length / 2.0
    dy = width / 2.0
    dz = height / 2.0

    corners = np.array(
        [
            [dx, dy, dz],
            [dx, -dy, dz],
            [-dx, -dy, dz],
            [-dx, dy, dz],
            [dx, dy, -dz],
            [dx, -dy, -dz],
            [-dx, -dy, -dz],
            [-dx, dy, -dz],
        ],
        dtype=np.float32,
    )
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    rot = np.array(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    corners = (rot @ corners.T).T
    corners += np.array([x, y, z], dtype=np.float32)
    return corners


def project_corners(corners_ego, ego_to_cam, intrinsic, distortion):
    pts_h = np.hstack([corners_ego, np.ones((corners_ego.shape[0], 1), dtype=np.float32)])
    pts_cam = (ego_to_cam @ pts_h.T).T[:, :3]
    in_front = pts_cam[:, 2] > 0.1
    if in_front.sum() < 4:
        return None

    pts_cam = pts_cam.astype(np.float32)
    rvec = np.zeros(3, dtype=np.float32)
    tvec = np.zeros(3, dtype=np.float32)
    image_points, _ = cv2.projectPoints(pts_cam, rvec, tvec, intrinsic, distortion)
    image_points = image_points.reshape(-1, 2)
    if not np.isfinite(image_points).all():
        return None
    return image_points


def apply_mask_postprocess(mask, dilate_radius, blur_ksize):
    if dilate_radius > 0:
        ksize = 2 * dilate_radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        mask = cv2.dilate(mask, kernel, iterations=1)
    if blur_ksize > 0:
        if blur_ksize % 2 == 0:
            blur_ksize += 1
        mask = cv2.GaussianBlur(mask, (blur_ksize, blur_ksize), 0)
        mask = (mask > 0).astype(np.uint8) * 255
    return mask


def build_vehicle_mask(frame_info, cam_info, image_shape, refine_with_semantic, video_scene, dilate_radius, blur_ksize):
    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    gt_boxes = frame_info.get("gt_boxes")
    gt_names = frame_info.get("gt_names")
    if gt_boxes is None or gt_names is None:
        return mask

    intrinsic = np.array(cam_info["cam_intrinsic"], dtype=np.float32)
    distortion = np.array(cam_info.get("distortion", [0, 0, 0, 0, 0]), dtype=np.float32)
    ego_to_cam = build_ego_to_cam(cam_info)

    for box, name in zip(gt_boxes, gt_names):
        if name != "vehicle":
            continue
        corners = box_corners_ego(box)
        image_points = project_corners(corners, ego_to_cam, intrinsic, distortion)
        if image_points is None:
            continue
        hull = cv2.convexHull(image_points.astype(np.float32))
        if hull is None or len(hull) == 0 or not np.isfinite(hull).all():
            continue
        cv2.fillConvexPoly(mask, hull.astype(np.int32), 255)

    if refine_with_semantic:
        sem_path = os.path.join(
            video_scene.raw_mask_path,
            video_scene.mask_suffix_cityscape,
            cam_info["data_path"],
        ).replace(".jpg", ".png")
        if os.path.exists(sem_path):
            sem = cv2.imread(sem_path, cv2.IMREAD_GRAYSCALE)
            sem_vehicle = np.isin(sem, VEHICLE_LABELS)
            mask = np.logical_and(mask > 0, sem_vehicle).astype(np.uint8) * 255
    mask = apply_mask_postprocess(mask, dilate_radius, blur_ksize)
    return mask


def select_samples(video_scene_dict, num_samples, camera, seed):
    rng = random.Random(seed)
    candidates = []
    for video_token, video_info in video_scene_dict.items():
        for frame_info in video_info["frame_infos"]:
            if frame_info.get("skipped", False):
                continue
            for cam_name in frame_info["cams"]:
                if camera and cam_name != camera:
                    continue
                candidates.append((video_token, frame_info, cam_name))
    if not candidates:
        raise RuntimeError("No valid frames found.")
    if num_samples >= len(candidates):
        return candidates
    return rng.sample(candidates, num_samples)


def run_lama(
    lama_dir,
    lama_python,
    model_path,
    input_dir,
    output_dir,
    device,
    refine,
    refiner_gpu_ids,
    refiner_n_iters,
    refiner_lr,
    refiner_min_side,
    refiner_max_scales,
    refiner_px_budget,
):
    env = os.environ.copy()
    env["PYTHONPATH"] = lama_dir
    env["TORCH_HOME"] = lama_dir
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)
    cmd = [
        lama_python,
        os.path.join(lama_dir, "bin", "predict.py"),
        f"model.path={model_path}",
        f"indir={input_dir}",
        f"outdir={output_dir}",
        f"device={device}",
    ]
    if refine:
        cmd.append("refine=True")
        if refiner_gpu_ids:
            cmd.append(f"refiner.gpu_ids='{refiner_gpu_ids}'")
        if refiner_n_iters is not None:
            cmd.append(f"refiner.n_iters={refiner_n_iters}")
        if refiner_lr is not None:
            cmd.append(f"refiner.lr={refiner_lr}")
        if refiner_min_side is not None:
            cmd.append(f"refiner.min_side={refiner_min_side}")
        if refiner_max_scales is not None:
            cmd.append(f"refiner.max_scales={refiner_max_scales}")
        if refiner_px_budget is not None:
            cmd.append(f"refiner.px_budget={refiner_px_budget}")
    subprocess.run(cmd, check=True, env=env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--road-block-config", required=True)
    parser.add_argument("--out-dir", default="outputs/lama_vehicle_box")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--camera", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--refine-with-semantic", action="store_true")
    parser.add_argument("--lama-dir", default="thirdparty/lama")
    parser.add_argument("--lama-model-path", default="thirdparty/lama/models/big-lama")
    parser.add_argument("--lama-python", default=sys.executable)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--refine", action="store_true")
    parser.add_argument("--refiner-gpu-ids", default="")
    parser.add_argument("--refiner-n-iters", type=int, default=None)
    parser.add_argument("--refiner-lr", type=float, default=None)
    parser.add_argument("--refiner-min-side", type=int, default=None)
    parser.add_argument("--refiner-max-scales", type=int, default=None)
    parser.add_argument("--refiner-px-budget", type=int, default=None)
    parser.add_argument("--mask-dilate", type=int, default=0)
    parser.add_argument("--mask-blur", type=int, default=0)
    parser.add_argument("--skip-inpaint", action="store_true")
    args = parser.parse_args()

    config = load_config(args.road_block_config)
    video_scene = VideoScene(config)
    video_scene_dict = video_scene.load_pickle(video_scene.pickle_path, verbose=False)

    input_dir = os.path.join(args.out_dir, "input")
    output_dir = os.path.join(args.out_dir, "output")
    mask_dir = os.path.join(args.out_dir, "mask")
    compare_dir = os.path.join(args.out_dir, "compare")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(compare_dir, exist_ok=True)

    samples = select_samples(video_scene_dict, args.num_samples, args.camera, args.seed)
    index_path = os.path.join(args.out_dir, "sources.txt")
    compare_records = []
    with open(index_path, "w", encoding="utf-8") as index_file:
        for idx, (video_token, frame_info, cam_name) in enumerate(samples):
            cam_info = frame_info["cams"][cam_name]
            image_path = os.path.join(video_scene.raw_image_path, cam_info["data_path"])
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Missing image: {image_path}")

            image = Image.open(image_path).convert("RGB")
            name = f"sample_{idx:03d}"
            image_out = os.path.join(input_dir, f"{name}.png")
            mask_out = os.path.join(input_dir, f"{name}_mask000.png")
            debug_mask_out = os.path.join(mask_dir, f"{name}_mask000.png")

            mask = build_vehicle_mask(
                frame_info,
                cam_info,
                image.size[::-1],
                args.refine_with_semantic,
                video_scene,
                args.mask_dilate,
                args.mask_blur,
            )

            image.save(image_out)
            Image.fromarray(mask).save(mask_out)
            Image.fromarray(mask).save(debug_mask_out)

            shutil.copyfile(image_out, os.path.join(compare_dir, f"{name}_input.png"))
            shutil.copyfile(mask_out, os.path.join(compare_dir, f"{name}_mask.png"))
            compare_records.append(name)

            index_file.write(f"{name} {video_token} {cam_info['data_path']}\n")

    if args.skip_inpaint:
        return

    lama_dir = os.path.abspath(args.lama_dir)
    model_path = os.path.abspath(args.lama_model_path)
    refiner_gpu_ids = args.refiner_gpu_ids
    if args.refine and not refiner_gpu_ids:
        if args.device.startswith("cuda"):
            device_id = args.device.split(":", 1)[1] if ":" in args.device else "0"
            refiner_gpu_ids = str(device_id)
        else:
            raise ValueError("Refine requires CUDA; set --device cuda[:id] and optionally --refiner-gpu-ids.")
    run_lama(
        lama_dir,
        args.lama_python,
        model_path,
        input_dir,
        output_dir,
        args.device,
        args.refine,
        refiner_gpu_ids,
        args.refiner_n_iters,
        args.refiner_lr,
        args.refiner_min_side,
        args.refiner_max_scales,
        args.refiner_px_budget,
    )

    for name in compare_records:
        inpaint_path = os.path.join(output_dir, f"{name}_mask000.png")
        if os.path.exists(inpaint_path):
            shutil.copyfile(inpaint_path, os.path.join(compare_dir, f"{name}_inpaint.png"))


if __name__ == "__main__":
    main()

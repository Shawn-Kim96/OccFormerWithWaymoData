#!/usr/bin/env python

import argparse
import cv2
import numpy as np
from pathlib import Path
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.parallel import collate, scatter, DataContainer
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

import projects.mmdet3d_plugin

WAYMO_COLORS = np.array([
    [255, 158, 0],    # 0 general_object
    [255, 99, 71],    # 1 vehicle
    [255, 140, 0],    # 2 pedestrian
    [255, 215, 0],    # 3 sign
    [138, 43, 226],   # 4 cyclist
    [255, 20, 147],   # 5 traffic_light
    [139, 69, 19],    # 6 pole
    [255, 165, 0],    # 7 construction_cone
    [0, 191, 255],    # 8 bicycle
    [148, 0, 211],    # 9 motorcycle
    [105, 105, 105],  # 10 building
    [34, 139, 34],    # 11 vegetation
    [85, 107, 47],    # 12 tree_trunk
    [128, 128, 128],  # 13 road
    [189, 183, 107],  # 14 walkable
    [220, 220, 220],  # 15 free
], dtype=np.float32) / 255.0


def parse_args():
    parser = argparse.ArgumentParser(description="Create a Waymo demo video (camera + BEV occupancy).")
    parser.add_argument("config", help="Path to config (Waymo OccFormer)")
    parser.add_argument("checkpoint", help="Path to checkpoint (.pth)")
    parser.add_argument(
        "--out",
        default="results/waymo_demo.mp4",
        help="Output video path (mp4)",
    )
    parser.add_argument(
        "--scene-id",
        type=int,
        default=None,
        help="Scene id to render (default: first scene in val set)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=150,
        help="Maximum frames to render from the chosen scene",
    )
    parser.add_argument(
        "--slice-idx",
        type=int,
        default=8,
        help="Height slice index for BEV visualization (0 = bottom)",
    )
    parser.add_argument("--fps", type=int, default=5, help="Video FPS")
    parser.add_argument("--device", default="cuda:0", help="Device for inference (e.g., cuda:0 or cpu)")
    return parser.parse_args()


def denormalize_images(imgs: torch.Tensor, mean, std):
    imgs = imgs.detach().cpu().numpy()
    imgs = np.transpose(imgs, (0, 2, 3, 1))  # (N, H, W, C)
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    imgs = imgs * std[None, None, None, :] + mean[None, None, None, :]
    imgs = np.clip(imgs, 0, 255).astype(np.uint8)
    return list(imgs)


def make_camera_grid(images, target_width=450):
    resized = []
    for i, img in enumerate(images):
        h, w = img.shape[:2]
        scale = target_width / w
        new_size = (target_width, int(h * scale))
        canvas = cv2.resize(img, new_size)
        cv2.putText(
            canvas,
            f"Cam {i}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        resized.append(canvas)

    while len(resized) < 6:
        blank = np.zeros_like(resized[0])
        resized.append(blank)

    row1 = np.concatenate(resized[:3], axis=1)
    row2 = np.concatenate(resized[3:6], axis=1)
    grid = np.concatenate([row1, row2], axis=0)
    return grid


def colorize_occ(slice_2d: np.ndarray) -> np.ndarray:
    slice_2d = np.clip(slice_2d, 0, len(WAYMO_COLORS) - 1).astype(np.int64)
    rgb = (WAYMO_COLORS[slice_2d] * 255).astype(np.uint8)
    return rgb


def unwrap_dc(obj):
    if isinstance(obj, DataContainer):
        return unwrap_dc(obj.data)
    if isinstance(obj, (list, tuple)):
        return type(obj)(unwrap_dc(o) for o in obj)
    if isinstance(obj, dict):
        return {k: unwrap_dc(v) for k, v in obj.items()}
    return obj


def resize_nearest_3d(arr: np.ndarray, target_shape) -> np.ndarray:
    src_h, src_w, src_d = arr.shape
    dst_h, dst_w, dst_d = target_shape

    hs = np.clip(np.round(np.linspace(0, src_h - 1, dst_h)).astype(int), 0, src_h - 1)
    ws = np.clip(np.round(np.linspace(0, src_w - 1, dst_w)).astype(int), 0, src_w - 1)
    ds = np.clip(np.round(np.linspace(0, src_d - 1, dst_d)).astype(int), 0, src_d - 1)

    return arr[np.ix_(hs, ws, ds)]


def pad_to_same_width(top: np.ndarray, bottom: np.ndarray, color=(0, 0, 0)) -> tuple:
    top_w = top.shape[1]
    bottom_w = bottom.shape[1]
    if top_w == bottom_w:
        return top, bottom

    def pad(img, target_w):
        if img.shape[1] == target_w:
            return img
        pad_w = target_w - img.shape[1]
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        return cv2.copyMakeBorder(
            img, 0, 0, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=color
        )

    target_w = max(top_w, bottom_w)
    return pad(top, target_w), pad(bottom, target_w)


def build_scene_indices(dataset, scene_id=None, max_frames=150):
    if hasattr(dataset, "data_infos_full"):
        infos = dataset.data_infos_full
    else:
        infos = dataset.data_infos

    indices = []
    for idx, info in enumerate(infos):
        sample_idx = info["image"]["image_idx"]
        scene = sample_idx % 1000000 // 1000
        if scene_id is None or scene == scene_id:
            indices.append(idx)
        if len(indices) >= max_frames:
            break

    if not indices:
        raise RuntimeError(
            f"No frames found for scene_id={scene_id}. "
            "Try a different scene id or drop the flag to use the first scene."
        )
    return indices


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    # Build dataset and model
    dataset = build_dataset(cfg.data.val)
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model.to(args.device).eval()

    scene_indices = build_scene_indices(dataset, args.scene_id, args.max_frames)
    scene_label = args.scene_id if args.scene_id is not None else "first available"
    print(f"Rendering up to {len(scene_indices)} frame(s) from scene {scene_label}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Video writer setup will be done after the first frame is composed
    writer = None

    mean = cfg.img_norm_cfg["mean"]
    std = cfg.img_norm_cfg["std"]

    written = 0
    for ds_idx in scene_indices:
        data = dataset[ds_idx]
        if data is None:
            continue

        # Visualization uses unwrapped objects; inference uses the original sample.
        vis_sample = unwrap_dc(data)
        imgs_cpu = vis_sample["img_inputs"][0]  # (Ncams, 3, H, W) float tensor
        gt_occ = vis_sample.get("gt_occ", None)

        # Prepare batch for inference
        batch = collate([data], samples_per_gpu=1)

        # mmcv.scatter expects integer GPU ids.
        if args.device.startswith("cuda"):
            try:
                dev_id = int(args.device.split(":")[1])
            except (IndexError, ValueError):
                dev_id = 0
            batch = scatter(batch, [dev_id])[0]
        else:
            batch = batch

        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **batch)

        pred_logits = result[0]["output_voxels"]  # [C, H, W, D]
        pred_occ = pred_logits.argmax(0).cpu().numpy()
        if gt_occ is not None:
            if hasattr(gt_occ, "cpu"):
                gt_occ_np = gt_occ.cpu().numpy()
            else:
                gt_occ_np = np.array(gt_occ)
        else:
            gt_occ_np = None

        # Align shapes (GT may be 200x200x16 while pred is 256x256x32)
        if gt_occ_np is not None and gt_occ_np.shape != pred_occ.shape:
            gt_occ_np = resize_nearest_3d(gt_occ_np, pred_occ.shape)

        # Visuals
        cam_imgs = denormalize_images(imgs_cpu, mean, std)
        cam_grid = make_camera_grid(cam_imgs)

        # Occupancy slices (flip y so ego-forward is up)
        slice_idx = max(0, min(args.slice_idx, pred_occ.shape[-1] - 1))
        pred_slice = np.flipud(pred_occ[:, :, slice_idx])
        pred_rgb = colorize_occ(pred_slice)

        if gt_occ_np is not None:
            gt_slice = np.flipud(gt_occ_np[:, :, slice_idx])
            gt_rgb = colorize_occ(gt_slice)
        else:
            gt_rgb = np.zeros_like(pred_rgb)

        bev_row = np.concatenate(
            [
                cv2.putText(pred_rgb.copy(), "Prediction", (12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA),
                cv2.putText(gt_rgb.copy(), "Ground Truth", (12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA),
            ],
            axis=1,
        )

        top, bottom = pad_to_same_width(cam_grid, bev_row)
        frame = np.concatenate([top, bottom], axis=0)

        if writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open video writer at {out_path}")

        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        written += 1
        print(f"Rendered {written}/{len(scene_indices)}", end="\r")

    if writer:
        writer.release()
    print(f"\nSaved demo to: {out_path}")


if __name__ == "__main__":
    main()

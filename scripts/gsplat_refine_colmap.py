"""Refine an existing ReSplat Gaussian PLY with gsplat and COLMAP cameras.

The script keeps the Gaussian count fixed: it optimizes the initialized
positions, scales, rotations, opacities, and SH colors against all registered
COLMAP images. It is intended as a lightweight post-process for the web UI.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from plyfile import PlyData
from tqdm import tqdm

from gsplat.rendering import rasterization

from infer_colmap import (
    camera_normalization,
    compute_target_shape,
    load_colmap_scene,
    select_context_views,
)
from src.model.ply_export import export_ply


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize a ReSplat PLY against all COLMAP images with gsplat."
    )
    parser.add_argument("--scene_path", type=Path, required=True)
    parser.add_argument("--sparse_dir", default="sparse")
    parser.add_argument("--images_dir", default="images")
    parser.add_argument("--init_ply", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--image_shape", type=int, nargs=2, default=None, metavar=("H", "W"))
    parser.add_argument("--max_resolution", type=int, default=960)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_context", type=int, default=None)
    parser.add_argument("--context_selection", choices=["fps", "uniform"], default="fps")
    parser.add_argument(
        "--pose_space",
        choices=["resplat_single", "global"],
        default="resplat_single",
        help=(
            "resplat_single normalizes COLMAP poses to the same middle context "
            "view as scripts/infer_colmap.py. global leaves poses unchanged."
        ),
    )
    parser.add_argument("--near", type=float, default=0.1)
    parser.add_argument("--far", type=float, default=1000.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--lr_means", type=float, default=1.6e-4)
    parser.add_argument("--lr_scales", type=float, default=5.0e-3)
    parser.add_argument("--lr_rotations", type=float, default=1.0e-3)
    parser.add_argument("--lr_opacities", type=float, default=5.0e-2)
    parser.add_argument("--lr_colors", type=float, default=2.5e-3)
    parser.add_argument("--ssim_weight", type=float, default=0.2)
    return parser.parse_args()


def load_images(image_paths: list[str], height: int, width: int) -> torch.Tensor:
    images = []
    for path in image_paths:
        image = Image.open(path).convert("RGB").resize((width, height), Image.LANCZOS)
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        images.append(tensor)
    return torch.stack(images, dim=0)


def load_gaussian_ply(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    ply = PlyData.read(path)
    vertices = ply["vertex"].data
    names = vertices.dtype.names or ()

    def stack_fields(prefix: str) -> np.ndarray:
        fields = sorted(
            [name for name in names if name.startswith(prefix)],
            key=lambda item: int(item.rsplit("_", 1)[1]),
        )
        if not fields:
            return np.zeros((len(vertices), 0), dtype=np.float32)
        return np.stack([vertices[name].astype(np.float32) for name in fields], axis=1)

    means = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1).astype(np.float32)
    f_dc = stack_fields("f_dc_")
    f_rest = stack_fields("f_rest_")
    log_scales = stack_fields("scale_")
    rotations = stack_fields("rot_")
    opacity_logits = vertices["opacity"].astype(np.float32)

    if f_dc.shape[1] != 3:
        raise ValueError(f"Expected 3 f_dc fields in {path}, found {f_dc.shape[1]}")
    if log_scales.shape[1] != 3:
        raise ValueError(f"Expected 3 scale fields in {path}, found {log_scales.shape[1]}")
    if rotations.shape[1] != 4:
        raise ValueError(f"Expected 4 rotation fields in {path}, found {rotations.shape[1]}")
    if f_rest.shape[1] % 3 != 0:
        raise ValueError(f"f_rest field count must be divisible by 3, found {f_rest.shape[1]}")

    rest_degree_count = f_rest.shape[1] // 3
    harmonics = np.zeros((len(vertices), 3, rest_degree_count + 1), dtype=np.float32)
    harmonics[:, :, 0] = f_dc
    if rest_degree_count:
        harmonics[:, :, 1:] = f_rest.reshape(len(vertices), 3, rest_degree_count)

    sh_count = harmonics.shape[-1]
    sh_degree = int(math.sqrt(sh_count)) - 1
    if (sh_degree + 1) ** 2 != sh_count:
        raise ValueError(f"SH coefficient count {sh_count} is not a square")

    return {
        "means": torch.tensor(means, device=device),
        "log_scales": torch.tensor(log_scales, device=device),
        "rotations": torch.tensor(rotations, device=device),
        "opacity_logits": torch.tensor(opacity_logits, device=device),
        "harmonics": torch.tensor(harmonics, device=device),
        "sh_degree": sh_degree,
    }


def normalized_poses(
    c2w: np.ndarray,
    num_context: int | None,
    context_selection: str,
    pose_space: str,
) -> tuple[np.ndarray, list[int]]:
    if pose_space == "global":
        return c2w.astype(np.float32), []

    total = len(c2w)
    context_count = min(num_context or total, total)
    context_indices = select_context_views(c2w, context_count, context_selection)
    c2w_tensor = torch.tensor(c2w, dtype=torch.float32)
    pivot = c2w_tensor[context_indices[len(context_indices) // 2] : context_indices[len(context_indices) // 2] + 1]
    normalized = camera_normalization(pivot, c2w_tensor).numpy().astype(np.float32)
    return normalized, [int(item) for item in context_indices.tolist()]


def make_ssim_window(channels: int, device: torch.device) -> torch.Tensor:
    window = torch.ones((channels, 1, 3, 3), dtype=torch.float32, device=device) / 9.0
    return window


def ssim_loss(pred: torch.Tensor, target: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
    pred_chw = pred.permute(0, 3, 1, 2)
    target_chw = target.permute(0, 3, 1, 2)
    channels = pred_chw.shape[1]
    mu_pred = F.conv2d(pred_chw, window, padding=1, groups=channels)
    mu_target = F.conv2d(target_chw, window, padding=1, groups=channels)
    sigma_pred = F.conv2d(pred_chw * pred_chw, window, padding=1, groups=channels) - mu_pred.square()
    sigma_target = F.conv2d(target_chw * target_chw, window, padding=1, groups=channels) - mu_target.square()
    sigma_cross = F.conv2d(pred_chw * target_chw, window, padding=1, groups=channels) - mu_pred * mu_target
    c1 = 0.01**2
    c2 = 0.03**2
    ssim = ((2.0 * mu_pred * mu_target + c1) * (2.0 * sigma_cross + c2)) / (
        (mu_pred.square() + mu_target.square() + c1) * (sigma_pred + sigma_target + c2)
    )
    return torch.clamp((1.0 - ssim) * 0.5, min=0.0, max=1.0).mean()


def render_batch(
    means: torch.Tensor,
    log_scales: torch.Tensor,
    rotations: torch.Tensor,
    opacity_logits: torch.Tensor,
    harmonics: torch.Tensor,
    sh_degree: int,
    viewmats: torch.Tensor,
    intrinsics: torch.Tensor,
    width: int,
    height: int,
    near: float,
    far: float,
) -> torch.Tensor:
    colors = harmonics.permute(0, 2, 1).contiguous()
    render_colors, _render_alphas, _meta = rasterization(
        means=means,
        quats=F.normalize(rotations, dim=-1),
        scales=torch.exp(log_scales).clamp(1e-6, 1e3),
        opacities=torch.sigmoid(opacity_logits),
        colors=colors,
        sh_degree=sh_degree,
        viewmats=viewmats,
        Ks=intrinsics,
        width=width,
        height=height,
        near_plane=near,
        far_plane=far,
        eps2d=0.1,
        rasterize_mode="antialiased",
        packed=True,
        absgrad=False,
        sparse_grad=False,
        render_mode="RGB",
    )
    return render_colors[..., :3]


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("[gsplat] Loading COLMAP scene...")
    scene_data = load_colmap_scene(args.scene_path, args.sparse_dir, args.images_dir)
    frame_count = len(scene_data["image_paths"])
    first_image = Image.open(scene_data["image_paths"][0])
    orig_w, orig_h = first_image.size
    target_h, target_w = compute_target_shape(orig_h, orig_w, args.max_resolution, args.image_shape)
    print(f"[gsplat] Images: {frame_count}, target size: {target_h}x{target_w}")

    c2w, context_indices = normalized_poses(
        scene_data["c2w"],
        args.num_context,
        args.context_selection,
        args.pose_space,
    )
    viewmats = torch.inverse(torch.tensor(c2w, dtype=torch.float32, device=device))
    intrinsics = torch.tensor(scene_data["intrinsics"], dtype=torch.float32, device=device)
    intrinsics[:, 0] *= target_w
    intrinsics[:, 1] *= target_h

    print("[gsplat] Loading target images...")
    images_cpu = load_images(scene_data["image_paths"], target_h, target_w)

    print(f"[gsplat] Loading initialized gaussians: {args.init_ply}")
    gaussian = load_gaussian_ply(args.init_ply, device)
    means = torch.nn.Parameter(gaussian["means"])
    log_scales = torch.nn.Parameter(gaussian["log_scales"])
    rotations = torch.nn.Parameter(gaussian["rotations"])
    opacity_logits = torch.nn.Parameter(gaussian["opacity_logits"])
    harmonics = torch.nn.Parameter(gaussian["harmonics"])
    sh_degree = int(gaussian["sh_degree"])
    print(f"[gsplat] Gaussians: {means.shape[0]}, SH degree: {sh_degree}")

    optimizer = torch.optim.Adam(
        [
            {"params": [means], "lr": args.lr_means},
            {"params": [log_scales], "lr": args.lr_scales},
            {"params": [rotations], "lr": args.lr_rotations},
            {"params": [opacity_logits], "lr": args.lr_opacities},
            {"params": [harmonics], "lr": args.lr_colors},
        ],
        eps=1e-15,
    )

    order = list(range(frame_count))
    random.shuffle(order)
    cursor = 0
    batch_size = max(1, min(args.batch_size, frame_count))
    ssim_window = make_ssim_window(3, device)
    losses = []

    progress = tqdm(range(1, args.steps + 1), desc="gsplat optimize", dynamic_ncols=True)
    for step in progress:
        if cursor + batch_size > frame_count:
            random.shuffle(order)
            cursor = 0
        indices = order[cursor : cursor + batch_size]
        cursor += batch_size

        batch_viewmats = viewmats[indices]
        batch_intrinsics = intrinsics[indices]
        target = images_cpu[indices].to(device, non_blocking=True).permute(0, 2, 3, 1)

        pred = render_batch(
            means,
            log_scales,
            rotations,
            opacity_logits,
            harmonics,
            sh_degree,
            batch_viewmats,
            batch_intrinsics,
            target_w,
            target_h,
            args.near,
            args.far,
        )
        l1 = F.l1_loss(pred, target)
        if args.ssim_weight > 0:
            loss = (1.0 - args.ssim_weight) * l1 + args.ssim_weight * ssim_loss(pred, target, ssim_window)
        else:
            loss = l1

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().cpu())
        losses.append(loss_value)
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(f"[gsplat] step {step}/{args.steps} loss={loss_value:.6f} l1={float(l1.detach().cpu()):.6f}")
        progress.set_postfix(loss=f"{loss_value:.5f}")

    output_ply = args.output_dir / "guassian_gsplat.ply"
    export_ply(
        torch.eye(4, dtype=torch.float32),
        means.detach(),
        torch.exp(log_scales.detach()).clamp(1e-6, 1e3),
        F.normalize(rotations.detach(), dim=-1),
        harmonics.detach(),
        torch.sigmoid(opacity_logits.detach()).clamp(1e-6, 1.0 - 1e-6),
        output_ply,
        align_to_view=False,
    )

    metadata = {
        "frame_count": frame_count,
        "image_shape": [target_h, target_w],
        "steps": args.steps,
        "batch_size": batch_size,
        "pose_space": args.pose_space,
        "context_indices": context_indices,
        "init_ply": str(args.init_ply),
        "output_ply": str(output_ply),
        "final_loss": losses[-1] if losses else None,
        "losses": losses,
    }
    (args.output_dir / "gsplat_manifest.json").write_text(json.dumps(metadata, indent=2))
    print(f"[gsplat] Saved optimized PLY: {output_ply}")


if __name__ == "__main__":
    main()

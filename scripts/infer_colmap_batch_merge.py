"""
Batch ReSplat inference for a single COLMAP scene, merged into one global PLY.

This script keeps the COLMAP reconstruction as the global coordinate frame. Each
ReSplat batch is run in the usual pivot-normalized local frame, then the kept
Gaussians are transformed back through that batch pivot camera before merging.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from PIL import Image
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.infer_colmap import (  # noqa: E402
    MODEL_PRESETS,
    build_batch,
    build_model,
    compute_target_shape,
    load_and_preprocess_images,
    load_colmap_scene,
    run_inference,
)


GAUSSIAN_TRIM = 2


def plan_batches(num_frames: int, batch_size: int, batch_overlap: int) -> list[dict]:
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if batch_overlap < 0:
        raise ValueError("batch_overlap must be non-negative")
    if batch_overlap >= batch_size:
        raise ValueError("batch_overlap must be smaller than batch_size")

    batch_size = min(batch_size, num_frames)
    if num_frames <= batch_size:
        starts = [0]
    else:
        stride = batch_size - batch_overlap
        starts = list(range(0, num_frames - batch_size + 1, stride))
        final_start = num_frames - batch_size
        if starts[-1] != final_start:
            starts.append(final_start)
        starts = sorted(set(starts))

    windows = []
    centers = []
    for start in starts:
        end = min(num_frames, start + batch_size)
        center = (start + end - 1) / 2.0
        centers.append(center)
        windows.append({"start": start, "end": end, "center": center})

    assignments: dict[int, int] = {}
    for frame_idx in range(num_frames):
        candidates = [
            (batch_idx, abs(frame_idx - centers[batch_idx]))
            for batch_idx, window in enumerate(windows)
            if window["start"] <= frame_idx < window["end"]
        ]
        if not candidates:
            raise RuntimeError(f"Frame {frame_idx} is not covered by any batch")
        assignments[frame_idx] = min(candidates, key=lambda item: (item[1], item[0]))[0]

    for batch_idx, window in enumerate(windows):
        core = [
            frame_idx
            for frame_idx, assigned_batch in assignments.items()
            if assigned_batch == batch_idx
        ]
        window["core_start"] = min(core)
        window["core_end"] = max(core) + 1
        window["core_indices"] = core

    return windows


def gaussian_view_shape(gaussians, num_views: int, image_h: int, image_w: int) -> tuple[int, int]:
    total = gaussians.means.shape[1]
    pixels = num_views * image_h * image_w
    if total == pixels:
        return image_h, image_w
    scale = pixels / total
    scale = int(round(math.sqrt(scale)))
    if scale <= 0 or image_h % scale != 0 or image_w % scale != 0:
        raise RuntimeError(
            f"Could not infer Gaussian grid from {total} gaussians, "
            f"{num_views} views, image shape {image_h}x{image_w}"
        )
    latent_h = image_h // scale
    latent_w = image_w // scale
    if num_views * latent_h * latent_w != total:
        raise RuntimeError(
            f"Gaussian grid mismatch: {total} != {num_views}*{latent_h}*{latent_w}"
        )
    return latent_h, latent_w


def gaussian_keep_mask(
    gaussians,
    num_context_views: int,
    keep_local_views: set[int],
    image_h: int,
    image_w: int,
) -> torch.Tensor:
    context_views = num_context_views
    h, w = gaussian_view_shape(gaussians, context_views, image_h, image_w)
    means = rearrange(
        gaussians.means,
        "() (v h w spp) xyz -> h w spp v xyz",
        v=context_views,
        h=h,
        w=w,
    )
    mask_grid = torch.zeros_like(means[..., 0], dtype=torch.bool)
    if h <= GAUSSIAN_TRIM * 2 or w <= GAUSSIAN_TRIM * 2:
        mask_grid[..., :] = True
    else:
        mask_grid[GAUSSIAN_TRIM:-GAUSSIAN_TRIM, GAUSSIAN_TRIM:-GAUSSIAN_TRIM, :, :] = True

    view_mask = torch.zeros((context_views,), dtype=torch.bool, device=mask_grid.device)
    for local_idx in keep_local_views:
        view_mask[local_idx] = True
    mask_grid &= view_mask.view(1, 1, 1, context_views)
    return mask_grid


def filter_gaussians(
    gaussians,
    num_context_views: int,
    keep_local_views: set[int],
    image_h: int,
    image_w: int,
) -> dict[str, torch.Tensor]:
    mask = gaussian_keep_mask(gaussians, num_context_views, keep_local_views, image_h, image_w)
    h, w = gaussian_view_shape(gaussians, num_context_views, image_h, image_w)

    def select(element: torch.Tensor) -> torch.Tensor:
        grid = rearrange(
            element,
            "() (v h w spp) ... -> h w spp v ...",
            v=num_context_views,
            h=h,
            w=w,
        )
        return grid[mask].detach().cpu()

    local_view_grid = torch.arange(num_context_views, device=mask.device).view(1, 1, 1, num_context_views)
    local_view_indices = local_view_grid.expand_as(mask)[mask].detach().cpu()

    return {
        "means": select(gaussians.means),
        "scales": select(gaussians.scales),
        "rotations": select(gaussians.rotations),
        "harmonics": select(gaussians.harmonics),
        "opacities": select(gaussians.opacities),
        "local_view_indices": local_view_indices,
    }


def transform_gaussians_to_global(
    filtered: dict[str, torch.Tensor],
    pivot_c2w: torch.Tensor,
    source_c2w: torch.Tensor,
) -> dict[str, torch.Tensor]:
    pivot_c2w = pivot_c2w.detach().cpu()
    source_c2w = source_c2w.detach().cpu()
    rotation = pivot_c2w[:3, :3]
    translation = pivot_c2w[:3, 3]

    means = filtered["means"] @ rotation.T + translation

    local_rotations = filtered["rotations"].numpy()
    source_indices = filtered["local_view_indices"].numpy()
    source_rotations = source_c2w[source_indices, :3, :3].numpy()
    global_mats = source_rotations @ R.from_quat(local_rotations).as_matrix()
    global_xyzw = R.from_matrix(global_mats).as_quat()
    x, y, z, w = np.moveaxis(global_xyzw, -1, 0)
    global_wxyz = torch.from_numpy(np.stack((w, x, y, z), axis=-1).astype(np.float32))

    return {
        "means": means,
        "scales": filtered["scales"],
        "harmonics": filtered["harmonics"],
        "opacities": filtered["opacities"],
        "rotations": global_wxyz,
    }


def concat_gaussians(parts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not parts:
        raise RuntimeError("No Gaussians were kept; cannot write merged PLY")
    return {
        key: torch.cat([part[key] for part in parts], dim=0)
        for key in ("means", "scales", "rotations", "harmonics", "opacities")
    }


def write_merged_ply(merged: dict[str, torch.Tensor], output_path: Path) -> None:
    from src.model.ply_export import export_ply

    export_ply(
        torch.eye(4),
        merged["means"],
        merged["scales"],
        merged["rotations"],
        merged["harmonics"],
        merged["opacities"],
        output_path,
        align_to_view=False,
    )


def write_rendered_video(rendered: torch.Tensor, output_path: Path, fps: int = 12) -> None:
    import imageio

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for image in rendered:
        frame = (
            image.clamp(0, 1)
            .permute(1, 2, 0)
            .detach()
            .cpu()
            .numpy()
            * 255
        ).astype(np.uint8)
        frames.append(frame)
    imageio.mimwrite(output_path, frames, fps=fps, quality=8)


def write_manifest(manifest: dict, output_dir: Path) -> None:
    manifest_path = output_dir / "batch_manifest.json"
    tmp_path = manifest_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tmp_path.replace(manifest_path)


def load_ground_alignment(scene_path: Path, allow_unaligned_scene: bool) -> dict | None:
    alignment_path = scene_path / "ground_alignment.json"
    if not alignment_path.exists():
        if allow_unaligned_scene:
            print(
                "Warning: ground_alignment.json not found; output PLY will use "
                "the scene's existing COLMAP coordinates."
            )
            return None
        raise FileNotFoundError(
            f"{alignment_path} not found. Batch merge expects an aligned COLMAP "
            "dataset produced by the COLMAP tab so the final PLY is in the "
            "ground-aligned z=0 coordinate frame."
        )
    alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
    candidate = alignment.get("candidate", {})
    if "transform" not in candidate:
        raise ValueError(
            f"{alignment_path} does not contain candidate.transform; cannot "
            "confirm the ground-aligned coordinate frame."
        )
    return alignment


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ReSplat over overlapping COLMAP frame batches and merge one global PLY"
    )
    parser.add_argument("--scene_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_preset", choices=list(MODEL_PRESETS.keys()), required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--experiment", default="dl3dv")
    parser.add_argument("--num_refine", type=int, default=None)
    parser.add_argument("--max_resolution", type=int, default=None)
    parser.add_argument("--image_shape", type=int, nargs=2, default=None, metavar=("H", "W"))
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--batch_overlap", type=int, default=None)
    parser.add_argument("--render_chunk_size", type=int, default=2)
    parser.add_argument("--near", type=float, default=0.01)
    parser.add_argument("--far", type=float, default=200.0)
    parser.add_argument("--images_dir", default="images")
    parser.add_argument("--sparse_dir", default="sparse")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overrides", nargs="*", default=[])
    parser.add_argument(
        "--allow_unaligned_scene",
        action="store_true",
        default=False,
        help="Allow running without ground_alignment.json. The output PLY will not be guaranteed to have ground at z=0.",
    )
    return parser.parse_args()


def apply_preset_defaults(args):
    preset = MODEL_PRESETS[args.model_preset]
    if args.checkpoint is None:
        args.checkpoint = preset["checkpoint"]
    if args.num_refine is None:
        args.num_refine = preset["num_refine"]
    if args.max_resolution is None:
        args.max_resolution = preset["max_resolution"]
    if args.batch_size is None:
        args.batch_size = preset["num_context"]
    if args.batch_overlap is None:
        args.batch_overlap = max(2, preset["num_context"] // 4)
    args.overrides = preset["overrides"] + args.overrides


def main():
    args = parse_args()
    apply_preset_defaults(args)

    output_dir = Path(args.output_dir)
    batches_dir = output_dir / "batches"
    output_dir.mkdir(parents=True, exist_ok=True)
    batches_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ReSplat COLMAP Batch Merge")
    print("=" * 60)
    print(f"  Scene: {args.scene_path}")
    print(f"  Preset: {args.model_preset}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Batch overlap: {args.batch_overlap}")

    scene_path = Path(args.scene_path)
    ground_alignment = load_ground_alignment(scene_path, args.allow_unaligned_scene)
    if ground_alignment is not None:
        candidate_id = ground_alignment.get("candidate_id")
        print(f"  Coordinate frame: ground-aligned z=0 (candidate {candidate_id})")

    scene_data = load_colmap_scene(args.scene_path, args.sparse_dir, args.images_dir)
    num_frames = len(scene_data["image_paths"])
    windows = plan_batches(num_frames, args.batch_size, args.batch_overlap)
    print(f"  Frames: {num_frames}")
    print(f"  Batches: {len(windows)}")

    first_img = Image.open(scene_data["image_paths"][0])
    orig_w, orig_h = first_img.size
    target_h, target_w = compute_target_shape(orig_h, orig_w, args.max_resolution, args.image_shape)
    print(f"  Resolution: {orig_h}x{orig_w} -> {target_h}x{target_w}")

    encoder, decoder, data_shim = build_model(
        experiment=args.experiment,
        checkpoint=args.checkpoint,
        num_refine=args.num_refine,
        image_shape=(target_h, target_w),
        overrides=args.overrides,
        device=args.device,
        no_strict_load=True,
    )

    merged_parts = []
    manifest = {
        "scene_path": args.scene_path,
        "model_preset": args.model_preset,
        "checkpoint": args.checkpoint,
        "frame_count": num_frames,
        "batch_size": args.batch_size,
        "batch_overlap": args.batch_overlap,
        "resolution": [target_h, target_w],
        "coordinate_frame": "ground_aligned_z0" if ground_alignment is not None else "scene_colmap",
        "ground_alignment": ground_alignment,
        "total_batches": len(windows),
        "completed_batches": 0,
        "batches": [],
    }
    write_manifest(manifest, output_dir)

    for batch_idx, window in enumerate(windows):
        frame_indices = np.arange(window["start"], window["end"])
        core_global = set(window["core_indices"])
        keep_local = {
            int(global_idx - window["start"])
            for global_idx in core_global
            if window["start"] <= global_idx < window["end"]
        }

        batch_output_dir = batches_dir / f"batch_{batch_idx:03d}"
        batch_output_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"\n[batch {batch_idx + 1}/{len(windows)}] "
            f"frames {window['start']}..{window['end'] - 1}, "
            f"core {window['core_start']}..{window['core_end'] - 1}"
        )

        images = load_and_preprocess_images(
            [scene_data["image_paths"][i] for i in frame_indices],
            target_h,
            target_w,
        )
        c2w = torch.tensor(scene_data["c2w"][frame_indices], dtype=torch.float32)
        intrinsics = torch.tensor(scene_data["intrinsics"][frame_indices], dtype=torch.float32)

        batch = build_batch(
            images,
            images,
            c2w,
            c2w,
            intrinsics,
            intrinsics,
            args.near,
            args.far,
            f"batch_{batch_idx:03d}",
            args.device,
        )
        batch = data_shim(batch)

        gaussians, _rendered, _rendered_depth, _visualization_dump = run_inference(
            encoder,
            decoder,
            batch,
            args.num_refine,
            args.render_chunk_size,
            save_depth=False,
        )

        filtered = filter_gaussians(gaussians, len(frame_indices), keep_local, target_h, target_w)
        pivot_global = c2w[len(c2w) // 2]
        global_part = transform_gaussians_to_global(filtered, pivot_global, c2w)
        kept = int(global_part["means"].shape[0])
        merged_parts.append(global_part)

        write_merged_ply(global_part, batch_output_dir / "gaussians_global_core.ply")
        batch_manifest = {
            "index": batch_idx,
            "start": window["start"],
            "end": window["end"],
            "core_start": window["core_start"],
            "core_end": window["core_end"],
            "core_indices": window["core_indices"],
            "kept_gaussians": kept,
            "output": str(batch_output_dir / "gaussians_global_core.ply"),
            "video": str(batch_output_dir / "video.mp4"),
        }
        try:
            write_rendered_video(_rendered, batch_output_dir / "video.mp4")
        except Exception as exc:
            print(f"Warning: failed to write batch video: {exc}")
            batch_manifest.pop("video", None)
        manifest["batches"].append(batch_manifest)
        manifest["completed_batches"] = len(manifest["batches"])
        write_manifest(manifest, output_dir)
        print(f"[batch {batch_idx + 1}/{len(windows)}] kept {kept} gaussians")

        del batch, gaussians, filtered, global_part, _rendered
        torch.cuda.empty_cache()

    merged = concat_gaussians(merged_parts)
    manifest["merged_gaussians"] = int(merged["means"].shape[0])
    output_ply = output_dir / "gaussians.ply"
    write_merged_ply(merged, output_ply)
    write_manifest(manifest, output_dir)
    print(f"\nSaved merged PLY: {output_ply}")
    print(f"Saved manifest: {output_dir / 'batch_manifest.json'}")
    print("Note: batch merge currently writes merged gaussians.ply only; video.mp4 is skipped.")


if __name__ == "__main__":
    main()

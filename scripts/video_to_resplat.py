#!/usr/bin/env python3
"""Build a COLMAP scene from a video and run ReSplat inference."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
from pathlib import Path


PRESET_SHAPES = {
    "dl3dv_8v_512x960": (512, 960),
    "dl3dv_16v_540x960": (540, 960),
    "dl3dv_8v_256x448": (256, 448),
    "dl3dv_16v_256x448": (256, 448),
    "dl3dv_32v_256x448": (256, 448),
    "dl3dv_8v_256x448_small": (256, 448),
    "dl3dv_8v_256x448_large": (256, 448),
}


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def check_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Required executable not found in PATH: {name}")


def colmap_supports_option(command: str, option: str) -> bool:
    result = subprocess.run(
        ["colmap", command, "-h"],
        check=True,
        capture_output=True,
        text=True,
    )
    return option in result.stdout or option in result.stderr


def colmap_gpu_option(command: str, new_option: str, legacy_option: str) -> str:
    if colmap_supports_option(command, new_option):
        return new_option
    if colmap_supports_option(command, legacy_option):
        return legacy_option
    raise RuntimeError(
        f"COLMAP {command} does not expose a known GPU option "
        f"({new_option} or {legacy_option})"
    )


def ffprobe_json(video: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames:stream_tags=rotate:format=duration",
            "-of",
            "json",
            str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def video_duration_seconds(video: Path) -> float:
    info = ffprobe_json(video)
    duration = float(info.get("format", {}).get("duration", 0.0))
    if duration <= 0:
        raise RuntimeError(f"Could not determine video duration for {video}")
    return duration


def reset_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"{path} already exists. Use --overwrite to regenerate it."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def extract_frames(
    video: Path,
    image_dir: Path,
    num_frames: int,
    start_time: float,
    end_time: float | None,
) -> None:
    reset_dir(image_dir, overwrite=True)
    video_duration = video_duration_seconds(video)
    if start_time < 0:
        raise ValueError("--start_time must be non-negative")
    if start_time >= video_duration:
        raise ValueError(
            f"--start_time must be before the video end ({video_duration:.3f}s)"
        )
    end = video_duration if end_time is None else min(end_time, video_duration)
    if end <= start_time:
        raise ValueError("--end_time must be after --start_time")
    duration = end - start_time
    if num_frames <= 0:
        raise ValueError("--num_frames must be positive")

    # Sample from bin centers so the first/last frames are not overly likely to
    # be black transition frames.
    for i in range(num_frames):
        timestamp = min(end - 1e-3, start_time + (i + 0.5) * duration / num_frames)
        output = image_dir / f"frame_{i:05d}.jpg"
        run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{timestamp:.6f}",
                "-i",
                str(video),
                "-frames:v",
                "1",
                "-q:v",
                "2",
                str(output),
            ]
        )

    extracted = sorted(image_dir.glob("*.jpg"))
    if len(extracted) != num_frames:
        raise RuntimeError(f"Expected {num_frames} frames, extracted {len(extracted)}")


def run_colmap(raw_images: Path, colmap_dir: Path, scene_dir: Path, use_gpu: int) -> None:
    database = colmap_dir / "database.db"
    sparse_dir = colmap_dir / "sparse"
    mapper_output = sparse_dir / "0"
    extraction_gpu_option = colmap_gpu_option(
        "feature_extractor",
        "--FeatureExtraction.use_gpu",
        "--SiftExtraction.use_gpu",
    )
    matching_gpu_option = colmap_gpu_option(
        "sequential_matcher",
        "--FeatureMatching.use_gpu",
        "--SiftMatching.use_gpu",
    )

    colmap_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    run(
        [
            "colmap",
            "feature_extractor",
            "--database_path",
            str(database),
            "--image_path",
            str(raw_images),
            "--ImageReader.single_camera",
            "1",
            "--ImageReader.camera_model",
            "SIMPLE_RADIAL",
            extraction_gpu_option,
            str(use_gpu),
        ]
    )
    run(
        [
            "colmap",
            "sequential_matcher",
            "--database_path",
            str(database),
            "--SequentialMatching.overlap",
            "10",
            matching_gpu_option,
            str(use_gpu),
        ]
    )
    run(
        [
            "colmap",
            "mapper",
            "--database_path",
            str(database),
            "--image_path",
            str(raw_images),
            "--output_path",
            str(sparse_dir),
        ]
    )

    if not mapper_output.exists():
        reconstructions = sorted(p for p in sparse_dir.iterdir() if p.is_dir())
        if not reconstructions:
            raise RuntimeError("COLMAP mapper did not produce a sparse reconstruction")
        mapper_output = reconstructions[0]

    run(
        [
            "colmap",
            "image_undistorter",
            "--image_path",
            str(raw_images),
            "--input_path",
            str(mapper_output),
            "--output_path",
            str(scene_dir),
            "--output_type",
            "COLMAP",
        ]
    )

    sparse_output = scene_dir / "sparse"
    if not (sparse_output / "cameras.bin").exists():
        raise RuntimeError(f"Undistorted COLMAP sparse files not found in {sparse_output}")


def run_resplat(args: argparse.Namespace, scene_dir: Path, output_dir: Path) -> None:
    cmd = [
        "python",
        "scripts/infer_colmap.py",
        "--model_preset",
        args.model_preset,
        "--scene_path",
        str(scene_dir),
        "--start_frame",
        "0",
        "--frame_distance",
        str(args.num_frames),
        "--images_dir",
        "images",
        "--sparse_dir",
        "sparse",
        "--output_dir",
        str(output_dir),
        "--target_selection",
        args.target_selection,
        "--save_images",
        "--save_video",
        "--save_ply",
        "--render_chunk_size",
        str(args.render_chunk_size),
        "--no_eval",
    ]

    if args.num_context is not None:
        cmd += ["--num_context", str(args.num_context)]
    elif args.num_frames < 8:
        cmd += ["--num_context", str(args.num_frames)]

    if args.image_shape is not None:
        cmd += ["--image_shape", str(args.image_shape[0]), str(args.image_shape[1])]
    elif args.use_preset_image_shape:
        shape = PRESET_SHAPES.get(args.model_preset)
        if shape is not None:
            cmd += ["--image_shape", str(shape[0]), str(shape[1])]

    run(cmd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract video frames, reconstruct cameras with COLMAP, and run ReSplat."
    )
    parser.add_argument("--video", required=True, type=Path, help="Input video path")
    parser.add_argument(
        "-N",
        "--num_frames",
        required=True,
        type=int,
        help="Number of frames to sample from the video",
    )
    parser.add_argument(
        "--work_dir",
        type=Path,
        default=Path("datasets/video_colmap"),
        help="Directory for extracted frames and COLMAP scene",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="ReSplat output directory. Default: results/<scene_name>-resplat",
    )
    parser.add_argument(
        "--scene_name",
        default=None,
        help="Scene folder name. Default: <video_stem>-N<num_frames>",
    )
    parser.add_argument(
        "--model_preset",
        default="dl3dv_8v_512x960",
        help="Preset passed to scripts/infer_colmap.py",
    )
    parser.add_argument(
        "--num_context",
        type=int,
        default=None,
        help="Context views passed to ReSplat. Default: preset value.",
    )
    parser.add_argument(
        "--target_selection",
        default="all",
        choices=["all", "remaining"],
        help="Which COLMAP views ReSplat should render.",
    )
    parser.add_argument(
        "--image_shape",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        default=None,
        help="Explicit ReSplat image shape.",
    )
    parser.add_argument(
        "--no_preset_image_shape",
        dest="use_preset_image_shape",
        action="store_false",
        help="Let infer_colmap.py derive resolution instead of using preset shape.",
    )
    parser.set_defaults(use_preset_image_shape=True)
    parser.add_argument(
        "--render_chunk_size",
        type=int,
        default=2,
        help="Number of target views rendered at once by ReSplat.",
    )
    parser.add_argument(
        "--start_time",
        type=float,
        default=0.0,
        help="Start time in seconds for video frame sampling.",
    )
    parser.add_argument(
        "--end_time",
        type=float,
        default=None,
        help="End time in seconds for video frame sampling. Default: video end.",
    )
    parser.add_argument(
        "--colmap_use_gpu",
        type=int,
        choices=[0, 1],
        default=1,
        help="Use GPU for COLMAP SIFT extraction/matching if this COLMAP build supports it.",
    )
    parser.add_argument(
        "--skip_colmap",
        action="store_true",
        help="Reuse an existing COLMAP scene in work_dir/scene_name/scene.",
    )
    parser.add_argument(
        "--skip_resplat",
        action="store_true",
        help="Only extract frames and run COLMAP.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and recreate the selected work scene before running.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    check_tool("ffmpeg")
    check_tool("ffprobe")
    check_tool("colmap")

    video = args.video.resolve()
    if not video.exists():
        raise FileNotFoundError(video)

    scene_name = args.scene_name or f"{video.stem}-N{args.num_frames}"
    root = args.work_dir / scene_name
    raw_images = root / "raw_images"
    colmap_dir = root / "colmap"
    scene_dir = root / "scene"
    output_dir = args.output_dir or Path("results") / f"{scene_name}-resplat"

    if root.exists() and args.overwrite and not args.skip_colmap:
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    if not args.skip_colmap:
        print(f"Extracting {args.num_frames} frames from {video}...")
        extract_frames(
            video,
            raw_images,
            args.num_frames,
            args.start_time,
            args.end_time,
        )
        if scene_dir.exists():
            shutil.rmtree(scene_dir)
        print("Running COLMAP reconstruction and undistortion...")
        run_colmap(raw_images, colmap_dir, scene_dir, args.colmap_use_gpu)

    if not args.skip_resplat:
        print("Running ReSplat...")
        run_resplat(args, scene_dir, output_dir)

    print("\nDone.")
    print(f"Scene:   {scene_dir}")
    if not args.skip_resplat:
        print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()

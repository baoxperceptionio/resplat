#!/usr/bin/env python3
"""Build a COLMAP scene from a video and run ReSplat inference."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
from pathlib import Path


VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".m4v",
    ".avi",
    ".mkv",
    ".webm",
    ".mpeg",
    ".mpg",
    ".mts",
    ".m2ts",
}
DEFAULT_SAMPLE_FPS = 4.0

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


def registered_image_count(model_path: Path) -> int:
    result = subprocess.run(
        ["colmap", "model_analyzer", "--path", str(model_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr
    match = re.search(r"Registered images:\s+(\d+)", output)
    if match is None:
        raise RuntimeError(f"Could not read registered image count for {model_path}")
    return int(match.group(1))


def largest_colmap_reconstruction(sparse_dir: Path) -> Path:
    reconstructions = sorted(p for p in sparse_dir.iterdir() if p.is_dir())
    if not reconstructions:
        raise RuntimeError("COLMAP mapper did not produce a sparse reconstruction")

    best = max(reconstructions, key=registered_image_count)
    count = registered_image_count(best)
    print(f"Selected COLMAP reconstruction: {best} ({count} registered images)")
    return best


def run_final_bundle_adjustment(
    mapper_output: Path,
    ba_output: Path,
    use_gpu: int,
) -> Path:
    reset_dir(ba_output, overwrite=True)
    run(
        [
            "colmap",
            "bundle_adjuster",
            "--input_path",
            str(mapper_output),
            "--output_path",
            str(ba_output),
            "--BundleAdjustment.refine_focal_length",
            "1",
            "--BundleAdjustment.refine_principal_point",
            "0",
            "--BundleAdjustment.refine_extra_params",
            "1",
            "--BundleAdjustment.refine_points3D",
            "1",
            "--BundleAdjustmentCeres.max_num_iterations",
            "200",
            "--BundleAdjustmentCeres.use_gpu",
            str(use_gpu),
            "--BundleAdjustmentCeres.min_num_images_gpu_solver",
            "1",
        ]
    )
    return ba_output


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


def video_paths(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"Input file is not a supported video: {input_path}")
        return [input_path]

    if input_path.is_dir():
        videos = sorted(
            p
            for p in input_path.rglob("*")
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        )
        if not videos:
            raise FileNotFoundError(f"No supported videos found under {input_path}")
        return videos

    raise FileNotFoundError(input_path)


def reset_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"{path} already exists. Use --overwrite to regenerate it."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def time_window(video: Path, start_time: float, end_time: float | None) -> tuple[float, float]:
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
    return start_time, end


def frame_count(video: Path, args: argparse.Namespace) -> int:
    if args.num_frames is not None:
        if args.num_frames <= 0:
            raise ValueError("--num_frames must be positive")
        return args.num_frames

    start, end = time_window(video, args.start_time, args.end_time)
    return max(1, math.ceil((end - start) * args.sample_fps))


def extract_video_frames(
    video: Path,
    image_dir: Path,
    num_frames: int,
    start_time: float,
    end_time: float | None,
    filename_prefix: str,
) -> int:
    start_time, end = time_window(video, start_time, end_time)
    duration = end - start_time

    # Sample from bin centers so the first/last frames are not overly likely to
    # be black transition frames.
    for i in range(num_frames):
        timestamp = min(end - 1e-3, start_time + (i + 0.5) * duration / num_frames)
        output = image_dir / f"{filename_prefix}_frame_{i:05d}.jpg"
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

    return num_frames


def extract_frames(videos: list[Path], image_dir: Path, args: argparse.Namespace) -> int:
    reset_dir(image_dir, overwrite=True)
    total = 0
    for video_index, video in enumerate(videos):
        count = frame_count(video, args)
        print(f"  {video}: extracting {count} frames")
        total += extract_video_frames(
            video,
            image_dir,
            count,
            args.start_time,
            args.end_time,
            f"video_{video_index:03d}",
        )

    extracted = sorted(image_dir.glob("*.jpg"))
    if len(extracted) != total:
        raise RuntimeError(f"Expected {total} frames, extracted {len(extracted)}")
    return total


def run_colmap(
    raw_images: Path,
    colmap_dir: Path,
    scene_dir: Path,
    use_gpu: int,
    matcher: str,
) -> None:
    database = colmap_dir / "database.db"
    sparse_dir = colmap_dir / "sparse"
    extraction_gpu_option = colmap_gpu_option(
        "feature_extractor",
        "--FeatureExtraction.use_gpu",
        "--SiftExtraction.use_gpu",
    )
    matching_gpu_option = colmap_gpu_option(
        matcher,
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
    match_cmd = [
        "colmap",
        matcher,
        "--database_path",
        str(database),
    ]
    if matcher == "sequential_matcher":
        match_cmd += ["--SequentialMatching.overlap", "10"]
    match_cmd += [matching_gpu_option, str(use_gpu)]
    run(match_cmd)
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
            "--Mapper.ba_use_gpu",
            str(use_gpu),
        ]
    )

    mapper_output = largest_colmap_reconstruction(sparse_dir)
    final_sparse = run_final_bundle_adjustment(
        mapper_output,
        colmap_dir / "sparse_ba" / "0",
        use_gpu,
    )

    run(
        [
            "colmap",
            "image_undistorter",
            "--image_path",
            str(raw_images),
            "--input_path",
            str(final_sparse),
            "--output_path",
            str(scene_dir),
            "--output_type",
            "COLMAP",
        ]
    )

    sparse_output = scene_dir / "sparse"
    if not (sparse_output / "cameras.bin").exists():
        raise RuntimeError(f"Undistorted COLMAP sparse files not found in {sparse_output}")


def run_resplat(
    args: argparse.Namespace,
    scene_dir: Path,
    output_dir: Path,
    frame_count: int,
) -> None:
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
        str(frame_count),
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
        "--smooth_video_fps",
        f"{args.output_video_fps:g}",
        "--smooth_video_source_fps",
        f"{args.sample_fps:g}",
        "--no_eval",
    ]

    if args.num_context is not None:
        cmd += ["--num_context", str(args.num_context)]
    elif frame_count < 8:
        cmd += ["--num_context", str(frame_count)]

    if args.image_shape is not None:
        cmd += ["--image_shape", str(args.image_shape[0]), str(args.image_shape[1])]
    elif args.use_preset_image_shape:
        shape = PRESET_SHAPES.get(args.model_preset)
        if shape is not None:
            cmd += ["--image_shape", str(shape[0]), str(shape[1])]

    run(cmd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract frames from one video or a folder of supported videos, "
            "reconstruct cameras with COLMAP, and run ReSplat."
        )
    )
    parser.add_argument(
        "--video",
        required=True,
        type=Path,
        help="Input video path, or a folder containing videos from one scene",
    )
    parser.add_argument(
        "-N",
        "--num_frames",
        default=None,
        type=int,
        help=(
            "Number of frames to sample per video. "
            "Default: sample each video at --sample_fps."
        ),
    )
    parser.add_argument(
        "--sample_fps",
        type=float,
        default=DEFAULT_SAMPLE_FPS,
        help="Frames per second to sample from each video when --num_frames is omitted.",
    )
    parser.add_argument(
        "--output_video_fps",
        type=float,
        default=30.0,
        help="FPS for the rendered MP4 output.",
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
        help="Scene folder name. Default: <input_stem>-N<num_frames> or <input_stem>-fps<sample_fps>",
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
        "--matcher",
        choices=["auto", "sequential", "exhaustive"],
        default="auto",
        help="COLMAP matcher. Default: sequential for one video, exhaustive for a folder.",
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

    input_path = args.video.resolve()
    videos = video_paths(input_path)

    if args.num_frames is None and args.sample_fps <= 0:
        raise ValueError("--sample_fps must be positive")
    if args.output_video_fps <= 0:
        raise ValueError("--output_video_fps must be positive")

    default_suffix = (
        f"N{args.num_frames}"
        if args.num_frames is not None
        else f"fps{args.sample_fps:g}"
    )
    scene_name = args.scene_name or f"{input_path.stem}-{default_suffix}"
    root = args.work_dir / scene_name
    raw_images = root / "raw_images"
    colmap_dir = root / "colmap"
    scene_dir = root / "scene"
    output_dir = args.output_dir or Path("results") / f"{scene_name}-resplat"

    if root.exists() and args.overwrite and not args.skip_colmap:
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    total_frames = len(list((scene_dir / "images").glob("*"))) if args.skip_colmap else 0

    if not args.skip_colmap:
        print(f"Extracting frames from {len(videos)} video(s) under {input_path}...")
        total_frames = extract_frames(videos, raw_images, args)
        if scene_dir.exists():
            shutil.rmtree(scene_dir)
        matcher = args.matcher
        if matcher == "auto":
            matcher = "exhaustive" if len(videos) > 1 else "sequential"
        matcher_command = f"{matcher}_matcher"
        print(f"Using COLMAP matcher: {matcher_command}")
        print("Running COLMAP reconstruction and undistortion...")
        run_colmap(
            raw_images,
            colmap_dir,
            scene_dir,
            args.colmap_use_gpu,
            matcher_command,
        )

    if total_frames <= 0:
        total_frames = len(list((scene_dir / "images").glob("*")))
    if total_frames <= 0:
        raise RuntimeError("Could not determine the number of frames for ReSplat")

    if not args.skip_resplat:
        print("Running ReSplat...")
        run_resplat(args, scene_dir, output_dir, total_frames)

    print("\nDone.")
    print(f"Scene:   {scene_dir}")
    if not args.skip_resplat:
        print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()

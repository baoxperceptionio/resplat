from __future__ import annotations

import csv
import io
import json
import os
import re
import shlex
import shutil
import struct
import subprocess
import threading
import time
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import docker
import numpy as np
import requests
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image


REPO_ROOT = Path(os.getenv("RESPLAT_REPO", "/workspace/resplat")).resolve()
JOB_ROOT = Path(os.getenv("FRONTEND_JOB_ROOT", REPO_ROOT / "users" / "webui-jobs")).resolve()
UPLOAD_ROOT = JOB_ROOT / "uploads"
RESPLAT_CONTAINER = os.getenv("RESPLAT_CONTAINER", "resplat")
MAX_CONCURRENT_JOBS = max(1, int(os.getenv("MAX_CONCURRENT_JOBS", "1")))
OUTPUT_VIDEO_FPS = float(os.getenv("RESPLAT_OUTPUT_VIDEO_FPS", "30"))
STATE_PATH = JOB_ROOT / "jobs.json"
PANORAMA_ROOT = JOB_ROOT / "panorama"
SEEDANCE_START_URL = "https://video.a2e.ai/api/v1/seedance2Video/start"
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
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
ALLOWED_DOWNLOADS = {"video.mp4", "gaussians.ply", "gaussians_preview.ply"}
COLMAP_ALLOWED_FILES = {"overview_zy.png"}
COLMAP_GROUND_RANSAC_THRESHOLD = 0.05
PLY_VIEWER_PREVIEW_MAX_BYTES = 300 * 1024 * 1024
PLY_VIEWER_PREVIEW_MAX_VERTICES = 512_000
PLY_TYPE_SIZES = {
    "char": 1,
    "uchar": 1,
    "int8": 1,
    "uint8": 1,
    "short": 2,
    "ushort": 2,
    "int16": 2,
    "uint16": 2,
    "int": 4,
    "uint": 4,
    "int32": 4,
    "uint32": 4,
    "float": 4,
    "float32": 4,
    "double": 8,
    "float64": 8,
}
CAMERA_MODEL_PARAMS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}

PRESET_SHAPES = {
    "dl3dv_8v_512x960": (512, 960),
    "dl3dv_16v_540x960": (540, 960),
    "dl3dv_8v_256x448": (256, 448),
    "dl3dv_16v_256x448": (256, 448),
    "dl3dv_32v_256x448": (256, 448),
    "dl3dv_8v_256x448_small": (256, 448),
    "dl3dv_8v_256x448_large": (256, 448),
}

MODEL_OPTIONS: list[dict[str, Any]] = [
    {
        "id": "resplat-base-dl3dv-256x448-view32",
        "label": "resplat-base-dl3dv-256x448-view32",
        "preset": "dl3dv_32v_256x448",
        "checkpoint": "pretrained/resplat-base-dl3dv-256x448-view32-439b63a6.pth",
        "views": 32,
        "resolution": "256x448",
    },
    {
        "id": "resplat-base-dl3dv-540x960-view16",
        "label": "resplat-base-dl3dv-540x960-view16",
        "preset": "dl3dv_16v_540x960",
        "checkpoint": "pretrained/resplat-base-dl3dv-540x960-view16-a72dc6d0.pth",
        "views": 16,
        "resolution": "540x960",
    },
    {
        "id": "resplat-base-dl3dv-512x960-view8",
        "label": "resplat-base-dl3dv-512x960-view8",
        "preset": "dl3dv_8v_512x960",
        "checkpoint": "pretrained/resplat-base-dl3dv-512x960-view8-8179ed87.pth",
        "views": 8,
        "resolution": "512x960",
    },
]
MODELS_BY_ID = {model["id"]: model for model in MODEL_OPTIONS}


def plan_batch_count(frame_count: int, batch_size: int, batch_overlap: int) -> int:
    frame_count = max(0, int(frame_count))
    batch_size = max(1, int(batch_size))
    batch_overlap = max(0, int(batch_overlap))
    if frame_count <= batch_size:
        return 1
    stride = max(1, batch_size - min(batch_overlap, batch_size - 1))
    starts = list(range(0, frame_count - batch_size + 1, stride))
    final_start = frame_count - batch_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return len(set(starts))


def dataset_ground_alignment_path(dataset: dict[str, Any]) -> Path:
    return Path(dataset["scene"]) / "ground_alignment.json"


def resolve_stored_path(value: str | Path) -> Path:
    path = Path(value)
    if path.exists():
        return path
    workspace_root = Path("/workspace/resplat")
    try:
        relative = path.relative_to(workspace_root)
    except ValueError:
        return path
    candidate = REPO_ROOT / relative
    return candidate if candidate.exists() else path


def read_colmap_bytes(fid, num_bytes: int, format_char_sequence: str) -> tuple[Any, ...]:
    data = fid.read(num_bytes)
    if len(data) != num_bytes:
        raise EOFError("Unexpected end of COLMAP binary file")
    return struct.unpack("<" + format_char_sequence, data)


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ],
        dtype=np.float32,
    )


def read_colmap_cameras(sparse_path: Path) -> dict[int, dict[str, Any]]:
    cameras_bin = sparse_path / "cameras.bin"
    cameras_txt = sparse_path / "cameras.txt"
    cameras: dict[int, dict[str, Any]] = {}
    if cameras_bin.exists():
        with cameras_bin.open("rb") as fid:
            num_cameras = read_colmap_bytes(fid, 8, "Q")[0]
            for _ in range(num_cameras):
                camera_id, model_id, width, height = read_colmap_bytes(fid, 24, "iiQQ")
                model_name, num_params = CAMERA_MODEL_PARAMS[int(model_id)]
                params = read_colmap_bytes(fid, 8 * num_params, "d" * num_params)
                cameras[int(camera_id)] = {
                    "model": model_name,
                    "width": int(width),
                    "height": int(height),
                    "params": np.array(params, dtype=np.float64),
                }
        return cameras
    if cameras_txt.exists():
        for line in cameras_txt.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cameras[int(parts[0])] = {
                "model": parts[1],
                "width": int(parts[2]),
                "height": int(parts[3]),
                "params": np.array([float(item) for item in parts[4:]], dtype=np.float64),
            }
    return cameras


def read_colmap_images(sparse_path: Path) -> list[dict[str, Any]]:
    images_bin = sparse_path / "images.bin"
    images_txt = sparse_path / "images.txt"
    images: list[dict[str, Any]] = []
    if images_bin.exists():
        with images_bin.open("rb") as fid:
            num_images = read_colmap_bytes(fid, 8, "Q")[0]
            for _ in range(num_images):
                props = read_colmap_bytes(fid, 64, "idddddddi")
                name_bytes = bytearray()
                while True:
                    char = read_colmap_bytes(fid, 1, "c")[0]
                    if char == b"\x00":
                        break
                    name_bytes.extend(char)
                num_points2d = read_colmap_bytes(fid, 8, "Q")[0]
                fid.seek(24 * int(num_points2d), os.SEEK_CUR)
                images.append(
                    {
                        "id": int(props[0]),
                        "qvec": np.array(props[1:5], dtype=np.float64),
                        "tvec": np.array(props[5:8], dtype=np.float64),
                        "camera_id": int(props[8]),
                        "name": name_bytes.decode("utf-8"),
                    }
                )
        return sorted(images, key=lambda item: item["name"])
    if images_txt.exists():
        lines = images_txt.read_text().splitlines()
        index = 0
        while index < len(lines):
            line = lines[index].strip()
            index += 1
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            images.append(
                {
                    "id": int(parts[0]),
                    "qvec": np.array([float(item) for item in parts[1:5]], dtype=np.float64),
                    "tvec": np.array([float(item) for item in parts[5:8]], dtype=np.float64),
                    "camera_id": int(parts[8]),
                    "name": parts[9],
                }
            )
            index += 1
    return sorted(images, key=lambda item: item["name"])


def colmap_image_c2w(image: dict[str, Any]) -> np.ndarray:
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = qvec_to_rotmat(image["qvec"])
    w2c[:3, 3] = image["tvec"].astype(np.float32)
    return np.linalg.inv(w2c).astype(np.float32)


def colmap_camera_intrinsics(camera: dict[str, Any]) -> dict[str, Any]:
    params = camera["params"]
    if camera["model"] == "PINHOLE":
        fx, fy, cx, cy = params[:4]
    elif camera["model"] == "SIMPLE_PINHOLE":
        fx = fy = params[0]
        cx, cy = params[1:3]
    else:
        return {
            "width": camera["width"],
            "height": camera["height"],
            "model": camera["model"],
        }
    return {
        "width": camera["width"],
        "height": camera["height"],
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "model": camera["model"],
    }


def numpy_farthest_point_indices(points: np.ndarray, count: int) -> np.ndarray:
    total = len(points)
    if count >= total:
        return np.arange(total)
    selected = [0]
    min_distances = np.full(total, np.inf, dtype=np.float64)
    for _ in range(1, count):
        delta = points - points[selected[-1]]
        min_distances = np.minimum(min_distances, np.sum(delta * delta, axis=1))
        selected.append(int(np.argmax(min_distances)))
    return np.sort(np.array(selected, dtype=np.int64))


def initial_camera_for_job(job: dict[str, Any], frame_index: int = 0) -> dict[str, Any] | None:
    dataset = job.get("colmap_dataset")
    if not dataset:
        return None
    sparse_path = resolve_stored_path(dataset["scene"]) / dataset.get("sparse_dir", "sparse")
    try:
        cameras = read_colmap_cameras(sparse_path)
        images = read_colmap_images(sparse_path)
        if not cameras or not images:
            return None
        frame_index = min(max(0, int(frame_index)), len(images) - 1)
        image = images[frame_index]
        c2w = colmap_image_c2w(image)
        if job.get("resplat_mode") != "batch_merge":
            context_count = min(int(MODELS_BY_ID[job["model_id"]]["views"]), len(images))
            all_c2w = np.stack([colmap_image_c2w(item) for item in images], axis=0)
            context_indices = numpy_farthest_point_indices(all_c2w[:, :3, 3], context_count)
            pivot = all_c2w[context_indices[len(context_indices) // 2]]
            c2w = np.linalg.inv(pivot) @ c2w
        return {
            "frame_index": frame_index,
            "image_name": image["name"],
            "c2w": c2w.astype(float).tolist(),
            "intrinsics": colmap_camera_intrinsics(cameras[image["camera_id"]]),
            "convention": "opencv_c2w",
        }
    except Exception:
        return None

app = FastAPI(title="ReSplat Web UI")
state_lock = threading.Lock()
run_semaphore = threading.Semaphore(MAX_CONCURRENT_JOBS)
metrics_lock = threading.Lock()
last_cpu_sample: tuple[int, int] | None = None
gpu_metrics_cache: tuple[float, list[dict[str, Any]]] | None = None


@app.middleware("http")
async def no_cache_for_ui(request, call_next):
    response = await call_next(request)
    if (
        request.url.path == "/"
        or request.url.path.startswith("/api/")
        or request.url.path.endswith((".js", ".css"))
    ):
        response.headers["Cache-Control"] = "no-store, max-age=0"
    return response


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip(".-")
    return slug[:120] or "upload"


def load_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return {"jobs": {}, "colmap_jobs": {}}
    try:
        state = json.loads(STATE_PATH.read_text())
    except json.JSONDecodeError:
        state = {}
    state.setdefault("jobs", {})
    state.setdefault("colmap_jobs", {})
    return state


def save_state(state: dict[str, Any]) -> None:
    JOB_ROOT.mkdir(parents=True, exist_ok=True)
    tmp_path = STATE_PATH.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(state, indent=2, sort_keys=True))
    tmp_path.replace(STATE_PATH)


def update_job(job_id: str, **updates: Any) -> dict[str, Any]:
    with state_lock:
        state = load_state()
        job = state["jobs"].get(job_id)
        if job is None:
            raise KeyError(job_id)
        job.update(updates)
        job["updated_at"] = utc_now()
        save_state(state)
        return job


def update_colmap_job(job_id: str, **updates: Any) -> dict[str, Any]:
    with state_lock:
        state = load_state()
        job = state["colmap_jobs"].get(job_id)
        if job is None:
            raise KeyError(job_id)
        job.update(updates)
        job["updated_at"] = utc_now()
        save_state(state)
        return job


def append_log(job_id: str, text: str) -> None:
    log_path = job_dir(job_id) / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", errors="replace") as handle:
        handle.write(text)


def append_colmap_log(job_id: str, text: str) -> None:
    log_path = colmap_job_dir(job_id) / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", errors="replace") as handle:
        handle.write(text)


def job_dir(job_id: str) -> Path:
    return JOB_ROOT / job_id


def colmap_job_dir(job_id: str) -> Path:
    return JOB_ROOT / "colmap" / job_id


def container_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        rel = resolved.relative_to(REPO_ROOT)
        return f"/workspace/resplat/{rel.as_posix()}"
    except ValueError:
        try:
            rel = resolved.relative_to(JOB_ROOT)
            return f"/workspace/resplat/users/webui-jobs/{rel.as_posix()}"
        except ValueError:
            return str(resolved)


def reset_job_workspace(root: Path) -> tuple[Path, Path, Path]:
    if root.exists():
        shutil.rmtree(root)
    raw_videos = root / "raw_videos"
    work = root / "work"
    results = root / "results"
    raw_videos.mkdir(parents=True, exist_ok=False)
    work.mkdir(parents=True, exist_ok=False)
    results.mkdir(parents=True, exist_ok=False)
    return raw_videos, work, results


def reset_colmap_workspace(root: Path) -> tuple[Path, Path, Path]:
    if root.exists():
        shutil.rmtree(root)
    raw_videos = root / "raw_videos"
    work = root / "work"
    proposals = root / "ground_proposals"
    raw_videos.mkdir(parents=True, exist_ok=False)
    work.mkdir(parents=True, exist_ok=False)
    proposals.mkdir(parents=True, exist_ok=False)
    return raw_videos, work, proposals


def create_gaussian_ply_preview(
    source_path: Path,
    preview_path: Path,
    max_vertices: int = PLY_VIEWER_PREVIEW_MAX_VERTICES,
) -> bool:
    if not source_path.exists():
        return False
    if source_path.stat().st_size <= PLY_VIEWER_PREVIEW_MAX_BYTES:
        return False
    if (
        preview_path.exists()
        and preview_path.stat().st_mtime_ns >= source_path.stat().st_mtime_ns
        and preview_path.stat().st_size > 0
    ):
        return True

    with source_path.open("rb") as source:
        header = bytearray()
        while True:
            line = source.readline()
            if not line:
                return False
            header.extend(line)
            if line == b"end_header\n":
                break
        data_offset = source.tell()

    header_text = header.decode("ascii", errors="strict")
    lines = header_text.splitlines()
    if "format binary_little_endian 1.0" not in lines:
        return False

    vertex_count = 0
    vertex_stride = 0
    in_vertex = False
    for line in lines:
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "element":
            in_vertex = parts[1] == "vertex"
            if in_vertex:
                vertex_count = int(parts[2])
            continue
        if in_vertex and len(parts) >= 3 and parts[0] == "property":
            if parts[1] == "list":
                return False
            vertex_stride += PLY_TYPE_SIZES.get(parts[1], 0)

    if vertex_count <= 0 or vertex_stride <= 0 or vertex_count <= max_vertices:
        return False

    preview_count = min(max_vertices, vertex_count)
    tmp_path = preview_path.with_suffix(".tmp")
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    next_index_num = 0
    with source_path.open("rb") as source, tmp_path.open("wb") as target:
        preview_header = re.sub(
            r"element vertex \d+",
            f"element vertex {preview_count}",
            header_text,
            count=1,
        )
        target.write(preview_header.encode("ascii"))
        for out_index in range(preview_count):
            index = (out_index * vertex_count) // preview_count
            if index < next_index_num:
                index = next_index_num
            source.seek(data_offset + index * vertex_stride)
            target.write(source.read(vertex_stride))
            next_index_num = index + 1
    tmp_path.replace(preview_path)
    return True


def artifact_version(path: Path) -> str | None:
    if not path.exists():
        return None
    stat = path.stat()
    return f"{int(stat.st_mtime_ns)}-{stat.st_size}"


def job_batch_public(job: dict[str, Any], batch: dict[str, Any]) -> dict[str, Any]:
    result_dir = Path(job["paths"]["results"])
    batch_index = int(batch["index"])
    batch_dir = result_dir / "batches" / f"batch_{batch_index:03d}"
    ply_path = batch_dir / "gaussians_global_core.ply"
    video_path = batch_dir / "video.mp4"
    ply_version = artifact_version(ply_path)
    video_version = artifact_version(video_path)
    return {
        **batch,
        "label": f"batch {batch_index + 1:02d}",
        "artifacts": {
            "video": (
                f"/api/jobs/{job['id']}/batches/{batch_index}/files/video.mp4?v={video_version}"
                if video_version else None
            ),
            "ply": (
                f"/api/jobs/{job['id']}/batches/{batch_index}/files/gaussians_global_core.ply?v={ply_version}"
                if ply_version else None
            ),
            "viewer_ply": (
                f"/api/jobs/{job['id']}/batches/{batch_index}/files/gaussians_global_core.ply?v={ply_version}"
                if ply_version else None
            ),
            "video_version": video_version,
            "ply_version": ply_version,
            "viewer_ply_version": ply_version,
            "initial_camera": initial_camera_for_job(job, int(batch.get("start") or 0)),
        },
    }


def job_public(job: dict[str, Any]) -> dict[str, Any]:
    result_dir = Path(job["paths"]["results"])
    video_path = result_dir / "video.mp4"
    ply_path = result_dir / "gaussians.ply"
    preview_ply_path = result_dir / "gaussians_preview.ply"
    enriched = dict(job)
    enriched.setdefault("output_video_fps", OUTPUT_VIDEO_FPS)
    manifest_path = result_dir / "batch_manifest.json"
    video_version = None
    ply_version = None
    preview_ply_version = None
    if video_path.exists():
        stat = video_path.stat()
        video_version = f"{int(stat.st_mtime_ns)}-{stat.st_size}"
    if ply_path.exists():
        stat = ply_path.stat()
        ply_version = f"{int(stat.st_mtime_ns)}-{stat.st_size}"
    if preview_ply_path.exists():
        stat = preview_ply_path.stat()
        preview_ply_version = f"{int(stat.st_mtime_ns)}-{stat.st_size}"
    if manifest_path.exists() and not enriched.get("batch_manifest"):
        try:
            enriched["batch_manifest"] = json.loads(manifest_path.read_text())
        except Exception:
            pass
    enriched["batches"] = [
        job_batch_public(job, batch)
        for batch in (enriched.get("batch_manifest") or {}).get("batches", [])
    ]
    enriched["artifacts"] = {
        "video": f"/api/jobs/{job['id']}/files/video.mp4?v={video_version}" if video_version else None,
        "ply": f"/api/jobs/{job['id']}/files/gaussians.ply?v={ply_version}" if ply_version else None,
        "viewer_ply": (
            f"/api/jobs/{job['id']}/files/gaussians_preview.ply?v={preview_ply_version}"
            if preview_ply_version
            else (f"/api/jobs/{job['id']}/files/gaussians.ply?v={ply_version}" if ply_version else None)
        ),
        "video_version": video_version,
        "ply_version": ply_version,
        "viewer_ply_version": preview_ply_version or ply_version,
        "initial_camera": initial_camera_for_job(job, 0),
    }
    return enriched


def colmap_public(job: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(job)
    proposals_path = Path(job["paths"]["proposals"]) / "ground_proposals.json"
    candidates = []
    if proposals_path.exists():
        try:
            proposals = json.loads(proposals_path.read_text())
            for item in proposals.get("candidates", []):
                image = item.get("image")
                candidates.append(
                    {
                        "id": item.get("id"),
                        "image": image,
                        "image_url": f"/api/colmap-jobs/{job['id']}/files/{image}" if image else None,
                        "inliers": item.get("inliers"),
                        "inlier_ratio": item.get("inlier_ratio"),
                        "rms_distance": item.get("rms_distance"),
                        "angle_to_camera_up_deg": item.get("angle_to_camera_up_deg"),
                        "camera_yz_track_angle_deg": item.get("camera_yz_track_angle_deg"),
                    }
                )
        except Exception:
            candidates = []
    enriched["candidates"] = candidates
    enriched["overview_url"] = (
        f"/api/colmap-jobs/{job['id']}/files/overview_zy.png"
        if (Path(job["paths"]["proposals"]) / "overview_zy.png").exists()
        else None
    )
    enriched["dataset_ready"] = bool(job.get("aligned_scene"))
    enriched["debug_reprojections"] = colmap_debug_reprojections_public(job)
    return enriched


def colmap_debug_reprojections_public(job: dict[str, Any]) -> list[dict[str, Any]]:
    aligned_scene = job.get("aligned_scene")
    if not aligned_scene:
        return []
    debug_dir = Path(aligned_scene) / "debug_reprojection"
    manifest_path = debug_dir / "manifest.json"
    if not manifest_path.exists():
        return []
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception:
        return []

    items = []
    for item in manifest.get("images", []):
        image_name = item.get("image")
        if not image_name or safe_slug(image_name) != image_name:
            continue
        path = debug_dir / image_name
        if not path.exists() or not re.fullmatch(r"reprojection_\d{2}_[A-Za-z0-9_.-]+\.jpg", image_name):
            continue
        version = artifact_version(path)
        items.append(
            {
                "image": image_name,
                "source_image": item.get("source_image"),
                "projected_count": item.get("projected_count"),
                "camera_id": item.get("camera_id"),
                "image_url": f"/api/colmap-jobs/{job['id']}/debug/{image_name}?v={version}",
            }
        )
    return items


def model_public(model: dict[str, Any]) -> dict[str, Any]:
    item = dict(model)
    item["checkpoint_exists"] = (REPO_ROOT / model["checkpoint"]).exists()
    return item


def colmap_dataset_public(job: dict[str, Any]) -> dict[str, Any] | None:
    aligned_scene = job.get("aligned_scene")
    if not aligned_scene:
        return None
    scene_path = Path(aligned_scene)
    if not scene_path.exists():
        return None
    return {
        "id": job["id"],
        "label": f"{job['id']} · candidate {job.get('selected_candidate_id')}",
        "frame_count": job.get("frame_count", 0),
        "sample_fps": job.get("sample_fps"),
        "start_time": job.get("start_time"),
        "end_time": job.get("end_time"),
        "candidate_id": job.get("selected_candidate_id"),
        "scene": str(scene_path),
        "scene_container": job.get("aligned_scene_container") or container_path(scene_path),
        "sparse_dir": job.get("aligned_sparse_dir", "sparse"),
    }


def get_colmap_dataset(dataset_id: str) -> dict[str, Any]:
    with state_lock:
        job = load_state()["colmap_jobs"].get(dataset_id)
    if job is None:
        raise HTTPException(status_code=404, detail="COLMAP dataset not found")
    dataset = colmap_dataset_public(job)
    if dataset is None:
        raise HTTPException(status_code=400, detail="COLMAP dataset is not ready")
    return dataset


def render_index(tool: str) -> str:
    tool = tool if tool in {"colmap", "pipeline", "panorama"} else "colmap"
    html = (Path(__file__).parent / "static" / "index.html").read_text()
    return html.replace('<body data-tool="pipeline">', f'<body data-tool="{tool}">')


def perspective_from_equirectangular(
    image: Image.Image,
    yaw_center_deg: float,
    hfov_deg: float,
    output_width: int,
    output_height: int,
) -> Image.Image:
    pano = np.asarray(image.convert("RGB"), dtype=np.float32)
    pano_h, pano_w, _ = pano.shape

    hfov = np.deg2rad(hfov_deg)
    vfov = 2.0 * np.arctan(np.tan(hfov / 2.0) * (output_height / output_width))
    yaw_center = np.deg2rad(yaw_center_deg)

    xs = (np.arange(output_width, dtype=np.float32) + 0.5) / output_width
    ys = (np.arange(output_height, dtype=np.float32) + 0.5) / output_height
    x = (2.0 * xs - 1.0) * np.tan(hfov / 2.0)
    y = (1.0 - 2.0 * ys) * np.tan(vfov / 2.0)
    ray_x, ray_y = np.meshgrid(x, y)
    ray_z = np.ones_like(ray_x)

    norm = np.sqrt(ray_x * ray_x + ray_y * ray_y + ray_z * ray_z)
    ray_x /= norm
    ray_y /= norm
    ray_z /= norm

    lon = np.arctan2(ray_x, ray_z) + yaw_center
    lat = np.arcsin(np.clip(ray_y, -1.0, 1.0))

    src_x = (lon / (2.0 * np.pi) % 1.0) * pano_w - 0.5
    src_y = (0.5 - lat / np.pi) * pano_h - 0.5
    src_y = np.clip(src_y, 0.0, pano_h - 1.0)

    x0 = np.floor(src_x).astype(np.int32) % pano_w
    x1 = (x0 + 1) % pano_w
    y0 = np.floor(src_y).astype(np.int32)
    y1 = np.clip(y0 + 1, 0, pano_h - 1)
    wx = (src_x - np.floor(src_x))[..., None]
    wy = (src_y - np.floor(src_y))[..., None]

    top = pano[y0, x0] * (1.0 - wx) + pano[y0, x1] * wx
    bottom = pano[y1, x0] * (1.0 - wx) + pano[y1, x1] * wx
    rectified = top * (1.0 - wy) + bottom * wy
    return Image.fromarray(np.clip(rectified, 0, 255).astype(np.uint8), mode="RGB")


def panorama_dir(panorama_id: str) -> Path:
    return PANORAMA_ROOT / panorama_id


def public_url(base_url: str, path: str) -> str:
    base = base_url.rstrip("/")
    if not base.startswith(("http://", "https://")):
        raise ValueError("Public base URL must start with http:// or https://")
    return f"{base}{path}"


def form_string(form: Any, name: str, default: str = "") -> str:
    value = form.get(name, default)
    if value is None or hasattr(value, "filename"):
        return default
    return str(value)


def form_float(form: Any, name: str, default: float | None = None) -> float | None:
    value = form_string(form, name, "")
    if value == "":
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"{name} must be a number") from exc


def form_int(form: Any, name: str, default: int) -> int:
    value = form_string(form, name, str(default))
    try:
        return int(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"{name} must be an integer") from exc


def form_uploads(form: Any) -> list[Any]:
    uploads: list[Any] = []
    for field in ("files", "files[]", "video", "videos"):
        uploads.extend(form.getlist(field))
    if not uploads:
        uploads = [value for _, value in form.multi_items() if hasattr(value, "filename")]
    return [item for item in uploads if hasattr(item, "filename") and hasattr(item, "read")]


def form_upload_ids(form: Any) -> list[str]:
    upload_ids: list[str] = []
    for field in ("upload_ids", "upload_ids[]"):
        upload_ids.extend(str(value) for value in form.getlist(field) if str(value).strip())
    return upload_ids


def form_existing_videos(form: Any) -> list[str]:
    videos: list[str] = []
    for field in ("existing_videos", "existing_videos[]", "existing_video"):
        videos.extend(str(value) for value in form.getlist(field) if str(value).strip())
    return videos


def users_root() -> Path:
    return REPO_ROOT / "users"


def resolve_user_video(relative_path: str) -> Path:
    if relative_path.startswith("/") or "\x00" in relative_path:
        raise HTTPException(status_code=400, detail="Invalid users video path")
    root = users_root().resolve()
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid users video path") from exc
    if not path.exists() or not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise HTTPException(status_code=404, detail="Users video not found")
    return path


def upload_dir(upload_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", upload_id):
        raise HTTPException(status_code=400, detail="Invalid upload id")
    return UPLOAD_ROOT / upload_id


def upload_meta(upload_id: str) -> dict[str, Any]:
    meta_path = upload_dir(upload_id) / "meta.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Upload not found")
    return json.loads(meta_path.read_text())


def staged_upload_plan(upload_ids: list[str], start_index: int = 0) -> list[dict[str, Any]]:
    plan = []
    for offset, upload_id in enumerate(upload_ids):
        meta = upload_meta(upload_id)
        if meta.get("status") != "complete":
            raise HTTPException(status_code=400, detail=f"Upload {upload_id} is not complete")
        source_path = Path(meta["path"])
        if not source_path.exists():
            raise HTTPException(status_code=404, detail=f"Uploaded file missing: {upload_id}")
        filename = normalized_video_filename(type("UploadRef", (), {"filename": meta["filename"], "content_type": "video/*"})(), offset)
        plan.append(
            {
                "index": start_index + offset,
                "upload": None,
                "source_path": source_path,
                "filename": filename,
                "original_name": meta["filename"],
                "bytes": int(meta.get("total_size") or source_path.stat().st_size),
                "upload_id": upload_id,
                "source_action": "move",
            }
        )
    return plan


def existing_video_plan(relative_paths: list[str], start_index: int = 0) -> list[dict[str, Any]]:
    plan = []
    for offset, relative_path in enumerate(relative_paths):
        source_path = resolve_user_video(relative_path)
        filename = normalized_video_filename(
            type("UploadRef", (), {"filename": source_path.name, "content_type": "video/*"})(),
            offset,
        )
        plan.append(
            {
                "index": start_index + offset,
                "upload": None,
                "source_path": source_path,
                "filename": filename,
                "original_name": relative_path,
                "bytes": source_path.stat().st_size,
                "upload_id": None,
                "source_action": "symlink",
            }
        )
    return plan


def direct_upload_plan(files: list[Any], start_index: int = 0) -> list[dict[str, Any]]:
    plan = []
    for offset, upload in enumerate(files):
        filename = normalized_video_filename(upload, offset)
        plan.append(
            {
                "index": start_index + offset,
                "upload": upload,
                "source_path": None,
                "filename": filename,
                "original_name": upload.filename,
                "bytes": None,
                "upload_id": None,
                "source_action": "direct",
            }
        )
    return plan


async def materialize_upload_plan(upload_plan: list[dict[str, Any]], raw_videos: Path) -> tuple[list[str], list[dict[str, Any]]]:
    uploaded_names = []
    source_manifest = []
    for item in upload_plan:
        target = raw_videos / f"{item['index']:03d}-{item['filename']}"
        bytes_written = 0
        if item["upload"] is not None:
            with target.open("wb") as handle:
                while chunk := await item["upload"].read(1024 * 1024):
                    bytes_written += len(chunk)
                    handle.write(chunk)
        else:
            source_path = Path(item["source_path"])
            bytes_written = int(item["bytes"])
            if item.get("source_action") == "symlink":
                target.symlink_to(source_path)
            else:
                shutil.move(str(source_path), target)
                shutil.rmtree(source_path.parent.parent, ignore_errors=True)

        uploaded_names.append(target.name)
        source_manifest.append(
            {
                "index": item["index"],
                "original_name": item["original_name"],
                "stored_name": target.name,
                "bytes": bytes_written,
                "upload_id": item.get("upload_id"),
                "source": item.get("source_action"),
            }
        )
    return uploaded_names, source_manifest


def log_job_request(message: str) -> None:
    print(f"[frontend:/api/jobs] {message}", flush=True)


def reject_job_request(detail: str) -> None:
    log_job_request(f"reject: {detail}")
    raise HTTPException(status_code=400, detail=detail)


def normalized_video_filename(upload: Any, index: int) -> str:
    filename = safe_slug(getattr(upload, "filename", "") or f"video-{index}.mp4")
    suffix = Path(filename).suffix.lower()
    content_type = (getattr(upload, "content_type", "") or "").lower()
    if suffix in VIDEO_EXTENSIONS:
        return filename
    if content_type.startswith("video/"):
        return f"{filename}.mp4"
    reject_job_request(
        f"Unsupported video extension: {filename} "
        f"(content_type={content_type or 'unknown'})"
    )


def run_resplat_job(job_id: str) -> None:
    with run_semaphore:
        job = update_job(job_id, status="running", started_at=utc_now())
        model = MODELS_BY_ID[job["model_id"]]
        if job.get("source_type") == "colmap":
            dataset = job["colmap_dataset"]
            frame_count = max(1, int(dataset.get("frame_count") or 1))
            source_video_fps = float(job.get("sample_fps") or dataset.get("sample_fps") or OUTPUT_VIDEO_FPS)
            if job.get("resplat_mode") == "batch_merge":
                command = [
                    "python",
                    "scripts/infer_colmap_batch_merge.py",
                    "--model_preset",
                    model["preset"],
                    "--scene_path",
                    dataset["scene_container"],
                    "--images_dir",
                    "images",
                    "--sparse_dir",
                    dataset.get("sparse_dir", "sparse"),
                    "--output_dir",
                    job["paths"]["results_container"],
                    "--batch_size",
                    str(job["batch_size"]),
                    "--batch_overlap",
                    str(job["batch_overlap"]),
                    "--render_chunk_size",
                    str(job["render_chunk_size"]),
                    "--video_fps",
                    f"{OUTPUT_VIDEO_FPS:g}",
                ]
            else:
                command = [
                    "python",
                    "scripts/infer_colmap.py",
                    "--model_preset",
                    model["preset"],
                    "--scene_path",
                    dataset["scene_container"],
                    "--start_frame",
                    "0",
                    "--frame_distance",
                    str(frame_count),
                    "--images_dir",
                    "images",
                    "--sparse_dir",
                    dataset.get("sparse_dir", "sparse"),
                    "--output_dir",
                    job["paths"]["results_container"],
                    "--target_selection",
                    "all",
                    "--save_images",
                    "--save_video",
                    "--save_ply",
                    "--render_chunk_size",
                    str(job["render_chunk_size"]),
                    "--smooth_video_fps",
                    f"{OUTPUT_VIDEO_FPS:g}",
                    "--smooth_video_source_fps",
                    f"{source_video_fps:g}",
                    "--no_eval",
                ]
                if frame_count < int(model["views"]):
                    command.extend(["--num_context", str(frame_count)])
            shape = PRESET_SHAPES.get(model["preset"])
            if shape is not None:
                command.extend(["--image_shape", str(shape[0]), str(shape[1])])
        else:
            command = [
                "python",
                "scripts/video_to_resplat.py",
                "--video",
                job["paths"]["uploads_container"],
                "--sample_fps",
                str(job["sample_fps"]),
                "--work_dir",
                job["paths"]["work_container"],
                "--output_dir",
                job["paths"]["results_container"],
                "--scene_name",
                job_id,
                "--model_preset",
                model["preset"],
                "--render_chunk_size",
                str(job["render_chunk_size"]),
                "--overwrite",
            ]
            if job.get("start_time") is not None:
                command.extend(["--start_time", str(job["start_time"])])
            if job.get("end_time") is not None:
                command.extend(["--end_time", str(job["end_time"])])

        command = [
            "bash",
            "-lc",
            "scripts/ensure_pointops.sh && " + shlex.join(command),
        ]

        append_log(job_id, "$ " + " ".join(command) + "\n")
        try:
            client = docker.from_env()
            container = client.containers.get(RESPLAT_CONTAINER)
            exec_id = client.api.exec_create(
                container.id,
                command,
                workdir=str(REPO_ROOT),
            )["Id"]
            for chunk in client.api.exec_start(exec_id, stream=True):
                append_log(job_id, chunk.decode("utf-8", errors="replace"))
            result = client.api.exec_inspect(exec_id)
            exit_code = result.get("ExitCode", 1)
            if exit_code == 0:
                updates: dict[str, Any] = {
                    "status": "succeeded",
                    "finished_at": utc_now(),
                    "exit_code": exit_code,
                }
                try:
                    if create_gaussian_ply_preview(
                        Path(job["paths"]["results"]) / "gaussians.ply",
                        Path(job["paths"]["results"]) / "gaussians_preview.ply",
                    ):
                        append_log(job_id, "\n[frontend] Wrote gaussians_preview.ply for browser viewer.\n")
                except Exception as exc:
                    append_log(job_id, f"\n[frontend] Failed to write viewer preview PLY: {exc}\n")
                manifest_path = Path(job["paths"]["results"]) / "batch_manifest.json"
                if manifest_path.exists():
                    try:
                        updates["batch_manifest"] = json.loads(manifest_path.read_text())
                    except Exception as exc:
                        append_log(job_id, f"\n[frontend] Failed to read batch manifest: {exc}\n")
                update_job(job_id, **updates)
            else:
                update_job(
                    job_id,
                    status="failed",
                    finished_at=utc_now(),
                    exit_code=exit_code,
                    error=f"ReSplat exited with code {exit_code}",
                )
        except Exception as exc:
            append_log(job_id, f"\n[frontend] {type(exc).__name__}: {exc}\n")
            update_job(
                job_id,
                status="failed",
                finished_at=utc_now(),
                exit_code=None,
                error=str(exc),
            )


def docker_exec_stream(command: list[str], log_fn, workdir: Path = REPO_ROOT) -> int:
    client = docker.from_env()
    container = client.containers.get(RESPLAT_CONTAINER)
    exec_id = client.api.exec_create(
        container.id,
        command,
        workdir=str(workdir),
    )["Id"]
    for chunk in client.api.exec_start(exec_id, stream=True):
        log_fn(chunk.decode("utf-8", errors="replace"))
    result = client.api.exec_inspect(exec_id)
    return int(result.get("ExitCode", 1))


def run_colmap_job(job_id: str) -> None:
    with run_semaphore:
        job = update_colmap_job(job_id, status="running", started_at=utc_now())
        scene_host = Path(job["paths"]["scene"])
        proposals_host = Path(job["paths"]["proposals"])
        colmap_command = [
            "python",
            "scripts/video_to_resplat.py",
            "--video",
            job["paths"]["uploads_container"],
            "--sample_fps",
            str(job["sample_fps"]),
            "--work_dir",
            job["paths"]["work_container"],
            "--scene_name",
            job_id,
            "--skip_resplat",
            "--matcher",
            job["matcher"],
            "--colmap_use_gpu",
            str(job["colmap_use_gpu"]),
            "--overwrite",
        ]
        if job.get("start_time") is not None:
            colmap_command.extend(["--start_time", str(job["start_time"])])
        if job.get("end_time") is not None:
            colmap_command.extend(["--end_time", str(job["end_time"])])

        propose_command = [
            "python",
            "scripts/align_colmap_ground.py",
            "propose",
            "--scene_path",
            job["paths"]["scene_container"],
            "--output_dir",
            job["paths"]["proposals_container"],
            "--num_candidates",
            str(job["num_candidates"]),
            "--iterations",
            str(job["ransac_iterations"]),
            "--threshold",
            str(COLMAP_GROUND_RANSAC_THRESHOLD),
        ]

        def log(text: str) -> None:
            append_colmap_log(job_id, text)

        try:
            append_colmap_log(job_id, "$ " + " ".join(colmap_command) + "\n")
            exit_code = docker_exec_stream(colmap_command, log)
            if exit_code != 0:
                update_colmap_job(
                    job_id,
                    status="failed",
                    finished_at=utc_now(),
                    exit_code=exit_code,
                    error=f"COLMAP exited with code {exit_code}",
                )
                return

            frame_count = len(list((scene_host / "images").glob("*")))
            update_colmap_job(job_id, frame_count=frame_count)

            append_colmap_log(job_id, "\n$ " + " ".join(propose_command) + "\n")
            exit_code = docker_exec_stream(propose_command, log)
            if exit_code != 0:
                update_colmap_job(
                    job_id,
                    status="failed",
                    finished_at=utc_now(),
                    exit_code=exit_code,
                    error=f"Ground proposal exited with code {exit_code}",
                )
                return

            proposal_count = len(list(proposals_host.glob("candidate_*_zy.png")))
            proposals_json = proposals_host / "ground_proposals.json"
            if proposals_json.exists():
                try:
                    proposal_count = len(json.loads(proposals_json.read_text()).get("candidates", []))
                except Exception as exc:
                    append_colmap_log(job_id, f"\n[frontend] Failed to read proposal count: {exc}\n")

            update_colmap_job(
                job_id,
                status="proposed",
                finished_at=utc_now(),
                exit_code=0,
                frame_count=frame_count,
                proposal_count=proposal_count,
            )
        except Exception as exc:
            append_colmap_log(job_id, f"\n[frontend] {type(exc).__name__}: {exc}\n")
            update_colmap_job(
                job_id,
                status="failed",
                finished_at=utc_now(),
                exit_code=None,
                error=str(exc),
            )


@app.on_event("startup")
def prepare_storage() -> None:
    JOB_ROOT.mkdir(parents=True, exist_ok=True)
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    PANORAMA_ROOT.mkdir(parents=True, exist_ok=True)
    (JOB_ROOT / "colmap").mkdir(parents=True, exist_ok=True)
    with state_lock:
        state = load_state()
        for job in state.get("jobs", {}).values():
            if job.get("status") in {"queued", "running"}:
                job["status"] = "interrupted"
                job["error"] = "Frontend restarted while this job was active."
                job["updated_at"] = utc_now()
        for job in state.get("colmap_jobs", {}).values():
            if job.get("status") in {"queued", "running"}:
                job["status"] = "interrupted"
                job["error"] = "Frontend restarted while this COLMAP job was active."
                job["updated_at"] = utc_now()
        save_state(state)


def read_cpu_sample() -> tuple[int, int]:
    with open("/proc/stat", "r", encoding="utf-8") as handle:
        fields = handle.readline().split()
    if not fields or fields[0] != "cpu":
        raise RuntimeError("Could not read aggregate CPU stats")
    values = [int(value) for value in fields[1:]]
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    total = sum(values)
    return idle, total


def cpu_utilization_percent() -> float | None:
    global last_cpu_sample
    with metrics_lock:
        previous = last_cpu_sample
        if previous is None:
            previous = read_cpu_sample()
            time.sleep(0.05)
        current = read_cpu_sample()
        last_cpu_sample = current
    idle_delta = current[0] - previous[0]
    total_delta = current[1] - previous[1]
    if total_delta <= 0:
        return None
    busy = 1.0 - (idle_delta / total_delta)
    return round(max(0.0, min(100.0, busy * 100.0)), 1)


def parse_optional_float(value: str) -> float | None:
    text = value.strip()
    if not text or text.lower() in {"[not supported]", "not supported", "n/a"}:
        return None
    try:
        return float(text.split()[0])
    except ValueError:
        return None


def parse_gpu_metrics(output: str) -> list[dict[str, Any]]:
    gpus = []
    reader = csv.reader(io.StringIO(output))
    for fallback_index, row in enumerate(reader):
        if not row:
            continue
        values = [item.strip() for item in row]
        if len(values) < 6:
            continue
        index_text, uuid, name, utilization_text, memory_used_text, memory_total_text = values[:6]
        try:
            index = int(index_text)
        except ValueError:
            index = fallback_index
        utilization = parse_optional_float(utilization_text)
        memory_used = parse_optional_float(memory_used_text)
        memory_total = parse_optional_float(memory_total_text)
        memory_percent = None
        if memory_used is not None and memory_total and memory_total > 0:
            memory_percent = round(max(0.0, min(100.0, memory_used / memory_total * 100.0)), 1)
        gpus.append(
            {
                "index": index,
                "uuid": uuid,
                "name": name,
                "utilization": round(utilization, 1) if utilization is not None else None,
                "memory_used_mb": round(memory_used, 1) if memory_used is not None else None,
                "memory_total_mb": round(memory_total, 1) if memory_total is not None else None,
                "memory_percent": memory_percent,
                "process_count": 0,
                "process_memory_mb": 0.0,
            }
        )
    return gpus


def attach_gpu_process_metrics(gpus: list[dict[str, Any]], output: str) -> None:
    by_uuid = {gpu.get("uuid"): gpu for gpu in gpus if gpu.get("uuid")}
    reader = csv.reader(io.StringIO(output))
    for row in reader:
        if len(row) < 4:
            continue
        uuid, _pid, _process_name, used_memory_text = [item.strip() for item in row[:4]]
        gpu = by_uuid.get(uuid)
        if gpu is None:
            continue
        used_memory = parse_optional_float(used_memory_text) or 0.0
        gpu["process_count"] = int(gpu.get("process_count") or 0) + 1
        gpu["process_memory_mb"] = round(float(gpu.get("process_memory_mb") or 0.0) + used_memory, 1)


def local_gpu_metrics() -> list[dict[str, Any]]:
    if shutil.which("nvidia-smi") is None:
        return []
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=1.5,
    )
    if result.returncode != 0:
        return []
    gpus = parse_gpu_metrics(result.stdout)
    try:
        process_result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=1.5,
        )
        if process_result.returncode == 0:
            attach_gpu_process_metrics(gpus, process_result.stdout)
    except Exception:
        pass
    return gpus


def container_nvidia_smi(client: docker.DockerClient, query: str) -> str | None:
    container = client.containers.get(RESPLAT_CONTAINER)
    exec_id = client.api.exec_create(
        container.id,
        [
            "nvidia-smi",
            query,
            "--format=csv,noheader,nounits",
        ],
    )["Id"]
    output = client.api.exec_start(exec_id, stream=False)
    result = client.api.exec_inspect(exec_id)
    if int(result.get("ExitCode", 1)) != 0:
        return None
    return output.decode("utf-8", errors="replace")


def container_gpu_metrics() -> list[dict[str, Any]]:
    client = docker.from_env()
    output = container_nvidia_smi(
        client,
        "--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total",
    )
    if output is None:
        return []
    gpus = parse_gpu_metrics(output)
    try:
        process_output = container_nvidia_smi(
            client,
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        )
        if process_output is not None:
            attach_gpu_process_metrics(gpus, process_output)
    except Exception:
        pass
    return gpus


def gpu_metrics() -> list[dict[str, Any]]:
    global gpu_metrics_cache
    now = time.monotonic()
    with metrics_lock:
        if gpu_metrics_cache is not None and now - gpu_metrics_cache[0] < 2.5:
            return gpu_metrics_cache[1]
    try:
        gpus = local_gpu_metrics()
    except Exception:
        gpus = []
    if not gpus:
        try:
            gpus = container_gpu_metrics()
        except Exception:
            gpus = []
    with metrics_lock:
        gpu_metrics_cache = (now, gpus)
    return gpus


@app.get("/api/system-metrics")
def system_metrics() -> dict[str, Any]:
    cpu = cpu_utilization_percent()
    gpus = gpu_metrics()
    utilization_values = [
        float(item["utilization"])
        for item in gpus
        if item.get("utilization") is not None
    ]
    memory_used_mb = sum(float(item.get("memory_used_mb") or 0.0) for item in gpus)
    memory_total_mb = sum(float(item.get("memory_total_mb") or 0.0) for item in gpus)
    process_count = sum(int(item.get("process_count") or 0) for item in gpus)
    if gpus:
        gpu_average = round(sum(utilization_values) / len(utilization_values), 1) if utilization_values else None
        gpu_max = round(max(utilization_values), 1) if utilization_values else None
        memory_percent = (
            round(max(0.0, min(100.0, memory_used_mb / memory_total_mb * 100.0)), 1)
            if memory_total_mb > 0
            else None
        )
        active = bool(
            (gpu_max is not None and gpu_max > 0)
            or process_count > 0
            or (memory_percent is not None and memory_percent >= 1.0)
        )
    else:
        gpu_average = None
        gpu_max = None
        memory_percent = None
        active = False
    return {
        "timestamp": utc_now(),
        "cpu": {"utilization": cpu},
        "gpu": {
            "available": bool(gpus),
            "active": active,
            "utilization": gpu_max,
            "average_utilization": gpu_average,
            "memory": {
                "used_mb": round(memory_used_mb, 1) if gpus else None,
                "total_mb": round(memory_total_mb, 1) if gpus else None,
                "percent": memory_percent,
            },
            "process_count": process_count,
            "gpus": gpus,
        },
    }


@app.get("/api/models")
def list_models() -> dict[str, Any]:
    return {"models": [model_public(model) for model in MODEL_OPTIONS]}


@app.get("/", response_class=HTMLResponse)
def index(tool: str = "colmap") -> HTMLResponse:
    return HTMLResponse(
        render_index(tool),
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.get("/api/jobs")
def list_jobs() -> dict[str, Any]:
    with state_lock:
        jobs = list(load_state()["jobs"].values())
    jobs.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    return {"jobs": [job_public(job) for job in jobs]}


@app.get("/api/colmap-jobs")
def list_colmap_jobs() -> dict[str, Any]:
    with state_lock:
        jobs = list(load_state()["colmap_jobs"].values())
    jobs.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    return {"jobs": [colmap_public(job) for job in jobs]}


@app.get("/api/colmap-datasets")
def list_colmap_datasets() -> dict[str, Any]:
    with state_lock:
        jobs = list(load_state()["colmap_jobs"].values())
    datasets = [dataset for job in jobs if (dataset := colmap_dataset_public(job))]
    datasets.sort(key=lambda item: item["id"], reverse=True)
    return {"datasets": datasets}


@app.get("/api/user-videos")
def list_user_videos() -> dict[str, Any]:
    root = users_root()
    videos = []
    if root.exists():
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            try:
                relative = path.relative_to(root).as_posix()
            except ValueError:
                continue
            if relative.startswith("webui-jobs/"):
                continue
            stat = path.stat()
            videos.append(
                {
                    "path": relative,
                    "name": path.name,
                    "size": stat.st_size,
                    "mtime": stat.st_mtime,
                }
            )
    return {"videos": videos}


@app.post("/api/uploads/init")
async def init_upload(request: Request) -> dict[str, Any]:
    payload = await request.json()
    filename = safe_slug(str(payload.get("filename") or "upload.bin"))
    total_size = int(payload.get("total_size") or 0)
    total_chunks = int(payload.get("total_chunks") or 0)
    if total_size <= 0:
        raise HTTPException(status_code=400, detail="total_size must be positive")
    if total_chunks <= 0 or total_chunks > 100000:
        raise HTTPException(status_code=400, detail="Invalid chunk count")

    upload_id = uuid.uuid4().hex
    root = upload_dir(upload_id)
    chunks = root / "chunks"
    chunks.mkdir(parents=True, exist_ok=False)
    meta = {
        "id": upload_id,
        "filename": filename,
        "original_filename": payload.get("filename"),
        "total_size": total_size,
        "total_chunks": total_chunks,
        "created_at": utc_now(),
        "status": "uploading",
    }
    (root / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"upload_id": upload_id}


@app.post("/api/uploads/{upload_id}/chunks")
async def upload_chunk(upload_id: str, request: Request, index: int) -> dict[str, Any]:
    meta = upload_meta(upload_id)
    if meta.get("status") not in {"uploading", "complete"}:
        raise HTTPException(status_code=400, detail="Upload is not writable")
    if index < 0 or index >= int(meta["total_chunks"]):
        raise HTTPException(status_code=400, detail="Chunk index out of range")

    chunk_path = upload_dir(upload_id) / "chunks" / f"{index:08d}.part"
    tmp_path = chunk_path.with_suffix(".tmp")
    data = await request.body()
    with tmp_path.open("wb") as handle:
        handle.write(data)
    tmp_path.replace(chunk_path)
    return {"upload_id": upload_id, "index": index, "bytes": len(data)}


@app.post("/api/uploads/{upload_id}/complete")
def complete_upload(upload_id: str) -> dict[str, Any]:
    meta = upload_meta(upload_id)
    root = upload_dir(upload_id)
    chunks = root / "chunks"
    complete_dir = root / "complete"
    complete_dir.mkdir(exist_ok=True)
    output_path = complete_dir / meta["filename"]

    with output_path.open("wb") as output:
        for index in range(int(meta["total_chunks"])):
            chunk_path = chunks / f"{index:08d}.part"
            if not chunk_path.exists():
                raise HTTPException(status_code=400, detail=f"Missing chunk {index}")
            with chunk_path.open("rb") as handle:
                shutil.copyfileobj(handle, output)

    size = output_path.stat().st_size
    if size != int(meta["total_size"]):
        output_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Upload size mismatch")

    meta.update({"status": "complete", "path": str(output_path), "completed_at": utc_now()})
    (root / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    shutil.rmtree(chunks, ignore_errors=True)
    return {"upload_id": upload_id, "filename": meta["filename"], "bytes": size}


@app.post("/api/panorama")
async def create_panorama(
    file: UploadFile = File(...),
    output_width: int = Form(1920),
    output_height: int = Form(1080),
    hfov_deg: float = Form(60.0),
    yaw_step_deg: float = Form(20.0),
) -> dict[str, Any]:
    filename = safe_slug(file.filename or "panorama.jpg")
    if Path(filename).suffix.lower() not in IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Upload a panorama image")
    if output_width < 128 or output_width > 4096:
        raise HTTPException(status_code=400, detail="Output width must be 128..4096")
    if output_height < 128 or output_height > 4096:
        raise HTTPException(status_code=400, detail="Output height must be 128..4096")
    if hfov_deg <= 0 or hfov_deg >= 180:
        raise HTTPException(status_code=400, detail="Horizontal FOV must be between 0 and 180")
    if yaw_step_deg <= 0 or yaw_step_deg > 180:
        raise HTTPException(status_code=400, detail="Yaw step must be between 0 and 180")

    panorama_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    root = panorama_dir(panorama_id)
    outputs = root / "rectified"
    root.mkdir(parents=True, exist_ok=True)
    outputs.mkdir(parents=True, exist_ok=True)
    source_path = root / filename
    with source_path.open("wb") as handle:
        while chunk := await file.read(1024 * 1024):
            handle.write(chunk)

    try:
        image = Image.open(source_path)
        image.load()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}") from exc

    starts = [float(value) for value in np.arange(0.0, 360.0, yaw_step_deg)]
    files = []
    for index, start_deg in enumerate(starts):
        end_deg = start_deg + hfov_deg
        yaw_center = start_deg + hfov_deg / 2.0
        rectified = perspective_from_equirectangular(
            image,
            yaw_center,
            hfov_deg,
            output_width,
            output_height,
        )
        name = f"{index:03d}_yaw_{int(round(start_deg)):03d}_{int(round(end_deg)):03d}.jpg"
        out_path = outputs / name
        rectified.save(out_path, quality=95, subsampling=1)
        files.append(
            {
                "name": name,
                "start_deg": start_deg,
                "end_deg": end_deg,
                "url": f"/api/panorama/{panorama_id}/files/{name}",
            }
        )

    zip_path = root / "rectified.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in files:
            archive.write(outputs / item["name"], item["name"])

    return {
        "panorama": {
            "id": panorama_id,
            "source": filename,
            "source_size": {"width": image.width, "height": image.height},
            "output_size": {"width": output_width, "height": output_height},
            "hfov_deg": hfov_deg,
            "yaw_step_deg": yaw_step_deg,
            "files": files,
            "zip": f"/api/panorama/{panorama_id}/rectified.zip",
        }
    }


@app.post("/api/colmap-jobs")
async def create_colmap_job(request: Request) -> dict[str, Any]:
    try:
        form = await request.form(max_files=250, max_fields=100)
    except Exception:
        raise
    files = form_uploads(form)
    upload_ids = form_upload_ids(form)
    existing_videos = form_existing_videos(form)
    sample_fps = form_float(form, "sample_fps")
    start_time = form_float(form, "start_time", None)
    end_time = form_float(form, "end_time", None)
    matcher = form_string(form, "matcher", "auto")
    colmap_use_gpu = form_int(form, "colmap_use_gpu", 1)
    num_candidates = form_int(form, "num_candidates", 8)
    ransac_iterations = form_int(form, "ransac_iterations", 12000)

    if not files and not upload_ids and not existing_videos:
        raise HTTPException(status_code=400, detail="Select or upload at least one video")
    if sample_fps is None or sample_fps <= 0 or sample_fps > 60:
        raise HTTPException(status_code=400, detail="FPS must be between 0 and 60")
    if start_time is not None and start_time < 0:
        raise HTTPException(status_code=400, detail="Start time must be non-negative")
    if end_time is not None and end_time <= 0:
        raise HTTPException(status_code=400, detail="End time must be positive")
    if start_time is not None and end_time is not None and end_time <= start_time:
        raise HTTPException(status_code=400, detail="End time must be after start time")
    if matcher not in {"auto", "sequential", "exhaustive"}:
        raise HTTPException(status_code=400, detail="Matcher must be auto, sequential, or exhaustive")
    if colmap_use_gpu not in {0, 1}:
        raise HTTPException(status_code=400, detail="COLMAP GPU must be 0 or 1")
    if num_candidates < 1 or num_candidates > 16:
        raise HTTPException(status_code=400, detail="Candidate count must be 1..16")
    if ransac_iterations < 100 or ransac_iterations > 100000:
        raise HTTPException(status_code=400, detail="RANSAC iterations must be 100..100000")

    job_id = f"colmap-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    root = colmap_job_dir(job_id)
    upload_plan = direct_upload_plan(files)
    upload_plan.extend(staged_upload_plan(upload_ids, start_index=len(upload_plan)))
    upload_plan.extend(existing_video_plan(existing_videos, start_index=len(upload_plan)))

    raw_videos, work, proposals = reset_colmap_workspace(root)
    try:
        uploaded_names, source_manifest = await materialize_upload_plan(upload_plan, raw_videos)
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        raise

    (root / "source_manifest.json").write_text(
        json.dumps(source_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    scene = work / job_id / "scene"
    job = {
        "id": job_id,
        "display_name": job_id,
        "status": "queued",
        "sample_fps": sample_fps,
        "start_time": start_time,
        "end_time": end_time,
        "matcher": matcher,
        "colmap_use_gpu": colmap_use_gpu,
        "num_candidates": num_candidates,
        "ransac_iterations": ransac_iterations,
        "upload_count": len(uploaded_names),
        "uploads": uploaded_names,
        "source_manifest": source_manifest,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "paths": {
            "root": str(root),
            "raw_videos": str(raw_videos),
            "uploads": str(raw_videos),
            "work": str(work),
            "scene": str(scene),
            "proposals": str(proposals),
            "raw_videos_container": container_path(raw_videos),
            "uploads_container": container_path(raw_videos),
            "work_container": container_path(work),
            "scene_container": container_path(scene),
            "proposals_container": container_path(proposals),
        },
    }
    with state_lock:
        state = load_state()
        state["colmap_jobs"][job_id] = job
        save_state(state)

    append_colmap_log(
        job_id,
        (
            f"[frontend] Created COLMAP task {job_id}\n"
            f"[frontend] Raw videos: {raw_videos}\n"
            f"[frontend] Work dir: {work}\n"
            f"[frontend] Proposals: {proposals}\n"
            f"[frontend] Received {len(uploaded_names)} video file(s)\n"
            f"[frontend] Source files: {', '.join(uploaded_names)}\n\n"
        ),
    )
    threading.Thread(target=run_colmap_job, args=(job_id,), daemon=True).start()
    return {"job": colmap_public(job)}


@app.post("/api/jobs")
async def create_job(request: Request) -> dict[str, Any]:
    try:
        form = await request.form(max_files=250, max_fields=100)
    except Exception as exc:
        log_job_request(f"multipart parse failed: {type(exc).__name__}: {exc}")
        raise
    files = form_uploads(form)
    upload_ids = form_upload_ids(form)
    existing_videos = form_existing_videos(form)
    model_id = form_string(form, "model_id")
    source_type = form_string(form, "source_type", "colmap")
    dataset_id = form_string(form, "dataset_id", "")
    requested_resplat_mode = form_string(form, "resplat_mode", "single")
    sample_fps = form_float(form, "sample_fps")
    render_chunk_size = form_int(form, "render_chunk_size", 2)
    start_time = form_float(form, "start_time", None)
    end_time = form_float(form, "end_time", None)
    upload_names = [getattr(file, "filename", "") for file in files]
    log_job_request(
        (
            f"received multipart fields={list(form.keys())} "
            f"files={len(files)} staged_uploads={len(upload_ids)} existing_videos={len(existing_videos)} names={upload_names} "
            f"model_id={model_id!r} source_type={source_type!r} dataset_id={dataset_id!r} "
            f"resplat_mode={requested_resplat_mode!r} "
            f"sample_fps={sample_fps!r} "
            f"render_chunk_size={render_chunk_size!r} "
            f"start_time={start_time!r} end_time={end_time!r}"
        )
    )

    if model_id not in MODELS_BY_ID:
        reject_job_request(f"Unknown ReSplat model: {model_id!r}")
    if source_type != "colmap":
        reject_job_request("ReSplat pipeline jobs must use a saved COLMAP dataset")
    if files or upload_ids or existing_videos:
        reject_job_request("Video uploads are disabled for ReSplat pipeline jobs; use a COLMAP dataset")
    if requested_resplat_mode not in {"single", "batch_merge"}:
        reject_job_request("Unknown ReSplat mode")
    colmap_dataset = None
    if not dataset_id:
        reject_job_request("Select a COLMAP dataset")
    colmap_dataset = get_colmap_dataset(dataset_id)
    if requested_resplat_mode == "batch_merge" and not dataset_ground_alignment_path(colmap_dataset).exists():
        reject_job_request(
            "Batch merge requires a ground-aligned COLMAP dataset with ground_alignment.json"
        )
    if source_type == "video" and sample_fps is None:
        reject_job_request("sample_fps is required")
    if source_type == "video" and (sample_fps <= 0 or sample_fps > 60):
        reject_job_request("FPS must be between 0 and 60")
    if render_chunk_size <= 0 or render_chunk_size > 32:
        reject_job_request("Render chunk size must be between 1 and 32")
    if start_time is not None and start_time < 0:
        reject_job_request("Start time must be non-negative")
    if end_time is not None and end_time <= 0:
        reject_job_request("End time must be positive")
    if start_time is not None and end_time is not None and end_time <= start_time:
        reject_job_request("End time must be after start time")

    job_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    root = job_dir(job_id)

    model = MODELS_BY_ID[model_id]
    frame_count = int(colmap_dataset.get("frame_count") or 0) if colmap_dataset else 0
    batch_size = int(model["views"])
    batch_overlap = max(2, batch_size // 4)
    resplat_mode = requested_resplat_mode
    if source_type != "colmap":
        resplat_mode = "single"
    batch_count = plan_batch_count(frame_count, batch_size, batch_overlap) if resplat_mode == "batch_merge" else 1

    upload_plan = []
    if source_type == "video":
        upload_plan = direct_upload_plan(files)
        upload_plan.extend(staged_upload_plan(upload_ids, start_index=len(upload_plan)))
        upload_plan.extend(existing_video_plan(existing_videos, start_index=len(upload_plan)))

    raw_videos, work, results = reset_job_workspace(root)

    uploaded_names = []
    source_manifest = []
    try:
        uploaded_names, source_manifest = await materialize_upload_plan(upload_plan, raw_videos)
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        raise

    (root / "source_manifest.json").write_text(
        json.dumps(source_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    container_root = f"/workspace/resplat/users/webui-jobs/{job_id}"
    job = {
        "id": job_id,
        "display_name": job_id,
        "status": "queued",
        "source_type": source_type,
        "model_id": model_id,
        "model_preset": model["preset"],
        "resplat_mode": resplat_mode,
        "batch_size": batch_size if resplat_mode == "batch_merge" else None,
        "batch_overlap": batch_overlap if resplat_mode == "batch_merge" else None,
        "batch_count": batch_count if resplat_mode == "batch_merge" else None,
        "batch_manifest": None,
        "sample_fps": colmap_dataset.get("sample_fps") if colmap_dataset else sample_fps,
        "output_video_fps": OUTPUT_VIDEO_FPS,
        "render_chunk_size": render_chunk_size,
        "start_time": start_time,
        "end_time": end_time,
        "colmap_dataset": colmap_dataset,
        "upload_count": len(uploaded_names),
        "uploads": uploaded_names,
        "source_manifest": source_manifest,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "paths": {
            "root": str(root),
            "raw_videos": str(raw_videos),
            "uploads": str(raw_videos),
            "work": str(work),
            "results": str(results),
            "raw_videos_container": f"{container_root}/raw_videos",
            "uploads_container": f"{container_root}/raw_videos",
            "work_container": f"{container_root}/work",
            "results_container": f"{container_root}/results",
        },
    }
    with state_lock:
        state = load_state()
        state["jobs"][job_id] = job
        save_state(state)
    log_job_request(f"created job={job_id} upload_count={len(uploaded_names)} raw_videos={raw_videos}")

    append_log(
        job_id,
        "".join(
            [
                f"[frontend] Created isolated task {job_id}\n",
                f"[frontend] Source type: {source_type}\n",
                f"[frontend] ReSplat mode: {resplat_mode}\n",
                (
                    f"[frontend] Batch merge: {batch_count} batches, "
                    f"batch_size={batch_size}, overlap={batch_overlap}, "
                    "coordinate_frame=ground_aligned_z0\n"
                    if resplat_mode == "batch_merge" else ""
                ),
                (
                    f"[frontend] COLMAP dataset: {colmap_dataset['id']} "
                    f"candidate {colmap_dataset['candidate_id']}\n"
                    if colmap_dataset else ""
                ),
                f"[frontend] Raw videos: {raw_videos}\n",
                f"[frontend] Work dir: {work}\n",
                f"[frontend] Results dir: {results}\n",
                f"[frontend] Received {len(uploaded_names)} video file(s)\n",
                f"[frontend] Source files: {', '.join(uploaded_names)}\n\n",
            ]
        ),
    )
    threading.Thread(target=run_resplat_job, args=(job_id,), daemon=True).start()
    return {"job": job_public(job)}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    with state_lock:
        job = load_state()["jobs"].get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job": job_public(job)}


@app.get("/api/jobs/{job_id}/logs")
def get_logs(job_id: str) -> PlainTextResponse:
    with state_lock:
        exists = job_id in load_state()["jobs"]
    if not exists:
        raise HTTPException(status_code=404, detail="Job not found")
    log_path = job_dir(job_id) / "run.log"
    if not log_path.exists():
        return PlainTextResponse("")
    return PlainTextResponse(log_path.read_text(errors="replace"))


@app.get("/api/jobs/{job_id}/files/{filename}")
def get_file(request: Request, job_id: str, filename: str) -> FileResponse:
    if filename not in ALLOWED_DOWNLOADS:
        raise HTTPException(status_code=404, detail="File not found")
    with state_lock:
        job = load_state()["jobs"].get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    path = Path(job["paths"]["results"]) / filename
    if filename == "gaussians.ply" and "t" in request.query_params:
        preview_path = Path(job["paths"]["results"]) / "gaussians_preview.ply"
        if preview_path.exists():
            path = preview_path
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not ready")
    media_type = "video/mp4" if filename == "video.mp4" else "application/octet-stream"
    return FileResponse(path, media_type=media_type, filename=filename)


@app.get("/api/jobs/{job_id}/batches/{batch_index}/files/{filename}")
def get_batch_file(job_id: str, batch_index: int, filename: str) -> FileResponse:
    if filename not in {"video.mp4", "gaussians_global_core.ply"}:
        raise HTTPException(status_code=404, detail="File not found")
    with state_lock:
        job = load_state()["jobs"].get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if batch_index < 0:
        raise HTTPException(status_code=404, detail="Batch not found")
    path = Path(job["paths"]["results"]) / "batches" / f"batch_{batch_index:03d}" / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not ready")
    media_type = "video/mp4" if filename == "video.mp4" else "application/octet-stream"
    return FileResponse(path, media_type=media_type, filename=filename)


@app.get("/api/colmap-jobs/{job_id}")
def get_colmap_job(job_id: str) -> dict[str, Any]:
    with state_lock:
        job = load_state()["colmap_jobs"].get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="COLMAP job not found")
    return {"job": colmap_public(job)}


@app.get("/api/colmap-jobs/{job_id}/logs")
def get_colmap_logs(job_id: str) -> PlainTextResponse:
    with state_lock:
        exists = job_id in load_state()["colmap_jobs"]
    if not exists:
        raise HTTPException(status_code=404, detail="COLMAP job not found")
    log_path = colmap_job_dir(job_id) / "run.log"
    if not log_path.exists():
        return PlainTextResponse("")
    return PlainTextResponse(log_path.read_text(errors="replace"))


@app.get("/api/colmap-jobs/{job_id}/files/{filename}")
def get_colmap_file(job_id: str, filename: str) -> FileResponse:
    safe_name = safe_slug(filename)
    if safe_name != filename:
        raise HTTPException(status_code=404, detail="File not found")
    if filename not in COLMAP_ALLOWED_FILES and not re.fullmatch(r"candidate_\d{2}_zy\.png", filename):
        raise HTTPException(status_code=404, detail="File not found")
    with state_lock:
        job = load_state()["colmap_jobs"].get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="COLMAP job not found")
    path = Path(job["paths"]["proposals"]) / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not ready")
    return FileResponse(path, media_type="image/png", filename=filename)


@app.get("/api/colmap-jobs/{job_id}/debug/{filename}")
def get_colmap_debug_file(job_id: str, filename: str) -> FileResponse:
    safe_name = safe_slug(filename)
    if safe_name != filename or not re.fullmatch(r"reprojection_\d{2}_[A-Za-z0-9_.-]+\.jpg", filename):
        raise HTTPException(status_code=404, detail="File not found")
    with state_lock:
        job = load_state()["colmap_jobs"].get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="COLMAP job not found")
    aligned_scene = job.get("aligned_scene")
    if not aligned_scene:
        raise HTTPException(status_code=404, detail="Debug images not ready")
    path = Path(aligned_scene) / "debug_reprojection" / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not ready")
    return FileResponse(path, media_type="image/jpeg", filename=filename)


@app.post("/api/colmap-jobs/{job_id}/align")
def align_colmap_job(job_id: str, candidate_id: int = Form(...)) -> dict[str, Any]:
    with state_lock:
        job = load_state()["colmap_jobs"].get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="COLMAP job not found")
    if job.get("status") not in {"proposed", "ready"}:
        raise HTTPException(status_code=400, detail="COLMAP job has no ground proposals yet")
    if candidate_id < 0 or candidate_id > 99:
        raise HTTPException(status_code=400, detail="Invalid candidate id")

    root = colmap_job_dir(job_id)
    aligned_scene = root / f"aligned-candidate{candidate_id:02d}"
    command = [
        "python",
        "scripts/align_colmap_ground.py",
        "apply",
        "--proposals",
        f"{job['paths']['proposals_container']}/ground_proposals.json",
        "--candidate_id",
        str(candidate_id),
        "--scene_path",
        job["paths"]["scene_container"],
        "--output_scene_path",
        container_path(aligned_scene),
        "--output_sparse_dir",
        "sparse",
        "--images_dir",
        "images",
        "--image_mode",
        "copy",
        "--overwrite",
    ]

    update_colmap_job(job_id, status="aligning", selected_candidate_id=candidate_id)
    append_colmap_log(job_id, "\n$ " + " ".join(command) + "\n")
    try:
        exit_code = docker_exec_stream(
            command,
            lambda text: append_colmap_log(job_id, text),
        )
    except Exception as exc:
        append_colmap_log(job_id, f"\n[frontend] {type(exc).__name__}: {exc}\n")
        update_colmap_job(job_id, status="failed", error=str(exc), exit_code=None)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if exit_code != 0:
        update_colmap_job(
            job_id,
            status="failed",
            exit_code=exit_code,
            error=f"Ground alignment exited with code {exit_code}",
        )
        raise HTTPException(status_code=500, detail=f"Ground alignment exited with code {exit_code}")

    job = update_colmap_job(
        job_id,
        status="ready",
        exit_code=0,
        aligned_scene=str(aligned_scene),
        aligned_scene_container=container_path(aligned_scene),
        aligned_sparse_dir="sparse",
        selected_candidate_id=candidate_id,
    )
    return {"job": colmap_public(job), "dataset": colmap_dataset_public(job)}


@app.get("/api/panorama/{panorama_id}/files/{filename}")
def get_panorama_file(panorama_id: str, filename: str) -> FileResponse:
    safe_name = safe_slug(filename)
    if safe_name != filename or Path(filename).suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise HTTPException(status_code=404, detail="File not found")
    path = panorama_dir(panorama_id) / "rectified" / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type="image/jpeg", filename=filename)


@app.post("/api/panorama/{panorama_id}/seedance")
def start_seedance_for_panorama(
    panorama_id: str,
    token: str = Form(...),
    prompt: str = Form("相机视角平移前进5米"),
    public_base_url: str = Form(...),
    model_version: str = Form("standard"),
    duration: int = Form(5),
    resolution: str = Form("720p"),
    aspect_ratio: str = Form("16:9"),
    generate_audio: bool = Form(True),
) -> dict[str, Any]:
    if not token.strip():
        raise HTTPException(status_code=400, detail="Seedance token is required")
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt is required")
    if model_version not in {"standard", "fast"}:
        raise HTTPException(status_code=400, detail="Model version must be standard or fast")
    if duration < 1 or duration > 30:
        raise HTTPException(status_code=400, detail="Duration must be 1..30 seconds")
    if resolution not in {"480p", "720p", "1080p"}:
        raise HTTPException(status_code=400, detail="Resolution must be 480p, 720p, or 1080p")

    root = panorama_dir(panorama_id)
    outputs = root / "rectified"
    if not outputs.exists():
        raise HTTPException(status_code=404, detail="Panorama rectified images not found")

    image_paths = sorted(outputs.glob("*.jpg"))
    if not image_paths:
        raise HTTPException(status_code=404, detail="No rectified images found")

    headers = {
        "Authorization": f"Bearer {token.strip()}",
        "Content-Type": "application/json",
    }
    results = []
    for image_path in image_paths:
        image_path_part = f"/api/panorama/{panorama_id}/files/{image_path.name}"
        image_url = public_url(public_base_url, image_path_part)
        payload = {
            "name": f"resplat-{panorama_id}-{image_path.stem}",
            "mode": "image-to-video",
            "prompt": prompt,
            "image_url": image_url,
            "model_version": model_version,
            "duration": duration,
            "resolution": resolution,
            "aspect_ratio": aspect_ratio,
            "generate_audio": generate_audio,
        }
        try:
            response = requests.post(
                SEEDANCE_START_URL,
                headers=headers,
                json=payload,
                timeout=60,
            )
            response_json = response.json()
        except Exception as exc:
            results.append(
                {
                    "image": image_path.name,
                    "image_url": image_url,
                    "ok": False,
                    "error": str(exc),
                }
            )
            continue

        results.append(
            {
                "image": image_path.name,
                "image_url": image_url,
                "ok": response.ok and response_json.get("code") == 0,
                "status_code": response.status_code,
                "response": response_json,
            }
        )

    record = {
        "created_at": utc_now(),
        "prompt": prompt,
        "model_version": model_version,
        "duration": duration,
        "resolution": resolution,
        "aspect_ratio": aspect_ratio,
        "generate_audio": generate_audio,
        "public_base_url": public_base_url,
        "results": results,
    }
    (root / "seedance_runs.json").write_text(json.dumps(record, indent=2, ensure_ascii=False))
    return {"seedance": record}


@app.get("/api/panorama/{panorama_id}/rectified.zip")
def get_panorama_zip(panorama_id: str) -> FileResponse:
    path = panorama_dir(panorama_id) / "rectified.zip"
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type="application/zip", filename="rectified.zip")


app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")

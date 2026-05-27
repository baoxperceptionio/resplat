from __future__ import annotations

import json
import os
import re
import shlex
import shutil
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
ALLOWED_DOWNLOADS = {"video.mp4", "gaussians.ply"}
COLMAP_ALLOWED_FILES = {"overview_zy.png"}

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

app = FastAPI(title="ReSplat Web UI")
state_lock = threading.Lock()
run_semaphore = threading.Semaphore(MAX_CONCURRENT_JOBS)


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


def job_public(job: dict[str, Any]) -> dict[str, Any]:
    result_dir = Path(job["paths"]["results"])
    video_path = result_dir / "video.mp4"
    ply_path = result_dir / "gaussians.ply"
    enriched = dict(job)
    video_version = None
    ply_version = None
    if video_path.exists():
        stat = video_path.stat()
        video_version = f"{int(stat.st_mtime_ns)}-{stat.st_size}"
    if ply_path.exists():
        stat = ply_path.stat()
        ply_version = f"{int(stat.st_mtime_ns)}-{stat.st_size}"
    enriched["artifacts"] = {
        "video": f"/api/jobs/{job['id']}/files/video.mp4?v={video_version}" if video_version else None,
        "ply": f"/api/jobs/{job['id']}/files/gaussians.ply?v={ply_version}" if ply_version else None,
        "video_version": video_version,
        "ply_version": ply_version,
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
    return enriched


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
                update_job(job_id, status="succeeded", finished_at=utc_now(), exit_code=exit_code)
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

            update_colmap_job(
                job_id,
                status="proposed",
                finished_at=utc_now(),
                exit_code=0,
                frame_count=frame_count,
                proposal_count=len(list(proposals_host.glob("candidate_*_zy.png"))),
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
    source_type = form_string(form, "source_type", "video")
    dataset_id = form_string(form, "dataset_id", "")
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
            f"sample_fps={sample_fps!r} "
            f"render_chunk_size={render_chunk_size!r} "
            f"start_time={start_time!r} end_time={end_time!r}"
        )
    )

    if model_id not in MODELS_BY_ID:
        reject_job_request(f"Unknown ReSplat model: {model_id!r}")
    if source_type not in {"video", "colmap"}:
        reject_job_request("Unknown source type")
    colmap_dataset = None
    if source_type == "colmap":
        if not dataset_id:
            reject_job_request("Select a COLMAP dataset")
        colmap_dataset = get_colmap_dataset(dataset_id)
    elif not files and not upload_ids and not existing_videos:
        reject_job_request("Select or upload at least one video")
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
        "model_preset": MODELS_BY_ID[model_id]["preset"],
        "sample_fps": sample_fps,
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
def get_file(job_id: str, filename: str) -> FileResponse:
    if filename not in ALLOWED_DOWNLOADS:
        raise HTTPException(status_code=404, detail="File not found")
    with state_lock:
        job = load_state()["jobs"].get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    path = Path(job["paths"]["results"]) / filename
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

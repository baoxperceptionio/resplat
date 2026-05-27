from __future__ import annotations

import json
import os
import re
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
        return {"jobs": {}}
    try:
        return json.loads(STATE_PATH.read_text())
    except json.JSONDecodeError:
        return {"jobs": {}}


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


def append_log(job_id: str, text: str) -> None:
    log_path = job_dir(job_id) / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", errors="replace") as handle:
        handle.write(text)


def job_dir(job_id: str) -> Path:
    return JOB_ROOT / job_id


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


def model_public(model: dict[str, Any]) -> dict[str, Any]:
    item = dict(model)
    item["checkpoint_exists"] = (REPO_ROOT / model["checkpoint"]).exists()
    return item


def render_index(tool: str) -> str:
    tool = tool if tool in {"pipeline", "panorama"} else "pipeline"
    html = (Path(__file__).parent / "static" / "index.html").read_text()
    if tool == "pipeline":
        return html

    replacements = {
        '<body data-tool="pipeline">': '<body data-tool="panorama">',
        '<a class="tab-button active" href="/?tool=pipeline" data-tab="pipeline" aria-selected="true">视频 pipeline</a>':
            '<a class="tab-button" href="/?tool=pipeline" data-tab="pipeline" aria-selected="false">视频 pipeline</a>',
        '<a class="tab-button" href="/?tool=panorama" data-tab="panorama" aria-selected="false">全景切图</a>':
            '<a class="tab-button active" href="/?tool=panorama" data-tab="panorama" aria-selected="true">全景切图</a>',
        '<div class="tab-panel active" data-panel="pipeline">':
            '<div class="tab-panel" data-panel="pipeline" hidden>',
        '<div class="tab-panel" data-panel="panorama" hidden>':
            '<div class="tab-panel active" data-panel="panorama">',
        '<div class="actions">': '<div class="actions" hidden>',
        '<div id="pipelineWorkspace" class="workspace-view active">':
            '<div id="pipelineWorkspace" class="workspace-view" hidden>',
        '<div id="panoramaWorkspace" class="workspace-view panorama-view" hidden>':
            '<div id="panoramaWorkspace" class="workspace-view panorama-view active">',
        '<p id="activeTitle">等待任务</p>':
            '<p id="activeTitle">全景切图</p>',
        '<span id="activeMeta">上传视频后会在这里显示产物</span>':
            '<span id="activeMeta">60 度水平视角，起点每 20 度生成一张</span>',
    }
    for before, after in replacements.items():
        html = html.replace(before, after)
    return html


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


@app.on_event("startup")
def prepare_storage() -> None:
    JOB_ROOT.mkdir(parents=True, exist_ok=True)
    PANORAMA_ROOT.mkdir(parents=True, exist_ok=True)
    with state_lock:
        state = load_state()
        for job in state.get("jobs", {}).values():
            if job.get("status") in {"queued", "running"}:
                job["status"] = "interrupted"
                job["error"] = "Frontend restarted while this job was active."
                job["updated_at"] = utc_now()
        save_state(state)


@app.get("/api/models")
def list_models() -> dict[str, Any]:
    return {"models": [model_public(model) for model in MODEL_OPTIONS]}


@app.get("/", response_class=HTMLResponse)
def index(tool: str = "pipeline") -> HTMLResponse:
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


@app.post("/api/jobs")
async def create_job(request: Request) -> dict[str, Any]:
    try:
        form = await request.form(max_files=250, max_fields=100)
    except Exception as exc:
        log_job_request(f"multipart parse failed: {type(exc).__name__}: {exc}")
        raise
    files = form_uploads(form)
    model_id = form_string(form, "model_id")
    sample_fps = form_float(form, "sample_fps")
    render_chunk_size = form_int(form, "render_chunk_size", 2)
    start_time = form_float(form, "start_time", None)
    end_time = form_float(form, "end_time", None)
    upload_names = [getattr(file, "filename", "") for file in files]
    log_job_request(
        (
            f"received multipart fields={list(form.keys())} "
            f"files={len(files)} names={upload_names} "
            f"model_id={model_id!r} sample_fps={sample_fps!r} "
            f"render_chunk_size={render_chunk_size!r} "
            f"start_time={start_time!r} end_time={end_time!r}"
        )
    )

    if model_id not in MODELS_BY_ID:
        reject_job_request(f"Unknown ReSplat model: {model_id!r}")
    if not files:
        reject_job_request("Upload at least one video")
    if sample_fps is None:
        reject_job_request("sample_fps is required")
    if sample_fps <= 0 or sample_fps > 60:
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
    for index, upload in enumerate(files):
        filename = normalized_video_filename(upload, index)
        upload_plan.append((index, upload, filename))

    raw_videos, work, results = reset_job_workspace(root)

    uploaded_names = []
    source_manifest = []
    try:
        for index, upload, filename in upload_plan:
            target = raw_videos / f"{index:03d}-{filename}"
            bytes_written = 0
            with target.open("wb") as handle:
                while chunk := await upload.read(1024 * 1024):
                    bytes_written += len(chunk)
                    handle.write(chunk)
            uploaded_names.append(target.name)
            source_manifest.append(
                {
                    "index": index,
                    "original_name": upload.filename,
                    "stored_name": target.name,
                    "bytes": bytes_written,
                }
            )
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
        "model_id": model_id,
        "model_preset": MODELS_BY_ID[model_id]["preset"],
        "sample_fps": sample_fps,
        "render_chunk_size": render_chunk_size,
        "start_time": start_time,
        "end_time": end_time,
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
        (
            f"[frontend] Created isolated task {job_id}\n"
            f"[frontend] Raw videos: {raw_videos}\n"
            f"[frontend] Work dir: {work}\n"
            f"[frontend] Results dir: {results}\n"
            f"[frontend] Received {len(uploaded_names)} video file(s)\n"
            f"[frontend] Source files: {', '.join(uploaded_names)}\n\n"
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

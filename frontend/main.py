from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import docker
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image


REPO_ROOT = Path(os.getenv("RESPLAT_REPO", "/workspace/resplat")).resolve()
JOB_ROOT = Path(os.getenv("FRONTEND_JOB_ROOT", REPO_ROOT / "users" / "webui-jobs")).resolve()
RESPLAT_CONTAINER = os.getenv("RESPLAT_CONTAINER", "resplat")
MAX_CONCURRENT_JOBS = max(1, int(os.getenv("MAX_CONCURRENT_JOBS", "1")))
STATE_PATH = JOB_ROOT / "jobs.json"
PANORAMA_ROOT = JOB_ROOT / "panorama"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v"}
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


def job_public(job: dict[str, Any]) -> dict[str, Any]:
    result_dir = Path(job["paths"]["results"])
    video_path = result_dir / "video.mp4"
    ply_path = result_dir / "gaussians.ply"
    enriched = dict(job)
    enriched["artifacts"] = {
        "video": f"/api/jobs/{job['id']}/files/video.mp4" if video_path.exists() else None,
        "ply": f"/api/jobs/{job['id']}/files/gaussians.ply" if ply_path.exists() else None,
    }
    return enriched


def model_public(model: dict[str, Any]) -> dict[str, Any]:
    item = dict(model)
    item["checkpoint_exists"] = (REPO_ROOT / model["checkpoint"]).exists()
    return item


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


@app.get("/api/jobs")
def list_jobs() -> dict[str, Any]:
    with state_lock:
        jobs = list(load_state()["jobs"].values())
    jobs.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    return {"jobs": [job_public(job) for job in jobs]}


@app.post("/api/panorama")
async def create_panorama(
    file: UploadFile = File(...),
    output_width: int = Form(576),
    output_height: int = Form(1024),
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
async def create_job(
    files: list[UploadFile] = File(...),
    model_id: str = Form(...),
    sample_fps: float = Form(...),
    render_chunk_size: int = Form(2),
    start_time: float | None = Form(None),
    end_time: float | None = Form(None),
) -> dict[str, Any]:
    if model_id not in MODELS_BY_ID:
        raise HTTPException(status_code=400, detail="Unknown ReSplat model")
    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one video")
    if sample_fps <= 0 or sample_fps > 60:
        raise HTTPException(status_code=400, detail="FPS must be between 0 and 60")
    if render_chunk_size <= 0 or render_chunk_size > 32:
        raise HTTPException(status_code=400, detail="Render chunk size must be between 1 and 32")
    if start_time is not None and start_time < 0:
        raise HTTPException(status_code=400, detail="Start time must be non-negative")
    if end_time is not None and end_time <= 0:
        raise HTTPException(status_code=400, detail="End time must be positive")
    if start_time is not None and end_time is not None and end_time <= start_time:
        raise HTTPException(status_code=400, detail="End time must be after start time")

    job_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    root = job_dir(job_id)
    uploads = root / "uploads"
    work = root / "work"
    results = root / "results"
    uploads.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)

    uploaded_names = []
    for index, upload in enumerate(files):
        filename = safe_slug(upload.filename or f"video-{index}.mp4")
        suffix = Path(filename).suffix.lower()
        if suffix not in VIDEO_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported video extension: {filename}")
        target = uploads / f"{index:03d}-{filename}"
        with target.open("wb") as handle:
            while chunk := await upload.read(1024 * 1024):
                handle.write(chunk)
        uploaded_names.append(target.name)

    container_root = f"/workspace/resplat/users/webui-jobs/{job_id}"
    job = {
        "id": job_id,
        "status": "queued",
        "model_id": model_id,
        "model_preset": MODELS_BY_ID[model_id]["preset"],
        "sample_fps": sample_fps,
        "render_chunk_size": render_chunk_size,
        "start_time": start_time,
        "end_time": end_time,
        "uploads": uploaded_names,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "paths": {
            "root": str(root),
            "uploads": str(uploads),
            "work": str(work),
            "results": str(results),
            "uploads_container": f"{container_root}/uploads",
            "work_container": f"{container_root}/work",
            "results_container": f"{container_root}/results",
        },
    }
    with state_lock:
        state = load_state()
        state["jobs"][job_id] = job
        save_state(state)

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


@app.get("/api/panorama/{panorama_id}/rectified.zip")
def get_panorama_zip(panorama_id: str) -> FileResponse:
    path = panorama_dir(panorama_id) / "rectified.zip"
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type="application/zip", filename="rectified.zip")


app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")

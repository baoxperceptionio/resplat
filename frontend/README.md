# ReSplat Web UI

Run the browser UI together with the CUDA ReSplat worker:

```bash
docker compose up -d frontend
```

Open `http://localhost:8000`.

The frontend stores uploaded videos, logs, COLMAP work files, and final artifacts under:

```text
users/webui-jobs/
```

Each job runs inside the existing `resplat` container through Docker's socket. The produced artifacts are served back from the job's `results` directory:

- `video.mp4`
- `gaussians.ply`

The in-browser 3DGS viewer uses SparkJS `2.1.0` with Three.js `0.180.0`.
After clicking the viewer, `W/A/S/D` translates across the ground plane, `Q/E`
translates vertically, and `R` restores the framed view.

The panorama tab converts an equirectangular panorama into rectified perspective
views. Defaults are 1920x1080 output images, 60 degree horizontal FOV, and 20
degree yaw-start steps, producing views from `000-060` through `340-400`.

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

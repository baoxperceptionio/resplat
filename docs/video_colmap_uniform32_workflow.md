# Video to ReSplat with 32 Uniform Context Views

This workflow is for long or difficult videos where sampling exactly 32 frames
for COLMAP may not register enough cameras. The robust path is:

1. Uniformly sample more frames from the video for COLMAP.
2. Run COLMAP reconstruction and undistortion.
3. Run ReSplat with `--context_selection uniform --num_context 32` on the
   registered COLMAP views.
4. Copy the COLMAP inputs and reconstruction into the final result directory.

The commands below assume the repository is mounted in the `resplat` Docker
container at `/workspace/resplat`.

## Example

Input video:

```bash
users/IMG_9188.MOV
```

If the video lives outside this repository, copy it into `users/` first so the
container can see it:

```bash
cp /home/ubuntu/vggt/users/IMG_9188.MOV users/IMG_9188.MOV
```

If CUDA is not visible inside the container, restart the worker:

```bash
docker compose restart resplat
docker exec resplat nvidia-smi
```

## 1. Build COLMAP from More Uniform Frames

Sample 128 frames uniformly from the video, run exhaustive COLMAP matching, and
stop before ReSplat:

```bash
docker exec resplat python scripts/video_to_resplat.py \
  --video users/IMG_9188.MOV \
  -N 128 \
  --scene_name IMG_9188-N128-colmap \
  --matcher exhaustive \
  --skip_resplat \
  --overwrite
```

Why 128 first? In the `IMG_9188.MOV` run, using exactly 32 uniformly sampled
frames only registered 5 COLMAP images. Sampling 128 frames gave COLMAP denser
overlap and registered 123 images, enough to select 32 ReSplat context views.

The COLMAP scene is written to:

```text
datasets/video_colmap/IMG_9188-N128-colmap/scene
```

Useful subdirectories:

```text
datasets/video_colmap/IMG_9188-N128-colmap/raw_images
datasets/video_colmap/IMG_9188-N128-colmap/colmap
datasets/video_colmap/IMG_9188-N128-colmap/scene/images
datasets/video_colmap/IMG_9188-N128-colmap/scene/sparse
```

## 2. Run ReSplat with 32 Uniform Context Views

Run the 32-view low-resolution preset and select 32 context images uniformly
from the registered COLMAP views:

```bash
docker exec resplat python scripts/infer_colmap.py \
  --model_preset dl3dv_32v_256x448 \
  --scene_path datasets/video_colmap/IMG_9188-N128-colmap/scene \
  --start_frame 0 \
  --frame_distance 128 \
  --images_dir images \
  --sparse_dir sparse \
  --output_dir results/IMG_9188-N128-uniform32-resplat \
  --num_context 32 \
  --context_selection uniform \
  --target_selection all \
  --save_images \
  --save_video \
  --save_ply \
  --render_chunk_size 2 \
  --no_eval \
  --image_shape 256 448
```

Expected confirmation in the log:

```text
Found 123 images
Context: 32 views (strategy: uniform)
Target: 123 views (strategy: all)
Context images: torch.Size([32, 3, 256, 448])
```

Main ReSplat outputs:

```text
results/IMG_9188-N128-uniform32-resplat/gaussians.ply
results/IMG_9188-N128-uniform32-resplat/video.mp4
results/IMG_9188-N128-uniform32-resplat/input
results/IMG_9188-N128-uniform32-resplat/rendered
```

## 3. Save COLMAP Images and Results with ReSplat Output

Create a `colmap/` archive inside the final result directory:

```bash
docker exec resplat mkdir -p \
  results/IMG_9188-N128-uniform32-resplat/colmap
```

Copy the sampled frames, undistorted COLMAP images, sparse model, and full
COLMAP workspace:

```bash
docker exec resplat cp -a \
  datasets/video_colmap/IMG_9188-N128-colmap/raw_images \
  results/IMG_9188-N128-uniform32-resplat/colmap/raw_images

docker exec resplat cp -a \
  datasets/video_colmap/IMG_9188-N128-colmap/scene/images \
  results/IMG_9188-N128-uniform32-resplat/colmap/undistorted_images

docker exec resplat cp -a \
  datasets/video_colmap/IMG_9188-N128-colmap/scene/sparse \
  results/IMG_9188-N128-uniform32-resplat/colmap/undistorted_sparse

docker exec resplat cp -a \
  datasets/video_colmap/IMG_9188-N128-colmap/colmap \
  results/IMG_9188-N128-uniform32-resplat/colmap/full_colmap_workspace
```

The resulting layout is:

```text
results/IMG_9188-N128-uniform32-resplat/
  gaussians.ply
  video.mp4
  input/
  rendered/
  colmap/
    raw_images/              # 128 uniformly sampled frames
    undistorted_images/      # registered undistorted images used by ReSplat
    undistorted_sparse/      # sparse model used by ReSplat
    full_colmap_workspace/   # database.db and original sparse outputs
```

Sanity checks:

```bash
docker exec resplat sh -lc \
  "find results/IMG_9188-N128-uniform32-resplat/colmap/raw_images -type f | wc -l"

docker exec resplat sh -lc \
  "find results/IMG_9188-N128-uniform32-resplat/colmap/undistorted_images -type f | wc -l"

docker exec resplat du -sh \
  results/IMG_9188-N128-uniform32-resplat/colmap
```

For `IMG_9188.MOV`, these were:

```text
128 raw sampled frames
123 undistorted registered images
172M archived COLMAP files
```

## Notes

- `--render_chunk_size` only chunks target-view rendering to reduce memory use.
  It does not increase the number of ReSplat input/context views.
- `--num_context 32 --context_selection uniform` selects 32 context views from
  the registered COLMAP images loaded by `infer_colmap.py`.
- The `dl3dv_32v_256x448` preset is the available 32-view preset in this
  repository. The high-resolution presets are 8-view and 16-view.

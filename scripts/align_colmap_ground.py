#!/usr/bin/env python3
"""Find and apply ground-plane alignments for COLMAP reconstructions.

The workflow is intentionally two-step:

1. propose: RANSAC candidate planes from points3D and save z-y projection plots.
2. apply: copy a COLMAP scene and transform images.bin + points3D.bin so the
   selected plane becomes z=0 in world coordinates.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import shutil
import struct
from pathlib import Path

import numpy as np


CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"]
)
ColmapCamera = collections.namedtuple(
    "ColmapCamera", ["id", "model", "width", "height", "params"]
)
BaseImage = collections.namedtuple(
    "ColmapImage",
    ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"],
)
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"]
)

CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
}
CAMERA_MODEL_IDS = {model.model_id: model for model in CAMERA_MODELS}
CAMERA_MODEL_NAMES = {model.model_name: model for model in CAMERA_MODELS}


class ColmapImage(BaseImage):
    def qvec2rotmat(self) -> np.ndarray:
        return qvec2rotmat(self.qvec)


def read_next_bytes(fid, num_bytes: int, format_char_sequence: str):
    data = fid.read(num_bytes)
    return struct.unpack("<" + format_char_sequence, data)


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
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
        dtype=np.float64,
    )


def rotmat2qvec(rotmat: np.ndarray) -> np.ndarray:
    m = np.asarray(rotmat, dtype=np.float64)
    k = np.array(
        [
            [m[0, 0] - m[1, 1] - m[2, 2], 0, 0, 0],
            [m[1, 0] + m[0, 1], m[1, 1] - m[0, 0] - m[2, 2], 0, 0],
            [m[2, 0] + m[0, 2], m[2, 1] + m[1, 2], m[2, 2] - m[0, 0] - m[1, 1], 0],
            [m[1, 2] - m[2, 1], m[2, 0] - m[0, 2], m[0, 1] - m[1, 0], m[0, 0] + m[1, 1] + m[2, 2]],
        ],
        dtype=np.float64,
    )
    k /= 3.0
    eigvals, eigvecs = np.linalg.eigh(k)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec / np.linalg.norm(qvec)


def read_cameras_binary(path: Path) -> dict[int, ColmapCamera]:
    cameras = {}
    with path.open("rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(fid, 24, "iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[model_id].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, 8 * num_params, "d" * num_params)
            cameras[camera_id] = ColmapCamera(
                id=camera_id,
                model=model_name,
                width=width,
                height=height,
                params=np.array(params, dtype=np.float64),
            )
    return cameras


def write_cameras_binary(cameras: dict[int, ColmapCamera], path: Path) -> None:
    with path.open("wb") as fid:
        fid.write(struct.pack("<Q", len(cameras)))
        for camera_id in sorted(cameras):
            cam = cameras[camera_id]
            model_id = CAMERA_MODEL_NAMES[cam.model].model_id
            fid.write(struct.pack("<iiQQ", cam.id, model_id, cam.width, cam.height))
            fid.write(struct.pack("<" + "d" * len(cam.params), *cam.params))


def read_images_binary(path: Path) -> dict[int, ColmapImage]:
    images = {}
    with path.open("rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            props = read_next_bytes(fid, 64, "idddddddi")
            image_id = props[0]
            qvec = np.array(props[1:5], dtype=np.float64)
            tvec = np.array(props[5:8], dtype=np.float64)
            camera_id = props[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            x_y_id_s = read_next_bytes(fid, 24 * num_points2D, "ddq" * num_points2D)
            xys = np.column_stack(
                [
                    tuple(map(float, x_y_id_s[0::3])),
                    tuple(map(float, x_y_id_s[1::3])),
                ]
            )
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])), dtype=np.int64)
            images[image_id] = ColmapImage(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=xys,
                point3D_ids=point3D_ids,
            )
    return images


def write_images_binary(images: dict[int, ColmapImage], path: Path) -> None:
    with path.open("wb") as fid:
        fid.write(struct.pack("<Q", len(images)))
        for image_id in sorted(images):
            img = images[image_id]
            fid.write(
                struct.pack(
                    "<idddddddi",
                    img.id,
                    *img.qvec.tolist(),
                    *img.tvec.tolist(),
                    img.camera_id,
                )
            )
            fid.write(img.name.encode("utf-8") + b"\x00")
            fid.write(struct.pack("<Q", len(img.xys)))
            for xy, point3d_id in zip(img.xys, img.point3D_ids):
                fid.write(struct.pack("<ddq", float(xy[0]), float(xy[1]), int(point3d_id)))


def read_points3d_binary(path: Path) -> dict[int, Point3D]:
    points3d = {}
    with path.open("rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            props = read_next_bytes(fid, 43, "QdddBBBd")
            point_id = props[0]
            xyz = np.array(props[1:4], dtype=np.float64)
            rgb = np.array(props[4:7], dtype=np.uint8)
            error = float(props[7])
            track_len = read_next_bytes(fid, 8, "Q")[0]
            track_elems = read_next_bytes(fid, 8 * track_len, "ii" * track_len)
            image_ids = np.array(track_elems[0::2], dtype=np.int32)
            point2d_idxs = np.array(track_elems[1::2], dtype=np.int32)
            points3d[point_id] = Point3D(
                id=point_id,
                xyz=xyz,
                rgb=rgb,
                error=error,
                image_ids=image_ids,
                point2D_idxs=point2d_idxs,
            )
    return points3d


def write_points3d_binary(points3d: dict[int, Point3D], path: Path) -> None:
    with path.open("wb") as fid:
        fid.write(struct.pack("<Q", len(points3d)))
        for point_id in sorted(points3d):
            point = points3d[point_id]
            fid.write(
                struct.pack(
                    "<QdddBBBd",
                    point.id,
                    float(point.xyz[0]),
                    float(point.xyz[1]),
                    float(point.xyz[2]),
                    int(point.rgb[0]),
                    int(point.rgb[1]),
                    int(point.rgb[2]),
                    float(point.error),
                )
            )
            fid.write(struct.pack("<Q", len(point.image_ids)))
            for image_id, point2d_idx in zip(point.image_ids, point.point2D_idxs):
                fid.write(struct.pack("<ii", int(image_id), int(point2d_idx)))


def sparse_path_for_scene(scene_path: Path, sparse_dir: str | None) -> Path:
    if sparse_dir:
        path = scene_path / sparse_dir
        if path.exists():
            return path
        raise FileNotFoundError(path)

    candidates = [scene_path / "sparse/0", scene_path / "sparse", scene_path]
    for path in candidates:
        if (path / "points3D.bin").exists() and (path / "images.bin").exists():
            return path
    raise FileNotFoundError(
        f"Could not find COLMAP sparse model under {scene_path}; tried sparse/0, sparse, and scene root"
    )


def read_model(sparse_path: Path):
    required = ["cameras.bin", "images.bin", "points3D.bin"]
    missing = [name for name in required if not (sparse_path / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing {missing} in {sparse_path}")
    return (
        read_cameras_binary(sparse_path / "cameras.bin"),
        read_images_binary(sparse_path / "images.bin"),
        read_points3d_binary(sparse_path / "points3D.bin"),
    )


def points_array(points3d: dict[int, Point3D]) -> tuple[np.ndarray, np.ndarray]:
    ids = np.array(sorted(points3d), dtype=np.uint64)
    points = np.stack([points3d[int(point_id)].xyz for point_id in ids], axis=0)
    return ids, points


def camera_centers(images: dict[int, ColmapImage]) -> np.ndarray:
    centers = []
    for image_id in sorted(images):
        img = images[image_id]
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = qvec2rotmat(img.qvec)
        w2c[:3, 3] = img.tvec
        c2w = np.linalg.inv(w2c)
        centers.append(c2w[:3, 3])
    return np.stack(centers, axis=0)


def camera_up_normal(images: dict[int, ColmapImage]) -> np.ndarray:
    """Estimate world up from COLMAP camera poses.

    COLMAP/OpenCV camera coordinates use +Y down, so camera-to-world column 1 is
    the downward direction. Averaging -Y over all registered cameras gives a
    robust up/ground-normal prior when the cameras are mostly level.
    """
    up_vectors = []
    for image_id in sorted(images):
        img = images[image_id]
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = qvec2rotmat(img.qvec)
        w2c[:3, 3] = img.tvec
        c2w = np.linalg.inv(w2c)
        up_vectors.append(-c2w[:3, 1])
    up = np.mean(np.stack(up_vectors, axis=0), axis=0)
    norm = np.linalg.norm(up)
    if norm < 1e-8:
        raise RuntimeError("Could not estimate camera up direction from poses")
    return up / norm


def fit_plane_svd(points: np.ndarray) -> tuple[np.ndarray, float]:
    center = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - center, full_matrices=False)
    normal = vh[-1]
    normal /= np.linalg.norm(normal)
    d = -float(normal @ center)
    return normal, d


def canonical_plane(normal: np.ndarray, d: float) -> tuple[np.ndarray, float]:
    normal = normal.copy()
    d = float(d)
    idx = int(np.argmax(np.abs(normal)))
    if normal[idx] < 0:
        normal *= -1
        d *= -1
    return normal, d


def plane_distance(points: np.ndarray, normal: np.ndarray, d: float) -> np.ndarray:
    return np.abs(points @ normal + d)


def orient_plane_to_reference(
    normal: np.ndarray,
    d: float,
    reference_normal: np.ndarray | None,
) -> tuple[np.ndarray, float]:
    if reference_normal is not None and float(normal @ reference_normal) < 0:
        return -normal, -float(d)
    return normal, float(d)


def angle_degrees(normal: np.ndarray, reference_normal: np.ndarray) -> float:
    dot = float(np.clip(normal @ reference_normal, -1.0, 1.0))
    return float(math.degrees(math.acos(dot)))


def camera_height_stats(
    centers: np.ndarray,
    normal: np.ndarray,
    d: float,
) -> tuple[float, float, float]:
    heights = centers @ normal + d
    above_ratio = float(np.mean(heights > 0.0))
    return float(np.min(heights)), float(np.median(heights)), above_ratio


def plane_is_duplicate(
    candidates: list[dict],
    normal: np.ndarray,
    d: float,
    angle_cos: float,
    offset_tol: float,
) -> int | None:
    for index, candidate in enumerate(candidates):
        cand_normal = np.array(candidate["normal"], dtype=np.float64)
        cand_d = float(candidate["d"])
        same = abs(float(cand_normal @ normal)) >= angle_cos
        if not same:
            continue
        offset = min(abs(cand_d - d), abs(cand_d + d))
        if offset <= offset_tol:
            return index
    return None


def ransac_planes(
    points: np.ndarray,
    *,
    camera_centers: np.ndarray,
    reference_normal: np.ndarray | None,
    max_normal_angle_deg: float,
    min_camera_above_ratio: float,
    iterations: int,
    threshold: float,
    max_candidates: int,
    min_inliers: int,
    angle_tol_deg: float,
    seed: int,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    candidates: list[dict] = []
    n_points = len(points)
    angle_cos = math.cos(math.radians(angle_tol_deg))
    offset_tol = max(threshold * 5.0, 1e-8)

    for _ in range(iterations):
        sample_idx = rng.choice(n_points, 3, replace=False)
        a, b, c = points[sample_idx]
        normal = np.cross(b - a, c - a)
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            continue
        normal /= norm
        d = -float(normal @ a)
        normal, d = orient_plane_to_reference(normal, d, reference_normal)
        if reference_normal is not None:
            angle_to_reference = angle_degrees(normal, reference_normal)
            if angle_to_reference > max_normal_angle_deg:
                continue
            min_camera_height, median_camera_height, camera_above_ratio = camera_height_stats(
                camera_centers, normal, d
            )
            if (
                min_camera_height <= 0.0
                or median_camera_height <= 0.0
                or camera_above_ratio < min_camera_above_ratio
            ):
                continue

        inliers = plane_distance(points, normal, d) <= threshold
        if int(inliers.sum()) < min_inliers:
            continue

        normal, d = fit_plane_svd(points[inliers])
        normal, d = orient_plane_to_reference(normal, d, reference_normal)
        if reference_normal is None:
            normal, d = canonical_plane(normal, d)
            angle_to_reference = None
            min_camera_height, median_camera_height, camera_above_ratio = camera_height_stats(
                camera_centers, normal, d
            )
        else:
            angle_to_reference = angle_degrees(normal, reference_normal)
            if angle_to_reference > max_normal_angle_deg:
                continue
            min_camera_height, median_camera_height, camera_above_ratio = camera_height_stats(
                camera_centers, normal, d
            )
            if (
                min_camera_height <= 0.0
                or median_camera_height <= 0.0
                or camera_above_ratio < min_camera_above_ratio
            ):
                continue
        distances = plane_distance(points, normal, d)
        inliers = distances <= threshold
        inlier_count = int(inliers.sum())
        if inlier_count < min_inliers:
            continue

        inlier_points = points[inliers]
        _, singular_values, _ = np.linalg.svd(
            inlier_points - inlier_points.mean(axis=0), full_matrices=False
        )
        spread = float(singular_values[0] * singular_values[1] / max(inlier_count, 1))
        score = float(inlier_count) + spread

        candidate = {
            "normal": normal.tolist(),
            "d": float(d),
            "inliers": inlier_count,
            "inlier_ratio": float(inlier_count / n_points),
            "rms_distance": float(np.sqrt(np.mean(distances[inliers] ** 2))),
            "angle_to_camera_up_deg": angle_to_reference,
            "min_camera_height": min_camera_height,
            "median_camera_height": median_camera_height,
            "camera_above_ratio": camera_above_ratio,
            "score": score,
        }

        duplicate_index = plane_is_duplicate(
            candidates, normal, d, angle_cos=angle_cos, offset_tol=offset_tol
        )
        if duplicate_index is None:
            candidates.append(candidate)
        elif score > candidates[duplicate_index]["score"]:
            candidates[duplicate_index] = candidate

    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates[:max_candidates]


def rotation_between_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = source / np.linalg.norm(source)
    target = target / np.linalg.norm(target)
    cross = np.cross(source, target)
    dot = float(np.clip(source @ target, -1.0, 1.0))
    if dot > 1.0 - 1e-12:
        return np.eye(3, dtype=np.float64)
    if dot < -1.0 + 1e-12:
        axis = np.array([1.0, 0.0, 0.0])
        if abs(source[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        axis = np.cross(source, axis)
        axis /= np.linalg.norm(axis)
        k = skew(axis)
        return np.eye(3) + 2.0 * (k @ k)
    k = skew(cross)
    return np.eye(3) + k + k @ k * ((1.0 - dot) / (np.linalg.norm(cross) ** 2))


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)


def pyplot():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def transform_for_plane(
    normal: np.ndarray,
    d: float,
    points: np.ndarray,
    centers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    normal = np.asarray(normal, dtype=np.float64)
    d = float(d)

    # Orient the plane so most camera centers have positive height.
    heights = centers @ normal + d
    if np.median(heights) < 0:
        normal = -normal
        d = -d

    rotation = rotation_between_vectors(normal, np.array([0.0, 0.0, 1.0]))
    rotated = points @ rotation.T
    translation = -np.median(rotated, axis=0)
    translation[2] = d

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform, normal, d


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def plot_candidate(
    output_path: Path,
    points: np.ndarray,
    centers: np.ndarray,
    transform: np.ndarray,
    normal: np.ndarray,
    d: float,
    threshold: float,
    title: str,
    max_plot_points: int,
    seed: int,
) -> None:
    plt = pyplot()
    rng = np.random.default_rng(seed)
    transformed = apply_transform(points, transform)
    transformed_centers = apply_transform(centers, transform)
    distances = plane_distance(points, normal, d)
    inliers = distances <= threshold

    if len(transformed) > max_plot_points:
        idx = rng.choice(len(transformed), max_plot_points, replace=False)
    else:
        idx = np.arange(len(transformed))

    sample = transformed[idx]
    sample_inliers = inliers[idx]
    yz = sample[:, [1, 2]]
    low = np.quantile(yz, 0.01, axis=0)
    high = np.quantile(yz, 0.99, axis=0)
    span = np.maximum(high - low, 1e-6)
    low -= span * 0.08
    high += span * 0.08
    low[1] = min(low[1], -threshold * 4.0)
    high[1] = max(high[1], threshold * 4.0)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.scatter(
        sample[~sample_inliers, 1],
        sample[~sample_inliers, 2],
        s=0.8,
        c="#9ca3af",
        alpha=0.28,
        linewidths=0,
        label="other points",
    )
    ax.scatter(
        sample[sample_inliers, 1],
        sample[sample_inliers, 2],
        s=1.2,
        c="#f97316",
        alpha=0.7,
        linewidths=0,
        label="plane inliers",
    )
    ax.scatter(
        transformed_centers[:, 1],
        transformed_centers[:, 2],
        s=12,
        c="#16a34a",
        marker="x",
        linewidths=0.8,
        label="cameras",
    )
    ax.axhline(0.0, color="#111827", linewidth=1.0)
    ax.set_xlim(float(low[0]), float(high[0]))
    ax.set_ylim(float(low[1]), float(high[1]))
    ax.set_xlabel("y after transform")
    ax.set_ylabel("z after transform")
    ax.set_title(title)
    ax.grid(True, color="#e5e7eb", linewidth=0.6)
    ax.legend(loc="upper right", markerscale=4, fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def make_overview(candidate_paths: list[Path], output_path: Path) -> None:
    plt = pyplot()
    images = [plt.imread(path) for path in candidate_paths]
    cols = min(2, len(images))
    rows = math.ceil(len(images) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 5 * rows), dpi=120)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.reshape(rows, cols)
    for axis in axes.ravel():
        axis.axis("off")
    for image, axis in zip(images, axes.ravel()):
        axis.imshow(image)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def command_propose(args: argparse.Namespace) -> None:
    scene_path = args.scene_path.resolve()
    sparse_path = sparse_path_for_scene(scene_path, args.sparse_dir)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _, images, points3d = read_model(sparse_path)
    _, points = points_array(points3d)
    centers = camera_centers(images)
    reference_normal = camera_up_normal(images)

    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if len(points) < 3:
        raise RuntimeError("Need at least three finite COLMAP points")

    extent = np.linalg.norm(np.quantile(points, 0.98, axis=0) - np.quantile(points, 0.02, axis=0))
    threshold = args.threshold
    if threshold is None:
        threshold = max(extent * args.threshold_ratio, 1e-6)

    rng = np.random.default_rng(args.seed)
    if len(points) > args.max_ransac_points:
        sample_idx = rng.choice(len(points), args.max_ransac_points, replace=False)
        ransac_points = points[sample_idx]
    else:
        ransac_points = points

    min_inliers = args.min_inliers
    if min_inliers is None:
        min_inliers = max(50, int(len(ransac_points) * args.min_inlier_ratio))

    candidates = ransac_planes(
        ransac_points,
        camera_centers=centers,
        reference_normal=None if args.disable_camera_prior else reference_normal,
        max_normal_angle_deg=args.max_normal_angle_deg,
        min_camera_above_ratio=args.min_camera_above_ratio,
        iterations=args.iterations,
        threshold=threshold,
        max_candidates=args.num_candidates,
        min_inliers=min_inliers,
        angle_tol_deg=args.angle_tol_deg,
        seed=args.seed,
    )
    if not candidates:
        raise RuntimeError(
            f"RANSAC found no planes. Try increasing --threshold above {threshold:.6g}."
        )

    image_paths = []
    proposal_items = []
    for candidate_id, candidate in enumerate(candidates):
        transform, oriented_normal, oriented_d = transform_for_plane(
            np.array(candidate["normal"], dtype=np.float64),
            float(candidate["d"]),
            points,
            centers,
        )
        image_name = f"candidate_{candidate_id:02d}_zy.png"
        image_path = output_dir / image_name
        angle_text = ""
        if candidate.get("angle_to_camera_up_deg") is not None:
            angle_text = f" angle={candidate['angle_to_camera_up_deg']:.1f}deg"
        title = (
            f"candidate {candidate_id}: inliers={candidate['inliers']} "
            f"ratio={candidate['inlier_ratio']:.3f} rms={candidate['rms_distance']:.4g}"
            f"{angle_text} h={candidate['median_camera_height']:.3g}"
        )
        plot_candidate(
            image_path,
            points,
            centers,
            transform,
            oriented_normal,
            oriented_d,
            threshold,
            title,
            args.max_plot_points,
            args.seed + candidate_id,
        )
        image_paths.append(image_path)
        proposal_items.append(
            {
                "id": candidate_id,
                "image": image_name,
                "inliers": candidate["inliers"],
                "inlier_ratio": candidate["inlier_ratio"],
                "rms_distance": candidate["rms_distance"],
                "angle_to_camera_up_deg": candidate.get("angle_to_camera_up_deg"),
                "min_camera_height": candidate.get("min_camera_height"),
                "median_camera_height": candidate.get("median_camera_height"),
                "camera_above_ratio": candidate.get("camera_above_ratio"),
                "normal": oriented_normal.tolist(),
                "d": float(oriented_d),
                "transform": transform.tolist(),
            }
        )

    overview_path = output_dir / "overview_zy.png"
    make_overview(image_paths, overview_path)

    proposals = {
        "scene_path": str(scene_path),
        "sparse_path": str(sparse_path),
        "point_count": int(len(points)),
        "camera_count": int(len(images)),
        "threshold": float(threshold),
        "threshold_ratio": None if args.threshold is not None else args.threshold_ratio,
        "camera_up_normal": reference_normal.tolist(),
        "camera_prior_enabled": not args.disable_camera_prior,
        "max_normal_angle_deg": args.max_normal_angle_deg,
        "min_camera_above_ratio": args.min_camera_above_ratio,
        "candidates": proposal_items,
    }
    proposals_path = output_dir / "ground_proposals.json"
    proposals_path.write_text(json.dumps(proposals, indent=2), encoding="utf-8")

    print(f"Wrote {len(proposal_items)} candidates to {proposals_path}")
    print(f"Overview: {overview_path}")
    for item in proposal_items:
        print(
            f"  candidate {item['id']}: inliers={item['inliers']} "
            f"ratio={item['inlier_ratio']:.3f} "
            f"angle={item['angle_to_camera_up_deg'] if item['angle_to_camera_up_deg'] is not None else 'n/a'} "
            f"camera_above={item['camera_above_ratio']:.2f} image={output_dir / item['image']}"
        )


def transformed_images(images: dict[int, ColmapImage], transform: np.ndarray) -> dict[int, ColmapImage]:
    out = {}
    for image_id, img in images.items():
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = qvec2rotmat(img.qvec)
        w2c[:3, 3] = img.tvec
        c2w = np.linalg.inv(w2c)
        new_c2w = transform @ c2w
        new_w2c = np.linalg.inv(new_c2w)
        out[image_id] = ColmapImage(
            id=img.id,
            qvec=rotmat2qvec(new_w2c[:3, :3]),
            tvec=new_w2c[:3, 3].astype(np.float64),
            camera_id=img.camera_id,
            name=img.name,
            xys=img.xys,
            point3D_ids=img.point3D_ids,
        )
    return out


def transformed_points(points3d: dict[int, Point3D], transform: np.ndarray) -> dict[int, Point3D]:
    out = {}
    for point_id, point in points3d.items():
        xyz = transform[:3, :3] @ point.xyz + transform[:3, 3]
        out[point_id] = Point3D(
            id=point.id,
            xyz=xyz.astype(np.float64),
            rgb=point.rgb,
            error=point.error,
            image_ids=point.image_ids,
            point2D_idxs=point.point2D_idxs,
        )
    return out


def copy_images(src_scene: Path, dst_scene: Path, images_dir: str, mode: str) -> None:
    src = src_scene / images_dir
    dst = dst_scene / images_dir
    if not src.exists():
        return
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    if mode == "copy":
        shutil.copytree(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src.resolve(), target_is_directory=True)
    elif mode == "none":
        return
    else:
        raise ValueError(f"Unknown image copy mode: {mode}")


def command_apply(args: argparse.Namespace) -> None:
    proposals = json.loads(args.proposals.read_text(encoding="utf-8"))
    candidate = next(
        (item for item in proposals["candidates"] if int(item["id"]) == args.candidate_id),
        None,
    )
    if candidate is None:
        raise ValueError(f"Candidate {args.candidate_id} not found in {args.proposals}")

    scene_path = args.scene_path.resolve() if args.scene_path else Path(proposals["scene_path"]).resolve()
    sparse_path = sparse_path_for_scene(scene_path, args.sparse_dir)
    output_scene = args.output_scene_path.resolve()
    output_sparse_dir = args.output_sparse_dir
    output_sparse_path = output_scene / output_sparse_dir

    if output_scene.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_scene} exists. Use --overwrite to replace it.")
        shutil.rmtree(output_scene)
    output_sparse_path.mkdir(parents=True, exist_ok=True)

    cameras, images, points3d = read_model(sparse_path)
    transform = np.array(candidate["transform"], dtype=np.float64)
    new_images = transformed_images(images, transform)
    new_points = transformed_points(points3d, transform)

    write_cameras_binary(cameras, output_sparse_path / "cameras.bin")
    write_images_binary(new_images, output_sparse_path / "images.bin")
    write_points3d_binary(new_points, output_sparse_path / "points3D.bin")

    for name in ["rigs.bin", "frames.bin", "project.ini"]:
        source = sparse_path / name
        if source.exists():
            shutil.copy2(source, output_sparse_path / name)

    copy_images(scene_path, output_scene, args.images_dir, args.image_mode)
    metadata = {
        "source_scene": str(scene_path),
        "source_sparse": str(sparse_path),
        "candidate_id": args.candidate_id,
        "candidate": candidate,
        "output_sparse_dir": output_sparse_dir,
        "images_dir": args.images_dir,
    }
    (output_scene / "ground_alignment.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Wrote aligned COLMAP scene: {output_scene}")
    print(f"Sparse model: {output_sparse_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RANSAC ground candidates and COLMAP world alignment"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    propose = subparsers.add_parser("propose", help="Find candidate ground planes")
    propose.add_argument("--scene_path", type=Path, required=True)
    propose.add_argument("--sparse_dir", default=None)
    propose.add_argument("--output_dir", type=Path, required=True)
    propose.add_argument("--num_candidates", type=int, default=8)
    propose.add_argument("--iterations", type=int, default=12000)
    propose.add_argument("--threshold", type=float, default=None)
    propose.add_argument("--threshold_ratio", type=float, default=0.008)
    propose.add_argument("--min_inliers", type=int, default=None)
    propose.add_argument("--min_inlier_ratio", type=float, default=0.02)
    propose.add_argument("--max_ransac_points", type=int, default=50000)
    propose.add_argument("--max_plot_points", type=int, default=160000)
    propose.add_argument("--angle_tol_deg", type=float, default=10.0)
    propose.add_argument(
        "--max_normal_angle_deg",
        type=float,
        default=10.0,
        help="Reject candidate planes whose normal differs from camera-estimated up by more than this angle.",
    )
    propose.add_argument(
        "--min_camera_above_ratio",
        type=float,
        default=1.0,
        help="Reject candidate planes unless this fraction of camera centers is above the plane.",
    )
    propose.add_argument(
        "--disable_camera_prior",
        action="store_true",
        help="Disable camera-pose angle and below-camera filtering.",
    )
    propose.add_argument("--seed", type=int, default=7)
    propose.set_defaults(func=command_propose)

    apply = subparsers.add_parser("apply", help="Apply one candidate to a new scene")
    apply.add_argument("--proposals", type=Path, required=True)
    apply.add_argument("--candidate_id", type=int, required=True)
    apply.add_argument("--scene_path", type=Path, default=None)
    apply.add_argument("--sparse_dir", default=None)
    apply.add_argument("--output_scene_path", type=Path, required=True)
    apply.add_argument("--output_sparse_dir", default="sparse")
    apply.add_argument("--images_dir", default="images")
    apply.add_argument("--image_mode", choices=["symlink", "copy", "none"], default="symlink")
    apply.add_argument("--overwrite", action="store_true")
    apply.set_defaults(func=command_apply)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

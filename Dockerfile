# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.10
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG TORCH_VERSION=2.7.0+cu128
ARG TORCHVISION_VERSION=0.22.0
ARG TORCH_CUDA_ARCH_LIST=9.0
ARG COLMAP_CUDA_ARCHITECTURES=90
ARG MAX_JOBS=8

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUDA_HOME=/usr/local/cuda \
    FORCE_CUDA=1 \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    MAX_JOBS=${MAX_JOBS} \
    PYTHONPATH=/workspace/resplat \
    HF_HOME=/workspace/.cache/huggingface \
    TORCH_HOME=/workspace/.cache/torch \
    MPLCONFIGDIR=/workspace/.cache/matplotlib \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        ffmpeg \
        gcc-10 \
        g++-10 \
        git \
        libegl1 \
        libboost-graph-dev \
        libboost-program-options-dev \
        libboost-system-dev \
        libceres-dev \
        libcurl4-openssl-dev \
        libeigen3-dev \
        libglew-dev \
        libglib2.0-0 \
        libgl1 \
        libglvnd0 \
        libgoogle-glog-dev \
        libgmock-dev \
        libgtest-dev \
        libmetis-dev \
        libopenblas-openmp-dev \
        libopenexr-dev \
        libopenimageio-dev \
        libsm6 \
        libsqlite3-dev \
        libsuitesparse-dev \
        libssl-dev \
        libxext6 \
        libxrender1 \
        ninja-build \
        openimageio-tools \
        pkg-config \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-venv \
        python3-pip \
        wget \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/local/bin/python \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/local/bin/python3 \
    && python -m pip install --upgrade pip setuptools wheel cmake \
    && mkdir -p /usr/include/opencv4 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/resplat

# Build COLMAP from source with CUDA support. Ubuntu repository COLMAP builds
# are CPU-only; ReSplat only needs the CLI/SfM path, so GUI/MVS/ONNX are disabled
# to keep the build smaller while preserving CUDA SIFT extraction/matching.
RUN git clone --depth 1 https://github.com/colmap/colmap.git /tmp/colmap \
    && cmake -S /tmp/colmap -B /tmp/colmap/build -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DCMAKE_CUDA_ARCHITECTURES=${COLMAP_CUDA_ARCHITECTURES} \
        -DBLA_VENDOR=OpenBLAS \
        -DCUDA_ENABLED=ON \
        -DGUI_ENABLED=OFF \
        -DMVS_ENABLED=OFF \
        -DONNX_ENABLED=OFF \
        -DCGAL_ENABLED=OFF \
        -DTESTS_ENABLED=OFF \
        -DBENCHMARK_ENABLED=OFF \
        -DCMAKE_C_COMPILER=/usr/bin/gcc-10 \
        -DCMAKE_CXX_COMPILER=/usr/bin/g++-10 \
        -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-10 \
    && cmake --build /tmp/colmap/build --target colmap --parallel ${MAX_JOBS} \
    && install -m 0755 /tmp/colmap/build/src/colmap/exe/colmap /usr/local/bin/colmap \
    && colmap -h | grep -i "with CUDA" \
    && rm -rf /tmp/colmap

# Install CUDA-enabled PyTorch wheels explicitly. On GH200/aarch64 this avoids
# accidental CPU-only installs or source builds.
RUN python -m pip install \
        torch==${TORCH_VERSION} \
        torchvision==${TORCHVISION_VERSION} \
        --index-url ${TORCH_INDEX_URL}

COPY requirements.txt ./requirements.txt
RUN python -m pip install -r requirements.txt

# ReSplat uses gsplat as an external CUDA dependency, so keep that dependency in
# the environment image. Local project code is bind-mounted at runtime.
RUN python -m pip install --no-build-isolation \
        "git+https://github.com/nerfstudio-project/gsplat.git@v1.5.3"

CMD ["bash", "-lc", "if [ -n \"${RESPLAT_CMD:-}\" ]; then exec bash -lc \"${RESPLAT_CMD}\"; else exec sleep infinity; fi"]

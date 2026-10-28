# syntax=docker/dockerfile:1
# Environment only. Mount project source and all model/data assets at runtime.
FROM nvidia/cuda:12.3.1-devel-ubuntu20.04@sha256:befbdfddbb52727f9ce8d0c574cac0f631c606b1e6f0e523f3a0777fe2720c99

ARG DEBIAN_FRONTEND=noninteractive
ARG MAX_JOBS=2
ARG CUDA_ARCHITECTURES="7.5;8.0;8.6;8.9;9.0+PTX"
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV TZ=Etc/UTC \
    PATH=/opt/conda/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYOPENGL_PLATFORM=egl \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    MPLCONFIGDIR=/tmp/matplotlib

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git build-essential cmake pkg-config ninja-build \
      libboost-dev libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev \
      libglib2.0-0 libsm6 libxext6 libxrender1 libsndfile1 libportaudio2 \
      ffmpeg unzip \
    && rm -rf /var/lib/apt/lists/*

COPY environment.yaml bootstrap.sh /opt/thunder-environment/
RUN bash /opt/thunder-environment/bootstrap.sh

# Separate cached stages keep dependency/source-build retries inexpensive.
COPY requirements.txt constraints.txt install.sh /opt/thunder-environment/
RUN bash /opt/thunder-environment/install.sh runtime
COPY requirements-sources.txt /opt/thunder-environment/
RUN bash /opt/thunder-environment/install.sh source chumpy
RUN bash /opt/thunder-environment/install.sh source psbody-mesh
RUN bash /opt/thunder-environment/install.sh source scikit-video
RUN bash /opt/thunder-environment/install.sh source pytorchvideo
RUN bash /opt/thunder-environment/install.sh source pytorch3d
RUN bash /opt/thunder-environment/install.sh source kaldifeat
# Runtime graphics libraries are needed even when rendering through EGL.
RUN apt-get update && apt-get install -y --no-install-recommends libglu1-mesa \
    && rm -rf /var/lib/apt/lists/*
RUN bash /opt/thunder-environment/install.sh verify

# No host account name, password, SSH client configuration or sudo access.
# A runtime launcher can override UID:GID to match output ownership on the host.
RUN groupadd --gid 1000 thunder \
    && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash thunder \
    && mkdir -p /workspace /cache /output \
    && chown thunder:thunder /workspace /cache /output \
    && find /opt/conda /opt/thunder-environment -type d ! -perm -005 -exec chmod a+rx {} + \
    && find /opt/conda /opt/thunder-environment -type f ! -perm -004 -exec chmod a+r {} +
WORKDIR /workspace
USER thunder
CMD ["/bin/bash"]

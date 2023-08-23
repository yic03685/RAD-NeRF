FROM runpod/pytorch:3.10-2.0.1-118-devel

# Make python 3.10 as default and create an environment
RUN : \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        wget \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ffmpeg \    
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        portaudio19-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && :

# install conda because of pytorch3d
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && /bin/bash ~/miniconda.sh -b -p /opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

WORKDIR /workspace

# dependency - pytorch3d
COPY ../../environment.yml .
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "pytorch3d", "/bin/bash", "-c"]

# RAD-Nerf
RUN git clone https://github.com/yic03685/RAD-NeRF.git

WORKDIR /workspace/RAD-NeRF

ENV CUDA_HOME=/usr/local/cuda-11.8

RUN pip install -r requirements.txt
RUN pip install -U scikit-learn scipy matplotlib
RUN bash scripts/install_ext.sh

# Copy required models
COPY ../../models/01_MorphableModel.mat /workspace/RAD-NeRF/data_utils/face_tracking/3DMM/
COPY ../../models/exp_info.npy /workspace/RAD-NeRF/data_utils/face_tracking/3DMM/
COPY ../../models/keys_info.npy /workspace/RAD-NeRF/data_utils/face_tracking/3DMM/
COPY ../../models/sub_mesh.npy /workspace/RAD-NeRF/data_utils/face_tracking/3DMM/
COPY ../../models/topology_info.npy /workspace/RAD-NeRF/data_utils/face_tracking/3DMM/
COPY ../../models/79999_iter.pth /workspace/RAD-NeRF/data_utils/face_parsing/

WORKDIR /workspace/RAD-NeRF/data_utils/face_tracking
RUN python convert_BFM.py

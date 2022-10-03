# Base image
FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

# Setup basic packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    vim \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    libglfw3-dev \
    libglm-dev \
    libx11-dev \
    libomp-dev \
    libegl1-mesa-dev \
    pkg-config \
    wget \
    zip \
    unzip &&\
    rm -rf /var/lib/apt/lists/*

# Install conda
RUN curl -L -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  &&\
    chmod +x ~/miniconda.sh &&\
    ~/miniconda.sh -b -p /opt/conda &&\
    rm ~/miniconda.sh &&\
    /opt/conda/bin/conda install numpy pyyaml scipy ipython mkl mkl-include &&\
    /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH

# Install cmake
# RUN wget https://github.com/Kitware/CMake/releases/download/v3.14.0/cmake-3.14.0-Linux-x86_64.sh
# RUN mkdir /opt/cmake
# RUN sh /cmake-3.14.0-Linux-x86_64.sh --prefix=/opt/cmake --skip-license
# RUN ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake
# RUN cmake --version

# Conda environment
# RUN conda env create -n sam-exp --file=sam-exp.yml
# RUN conda activate sam-exp

# Make RUN commands use the new environment:
# SHELL ["conda", "run", "-n", "sam-exp", "/bin/bash", "-c"]

# Install torch and cuda at correct version
# RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
# BAD, cpu only
# RUN conda install -y pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia

# Install sam repo
# RUN git clone --branch stable https://github.com/z
# RUN /bin/bash -c ". activate habitat; cd habitat-lab; python setup.py install"

# Setup habitat-sim
# RUN conda install habitat-sim withbullet headless -c conda-forge -c aihabitat

# Setup conda environment (need to manually activate in interactive mode)
# ENV PATH /opt/conda/envs/habitat/sam-exp/bin:$PATH
# RUN conda init bash && \
#     . /root/.bashrc 

# WORKDIR /sam-exploration

# Silence habitat-sim logs
# ENV GLOG_minloglevel=2
# ENV MAGNUM_LOG="quiet"

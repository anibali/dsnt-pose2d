FROM nvidia/cuda:8.0-cudnn5-devel

# Install curl and sudo
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
 && rm -rf /var/lib/apt/lists/*

# Use Tini as the init process with PID 1
RUN curl -Lso /tini https://github.com/krallin/tini/releases/download/v0.14.0/tini \
 && chmod +x /tini
ENTRYPOINT ["/tini", "--"]

# Create a working directory
RUN mkdir /app
WORKDIR /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# Install Git and X11 client
RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    git \
    libx11-6 \
 && sudo rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN curl -so ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-4.3.14-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh

# Create a Python 3.6 environment
RUN /home/user/miniconda/bin/conda install conda-build \
 && /home/user/miniconda/bin/conda create -y --name py36 python=3.6.3 \
 && /home/user/miniconda/bin/conda clean -ya
ENV PATH=/home/user/miniconda/envs/py36/bin:$PATH \
    CONDA_DEFAULT_ENV=py36 \
    CONDA_PREFIX=/home/user/miniconda/envs/py36

# Install some dependencies from conda
RUN conda install -y --name py36 -c soumith \
    pytorch=0.2.0 \
    torchvision=0.1.9 \
    graphviz=2.38.0 \
    cuda80=1.0 \
 && conda clean -ya

# Install other dependencies from pip
COPY requirements.txt .
RUN pip install -r requirements.txt

# Replace Pillow with Pillow-SIMD (a faster implementation)
RUN pip uninstall -y pillow \
 && pip install pillow-simd==4.2.1.post0

# Set the default command to python3
CMD ["python3"]

FROM nvidia/cuda:10.2-base-ubuntu18.04

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /app
WORKDIR /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
&& chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# Install Miniconda and Python 3.8
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/home/user/miniconda/bin:$PATH
RUN curl -sLo ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-py38_4.8.2-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh \
 && conda install -y python==3.8.1 \
 && conda clean -ya


RUN conda create -n mne_pyt python=3.8

RUN pip install scipy

RUN pip install numpy

RUN pip install scikit-learn

RUN pip install pandas

RUN pip install jupyter

RUN pip install h5py

RUN pip install graphviz

RUN pip install pydot

RUN pip install keras

RUN pip install matplotlib

RUN pip install seaborn

RUN pip install joblib

RUN pip install -U mne

RUN conda install pytorch==1.7.0 torchvision==0.8.0 cudatoolkit=10.2 -c pytorch

RUN pip install torch-scatter==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.7.0.html
RUN pip install torch-sparse==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.7.0.html
RUN pip install torch-cluster==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.7.0.html
RUN pip install torch-spline-conv==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.7.0.html
RUN pip install torch-geometric
RUN pip install captum
RUN pip install tensorboard
ENV NUMBA_CACHE_DIR=/tmp/numba_cache
RUN pip install glob2

RUN pip install dash
RUN pip install dash-uploader
RUN pip install visdcc
RUN pip install dash_daq
RUN pip install dash_table
RUN pip install dash_cytoscape
RUN pip install dash-bootstrap-components
RUN pip install dash-extensions

RUN pip install flake8
RUN pip install fasteners
RUN pip install psutil
RUN pip install toml
RUN pip install webdataset

# Create some command aliases and turn off jupyter notebook tokens
RUN echo 'alias jp="jupyter notebook --no-browser --ip=0.0.0.0 --allow-root"' >> /home/user/.bashrc
# RUN echo 'alias jl="jupyter lab --no-browser --ip=0.0.0.0 --allow-root"' >> /home/user/.bashrc
RUN mkdir -p /home/user/.jupyter && echo "c.NotebookApp.token = u'yoga_projects'" >> /home/user/.jupyter/jupyter_notebook_config.py && echo "c.NotebookApp.notebook_dir = '/home/varatha2/projects/john_wei'" >> /home/user/.jupyter/jupyter_notebook_config.py
ENV MPLCONFIGDIR=/tmp/matplotlib_cache

RUN pip install lmdb
RUN pip install msgpack

RUN pip install comet_ml --upgrade

RUN pip install braindecode
RUN pip install lightgbm
RUN pip install antropy
RUN pip install fooof




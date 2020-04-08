#!/bin/bash

# Install NVIDIA Docker support
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Check GPU
lspci | grep -i nvidia
docker run --gpus all --rm nvidia/cuda nvidia-smi

# Docker run with model
SAVED_PATH="$( realpath ../data/saved )"
docker run --gpus all --name spesep -p 8888:8888 -v ${SAVED_PATH}:/usr/dev/speech-separation/data/saved --rm lapter57/speech-separation

FROM continuumio/anaconda3

RUN mkdir -p /usr/dev/speech-separation
WORKDIR /usr/dev/speech-separation

COPY . /usr/dev/speech-separation

USER root

RUN pip3 install --prefix /usr/local/bin --upgrade youtube-dl
RUN apt-get update && 
    apt-get install -y \
       sox \
       snapd && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN snap install ffmpeg

ARG CONDA_ENV=spesep
RUN conda create -y -n ${CONDA_ENV} python=3.5 && \
    conda activate ${CONDA_ENV} && \
    pip3 install ipykernel && \
    conda install -y -c conda-forge jupyterlab && \
    conda install -y tensorflow-gpu && \
    pip3 install --no-cache-dir -r requirements.txt

EXPOSE 8888

RUN jupyter lab

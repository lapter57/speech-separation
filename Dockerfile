FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

RUN mkdir -p /usr/dev/speech-separation
WORKDIR /usr/dev/speech-separation

COPY . /usr/dev/speech-separation

SHELL ["/bin/bash", "-c"]

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    apt-get update && \
    add-apt-repository ppa:jonathonf/ffmpeg-4 && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
    locales \
    vim \
    python3.7 \
    python3-pip python3-dev  \
    libsm6 libxext6 libxrender-dev \
    sox ffmpeg && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip
RUN pip3 install --prefix /usr/local --upgrade youtube-dl
RUN python3 -m pip install ipykernel
RUN python3 -m ipykernel install --user

RUN pip3 install --no-cache-dir -r requirements.txt

RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    locale-gen
ENV LANG en_US.UTF-8  
ENV LANGUAGE en_US:en  
ENV LC_ALL en_US.UTF-8 

EXPOSE 8085

ENTRYPOINT ["/bin/bash"]

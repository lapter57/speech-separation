FROM tensorflow/tensorflow:latest-gpu

RUN mkdir -p /usr/dev/speech-separation
WORKDIR /usr/dev/speech-separation

COPY . /usr/dev/speech-separation

SHELL ["/bin/bash", "-c"]

RUN apt-get install -y software-properties-common && \
    apt-get update && \
    add-apt-repository ppa:jonathonf/ffmpeg-4 && \
    apt-get install -y \
    python3-pip python3-dev  \
    libsm6 libxext6 libxrender-dev \
    sox ffmpeg && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip
RUN pip3 install --prefix /usr/local --upgrade youtube-dl
RUN python3 -m pip install ipykernel
RUN python3 -m ipykernel install --user

RUN pip3 install --no-cache-dir -r requirements.txt

EXPOSE 8888

CMD ["jupyter", "lab", "--ip='0.0.0.0'", "--port=8888", "--no-browser", "--allow-root"]

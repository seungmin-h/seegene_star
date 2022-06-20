FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel
WORKDIR /workspace
RUN apt update && apt install -y python3-pip libopenslide-dev openslide-tools
RUN pip install -U pip
COPY requirements.txt .
RUN pip install -r requirements.txt

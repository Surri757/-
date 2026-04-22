# 2026清华大学大数据挑战赛 Dockerfile
# Python 3.10.14 + PyTorch 2.2.1 + CUDA 12.1

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python

RUN pip install --no-cache-dir numpy==1.26.4 pandas==2.2.2 scikit-learn==1.5.0
RUN pip install --no-cache-dir lightgbm==4.3.0 xgboost==2.0.3
RUN pip install --no-cache-dir torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir catboost==1.2.5 ngboost==0.4.1 scipy==1.13.1 statsmodels==0.14.2

WORKDIR /app

COPY app/code /app/code
COPY app/data /app/data
COPY app/readme.md /app/readme.md

RUN chmod +x /app/code/*.sh

ENV PYTHONPATH=/app/code:$PYTHONPATH

CMD ["bash", "-c", "bash /app/code/init.sh && bash /app/code/train.sh"]
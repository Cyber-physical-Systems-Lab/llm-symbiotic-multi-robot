FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

ENV PYTHONPATH=/app


RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    cmake \
    ninja-build \
    pkg-config \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 先装构建工具 + 固定 numpy + cython（避免隔离环境重复拉依赖）
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip setuptools wheel \
    && pip install "numpy==1.26.4" "Cython<3" "pybind11<3"

# 关键：先单独装 pyastar2d（在主环境里编译/安装）
RUN pip install -v --no-build-isolation pyastar2d

# 再装你的 requirements（如果里面也写了 numpy，建议删掉那行，避免版本冲突）
RUN pip install -r /app/requirements.txt

# 拷贝代码
COPY . /app

# 关键：安装 TA-RWARE 时禁用 build isolation（避免它又去隔离环境装 numpy）
RUN pip install -v --no-build-isolation -e /app/task-assignment-robotic-warehouse

RUN mkdir -p /app/outputs
ENTRYPOINT ["python", "-m", "scripts.run_experiment"]

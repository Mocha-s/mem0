FROM python:3.12-slim

LABEL org.opencontainers.image.name="mem0/openmemory-mcp"

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/openmemory

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY config.json .
COPY . .

EXPOSE 8765
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8765"]

FROM python:3.11.5

WORKDIR /app

COPY requirements.txt ./
COPY app.py ./

# OpenGL 関連のライブラリをインストール (Debian/Ubuntu の場合)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]
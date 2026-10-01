#!/bin/sh
set -eu

cd /root/spam-filter
docker build -t spam-filter:latest .
docker rm -f spam-filter >/dev/null 2>&1 || true
docker run -d \
    --name spam-filter \
    --restart unless-stopped \
    -p 8787:8787 \
    -v spam-filter-data:/app/backend/data \
    spam-filter:latest
sleep 5
curl -fsS http://127.0.0.1:8787/health
echo
docker exec spam-filter ls -lh /app/backend/models/spam_pipeline_pinyin.joblib
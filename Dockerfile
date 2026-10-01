FROM node:24-alpine AS frontend
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY index.html tsconfig*.json vite.config.ts eslint.config.js ./
COPY public ./public
COPY src ./src
RUN npm run build

FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1 \
	PYTHONPATH=/app/backend \
	SPAM_MODEL_PATH=/app/backend/models/spam_pipeline_pinyin.joblib
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir \
	--index-url https://mirrors.aliyun.com/pypi/simple/ \
	--timeout 180 \
	--retries 10 \
	-r backend/requirements.txt
COPY backend ./backend
COPY backend/models/spam_pipeline_pinyin.joblib ./backend/models/spam_pipeline_pinyin.joblib
RUN test -s "$SPAM_MODEL_PATH"
COPY --from=frontend /app/dist ./dist
EXPOSE 8787
CMD ["uvicorn", "app.main:app", "--app-dir", "backend", "--host", "0.0.0.0", "--port", "8787"]
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import classifier as classifier_module
from .classifier import SpamClassifier
from .pipeline import ConfigError, apply_config, load_pipeline, save_config
from .storage import ResultStore


BASE_DIR = Path(__file__).resolve().parents[1]
PIPELINE = load_pipeline()
classifier = SpamClassifier(
    Path(os.getenv("SPAM_MODEL_PATH", BASE_DIR / "models" / "spam_pipeline_pinyin.joblib")),
    Path(os.getenv("SPAM_ABUSE_MODEL_PATH", BASE_DIR / "models" / "abuse_pipeline_pinyin.joblib")),
)
store = ResultStore(Path(os.getenv("SPAM_DATABASE_PATH", BASE_DIR / "data" / "spam_filter.db")))

app = FastAPI(title="中文垃圾文本过滤 API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class FilterRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10_000)


class BatchFilterRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=100)


class FeedbackRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10_000)
    predicted_label: Literal["spam", "normal", "review"]
    corrected_label: Literal["spam", "normal"]
    corrected_abuse_label: Literal["abuse", "normal"] | None = None
    note: str = Field(default="", max_length=500)


def filter_one(text: str) -> dict[str, object]:
    result = classifier.predict(text)
    store.record_detection(
        text,
        str(result["label"]),
        float(result["score"]),
        str(result["abuse_label"]),
        float(result["abuse_score"]),
    )
    return {"text": text, **result}


@app.get("/health")
def health() -> dict[str, str | bool]:
    return {
        "status": "ok",
        "model_loaded": classifier.model is not None,
        "abuse_model_loaded": classifier.abuse_model is not None,
        "version": "1.1.0",
    }


@app.post("/api/filter")
def filter_text(request: FilterRequest) -> dict[str, object]:
    return filter_one(request.text)


@app.post("/api/filter/batch")
def filter_batch(request: BatchFilterRequest) -> dict[str, object]:
    if any(not text.strip() for text in request.texts):
        raise HTTPException(status_code=422, detail="批量文本不能为空")
    return {"results": [filter_one(text) for text in request.texts]}


@app.post("/api/feedback", status_code=201)
def submit_feedback(request: FeedbackRequest) -> dict[str, int | str]:
    feedback_id = store.record_feedback(
        request.text,
        request.predicted_label,
        request.corrected_label,
        request.note,
        request.corrected_abuse_label or "normal",
    )
    return {"id": feedback_id, "status": "accepted"}


@app.get("/api/pipeline")
def get_pipeline() -> dict[str, object]:
    """返回检测流水线的完整配置：模型、特征权重、规则权重、融合系数与阈值。"""
    return PIPELINE.describe()


@app.put("/api/pipeline")
def update_pipeline(payload: dict[str, object]) -> dict[str, object]:
    """在线调参：校验后热加载并落盘，规则权重与阈值立即生效。

    特征权重会写入配置但需重新训练模型才真正生效，响应中单独标记。
    """
    global PIPELINE
    try:
        updated = apply_config(PIPELINE, payload)
    except ConfigError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    classifier_module.apply_pipeline(updated)
    PIPELINE = updated
    save_config(updated)
    return {
        "status": "applied",
        "requires_retrain": ["models.*.features.*"],
        "pipeline": updated.describe(),
    }


@app.get("/api/stats")
def get_stats() -> dict[str, int | float]:
    return store.stats()


FRONTEND_DIR = BASE_DIR.parent / "dist"
if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def frontend(path: str) -> FileResponse:
        requested_file = (FRONTEND_DIR / path).resolve()
        if requested_file.is_relative_to(FRONTEND_DIR.resolve()) and requested_file.is_file():
            return FileResponse(requested_file)
        return FileResponse(FRONTEND_DIR / "index.html")
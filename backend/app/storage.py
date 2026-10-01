from __future__ import annotations

import hashlib
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path


class ResultStore:
    def __init__(self, database_path: Path) -> None:
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self.database_path = database_path
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection, connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text_hash TEXT NOT NULL,
                    text_length INTEGER NOT NULL,
                    label TEXT NOT NULL,
                    score REAL NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    predicted_label TEXT NOT NULL,
                    corrected_label TEXT NOT NULL,
                    note TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );
                """
            )
            self._add_missing_columns(
                connection,
                "detections",
                {
                    "abuse_label": "TEXT NOT NULL DEFAULT 'normal'",
                    "abuse_score": "REAL NOT NULL DEFAULT 0",
                },
            )
            self._add_missing_columns(
                connection,
                "feedback",
                {"corrected_abuse_label": "TEXT NOT NULL DEFAULT 'normal'"},
            )

    @staticmethod
    def _add_missing_columns(
        connection: sqlite3.Connection, table: str, columns: dict[str, str]
    ) -> None:
        """为已存在的库补齐新增列，避免旧数据被丢弃。"""
        existing = {
            row["name"] for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
        }
        for name, definition in columns.items():
            if name not in existing:
                connection.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")

    def record_detection(
        self, text: str, label: str, score: float, abuse_label: str = "normal", abuse_score: float = 0.0
    ) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                INSERT INTO detections (text_hash, text_length, label, score, abuse_label, abuse_score, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    len(text),
                    label,
                    score,
                    abuse_label,
                    abuse_score,
                    _now(),
                ),
            )

    def record_feedback(
        self,
        text: str,
        predicted_label: str,
        corrected_label: str,
        note: str,
        corrected_abuse_label: str = "normal",
    ) -> int:
        with closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                """
                INSERT INTO feedback (text, predicted_label, corrected_label, corrected_abuse_label, note, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (text, predicted_label, corrected_label, corrected_abuse_label, note, _now()),
            )
            return int(cursor.lastrowid)

    def stats(self) -> dict[str, int | float]:
        with closing(self._connect()) as connection, connection:
            totals = connection.execute(
                """
                SELECT COUNT(*) AS total,
                       SUM(CASE WHEN label = 'spam' THEN 1 ELSE 0 END) AS spam,
                       SUM(CASE WHEN label = 'review' THEN 1 ELSE 0 END) AS review,
                       SUM(CASE WHEN abuse_label IN ('spam', 'review') THEN 1 ELSE 0 END) AS abuse,
                       AVG(score) AS average_score
                FROM detections
                """
            ).fetchone()
            feedback_count = connection.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        return {
            "total": totals["total"] or 0,
            "spam": totals["spam"] or 0,
            "review": totals["review"] or 0,
            "abuse": totals["abuse"] or 0,
            "feedback": feedback_count,
            "average_score": round(totals["average_score"] or 0, 4),
        }


def _now() -> str:
    return datetime.now(UTC).isoformat()
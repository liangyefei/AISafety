from __future__ import annotations

import re

from pypinyin import Style, lazy_pinyin

from .classifier import normalize_text


def phonetic_text(text: str) -> str:
    normalized = normalize_text(text)
    phonetic = "".join(
        lazy_pinyin(
            normalized,
            style=Style.NORMAL,
            errors=lambda characters: list(characters),
        )
    )
    return re.sub(r"[^a-z0-9]+", "", phonetic)
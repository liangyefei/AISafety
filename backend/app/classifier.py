from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .pipeline import SLUR_RULE_WEIGHT, load_pipeline

PIPELINE = load_pipeline()


DIGIT_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}
CHINESE_DIGITS = str.maketrans("零〇一二两三四五六七八九", "001223456789")
_DIGIT_WORD = "(?:zero|one|two|three|four|five|six|seven|eight|nine|\\d)"
ENGLISH_DIGIT_SEQUENCE = re.compile(
    rf"(?<![a-z]){_DIGIT_WORD}(?:[\s._()\-]*{_DIGIT_WORD})+(?![a-z])",
    re.I,
)
CHINESE_DIGIT_SEQUENCE = re.compile(r"[0-9零〇一二两三四五六七八九]{3,}")
PARENTHESIZED_DIGIT = re.compile(r"\(([0-9])\)")


@dataclass(frozen=True)
class Rule:
    rule_id: str
    category: str
    reason: str
    pattern: re.Pattern[str]
    weight: float


@dataclass(frozen=True)
class Match:
    rule_id: str
    category: str
    reason: str
    weight: float


RULES = tuple(
    Rule(spec.id, spec.category, spec.reason, spec.compiled(), spec.weight)
    for spec in PIPELINE.spam_rules
)

# 隐式歧视俚语：来自 STATE-ToxiCN 学术词表，运行时从 datasets/slurs.txt 载入。
# 这些词（如「嘿忍」「小氙钕」「偷井盖」）几乎不含显式脏话，规则是主要防线。
SLUR_PATH = Path(__file__).resolve().parents[1] / "datasets" / "slurs.txt"


def load_slurs() -> tuple[str, ...]:
    """读取俚语词表；文件缺失时返回空元组，规则自动跳过。"""
    if not SLUR_PATH.exists():
        return ()
    return tuple(
        line.strip()
        for line in SLUR_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _build_abuse_rules() -> tuple[Rule, ...]:
    rules = [
        Rule(spec.id, spec.category, spec.reason, spec.compiled(), spec.weight)
        for spec in PIPELINE.abuse_rules
        if spec.pattern
    ]
    slurs = load_slurs()
    if slurs:
        rules.append(
            Rule(
                "abuse-slur",
                "歧视言论",
                "包含隐式歧视俚语",
                re.compile("|".join(map(re.escape, slurs))),
                SLUR_RULE_WEIGHT,
            )
        )
    return tuple(rules)


ABUSE_RULES = _build_abuse_rules()

RULE_SPAM_THRESHOLD = PIPELINE.thresholds.rule_spam
MODEL_SPAM_THRESHOLD = PIPELINE.thresholds.model_spam
REVIEW_THRESHOLD = PIPELINE.thresholds.review
ABUSE_SPAM_THRESHOLD = PIPELINE.thresholds.abuse_spam
ABUSE_SUSPICIOUS_THRESHOLD = PIPELINE.thresholds.abuse_suspicious
ABUSE_REVIEW_THRESHOLD = PIPELINE.thresholds.abuse_review
RULE_BOOST_FACTOR = PIPELINE.fusion.rule_boost_factor
LONG_TEXT_BONUS = PIPELINE.fusion.long_text_bonus
LONG_TEXT_THRESHOLD = PIPELINE.fusion.long_text_threshold


def apply_pipeline(pipeline: Any) -> None:
    """把新配置热加载到模块级变量，改权重后无需重启服务。

    规则按 id 复用已编译的正则，避免每次调参都重新编译上百条正则。
    """
    global RULES, ABUSE_RULES, PIPELINE
    global RULE_SPAM_THRESHOLD, MODEL_SPAM_THRESHOLD, REVIEW_THRESHOLD
    global ABUSE_SPAM_THRESHOLD, ABUSE_SUSPICIOUS_THRESHOLD, ABUSE_REVIEW_THRESHOLD
    global RULE_BOOST_FACTOR, LONG_TEXT_BONUS, LONG_TEXT_THRESHOLD

    compiled_spam = {rule.rule_id: rule.pattern for rule in RULES}
    compiled_abuse = {rule.rule_id: rule.pattern for rule in ABUSE_RULES}
    PIPELINE = pipeline
    RULES = tuple(
        Rule(spec.id, spec.category, spec.reason, compiled_spam.get(spec.id) or spec.compiled(), spec.weight)
        for spec in pipeline.spam_rules
    )
    # pipeline 里的 abuse-slur 占位规则 pattern 为空，此处跳过，末尾按词表重建。
    ABUSE_RULES = tuple(
        Rule(
            spec.id,
            spec.category,
            spec.reason,
            compiled_abuse.get(spec.id) or spec.compiled(),
            spec.weight,
        )
        for spec in pipeline.abuse_rules
        if spec.pattern
    )

    slurs = load_slurs()
    if slurs:
        ABUSE_RULES = ABUSE_RULES + (
            Rule(
                "abuse-slur",
                "歧视言论",
                "包含隐式歧视俚语",
                re.compile("|".join(map(re.escape, slurs))),
                SLUR_RULE_WEIGHT,
            ),
        )

    thresholds = pipeline.thresholds
    RULE_SPAM_THRESHOLD = thresholds.rule_spam
    MODEL_SPAM_THRESHOLD = thresholds.model_spam
    REVIEW_THRESHOLD = thresholds.review
    ABUSE_SPAM_THRESHOLD = thresholds.abuse_spam
    ABUSE_SUSPICIOUS_THRESHOLD = thresholds.abuse_suspicious
    ABUSE_REVIEW_THRESHOLD = thresholds.abuse_review
    RULE_BOOST_FACTOR = pipeline.fusion.rule_boost_factor
    LONG_TEXT_BONUS = pipeline.fusion.long_text_bonus
    LONG_TEXT_THRESHOLD = pipeline.fusion.long_text_threshold


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).lower()
    normalized = re.sub(r"[\u200b-\u200f\ufeff]", "", normalized)
    normalized = re.sub(r"(.)\1{4,}", r"\1\1", normalized)
    normalized = PARENTHESIZED_DIGIT.sub(r"\1", normalized)
    normalized = CHINESE_DIGIT_SEQUENCE.sub(
        lambda match: match.group(0).translate(CHINESE_DIGITS),
        normalized,
    )
    normalized = ENGLISH_DIGIT_SEQUENCE.sub(_replace_english_digits, normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _replace_english_digits(match: re.Match[str]) -> str:
    parts = re.findall(r"[a-z]+|\d", match.group(0), re.I)
    if not any(part.lower() in DIGIT_WORDS for part in parts):
        return match.group(0)
    return "".join(DIGIT_WORDS.get(part.lower(), part) for part in parts)


def score_text(text: str) -> tuple[float, tuple[Match, ...]]:
    normalized = normalize_text(text)
    matches = tuple(
        Match(rule.rule_id, rule.category, rule.reason, rule.weight)
        for rule in RULES
        if rule.pattern.search(normalized)
    )
    combined_risk = 1.0
    for match in matches:
        combined_risk *= 1.0 - match.weight
    score = 1.0 - combined_risk
    if len(normalized) > LONG_TEXT_THRESHOLD:
        score = min(1.0, score + LONG_TEXT_BONUS)
    return round(score, 4), matches


def score_abuse_rules(text: str) -> tuple[float, tuple[Match, ...]]:
    """规则层的辱骂评分，与营销规则彼此独立。

    正权重规则按乘性累积风险，负权重规则（反网暴、维权等正当表达）直接从结果中扣减。
    当文本同时命中俚语与正当表达时（如「关爱孤儿，不要网暴他」会命中词表中的相关词条），
    以正当表达为准并直接归零——举报网暴是维权行为，不应被当作辱骂。
    """
    normalized = normalize_text(text)
    matches = tuple(
        Match(rule.rule_id, rule.category, rule.reason, rule.weight)
        for rule in ABUSE_RULES
        if rule.pattern.search(normalized)
    )
    negative = [match for match in matches if match.weight < 0]
    positive = [match for match in matches if match.weight > 0]
    if negative:
        # 正当表达优先：不叠加任何辱骂信号
        return 0.0, matches

    combined = 1.0
    for match in positive:
        combined *= 1.0 - match.weight
    return round(max(0.0, min(1.0, 1.0 - combined)), 4), matches


def classify_text(text: str) -> dict[str, object]:
    score, matches = score_text(text)
    label = "spam" if score >= RULE_SPAM_THRESHOLD else "review" if score >= REVIEW_THRESHOLD else "normal"
    return {
        "label": label,
        "score": score,
        "reasons": list(dict.fromkeys(match.reason for match in matches)),
        "matched_rules": [match.rule_id for match in matches],
        "categories": list(dict.fromkeys(match.category for match in matches)),
        "model_version": "rules-v1",
        "rule_score": score,
        "model_score": None,
        "abuse_score": 0.0,
        "abuse_label": "normal",
        "abuse_suspicious": False,
        "abuse_reasons": [],
        "abuse_categories": [],
    }


class SpamClassifier:
    """营销垃圾与辱骂攻击使用两个独立模型，避免互相污染判定边界。

    label 表示是否需要处置（spam/review/normal），abuse_label 单独标注辱骂维度，
    因此「辱骂但不营销」的文本可以独立统计与处置。
    """

    def __init__(self, model_path: Path | None = None, abuse_model_path: Path | None = None) -> None:
        self.model_path = model_path
        self.model: Any | None = None
        self.abuse_model_path = abuse_model_path
        self.abuse_model: Any | None = None
        if model_path and model_path.exists():
            import joblib

            self.model = joblib.load(model_path)
        if abuse_model_path and abuse_model_path.exists():
            import joblib

            self.abuse_model = joblib.load(abuse_model_path)

    def _abuse_model_score(self, text: str) -> float:
        if self.abuse_model is None:
            return 0.0
        return float(self.abuse_model.predict_proba([normalize_text(text)])[0][1])

    def _apply_abuse(self, result: dict[str, object], text: str) -> None:
        rule_score, matches = score_abuse_rules(text)
        model_score = self._abuse_model_score(text)
        abuse_score = round(max(rule_score, model_score), 4)
        abuse_label = (
            "spam" if abuse_score >= ABUSE_SPAM_THRESHOLD
            else "review" if abuse_score >= ABUSE_REVIEW_THRESHOLD
            else "normal"
        )
        reasons = [match.reason for match in matches if match.weight > 0]
        if self.abuse_model is not None and model_score >= ABUSE_REVIEW_THRESHOLD:
            reasons.append("模型识别出辱骂或攻击性表达")
        result["abuse_score"] = abuse_score
        result["abuse_rule_score"] = rule_score
        result["abuse_model_score"] = round(model_score, 4)
        result["abuse_label"] = abuse_label
        result["abuse_suspicious"] = (
            ABUSE_REVIEW_THRESHOLD <= abuse_score < ABUSE_SPAM_THRESHOLD
        )
        result["abuse_reasons"] = list(dict.fromkeys(reasons))
        result["abuse_categories"] = list(
            dict.fromkeys(match.category for match in matches if match.weight > 0)
        )
        if abuse_label != "normal":
            categories = list(result["categories"])
            categories.append("辱骂攻击")
            result["categories"] = list(dict.fromkeys(categories))

    def predict(self, text: str) -> dict[str, object]:
        result = classify_text(text)
        if self.model is None:
            self._apply_abuse(result, text)
            return result

        rule_score = float(result["score"])
        model_score = float(self.model.predict_proba([normalize_text(text)])[0][1])
        boosted_model_score = model_score + (1.0 - model_score) * rule_score * RULE_BOOST_FACTOR
        score = max(rule_score, boosted_model_score)
        result["score"] = round(score, 4)
        # 双阈值：规则分够高可独立判定拦截（纯规则路径），否则走模型/复核线。
        # 之前 RULE_SPAM_THRESHOLD 在加载模型的路径上从未生效，调高它形同虚设。
        if rule_score >= RULE_SPAM_THRESHOLD:
            label = "spam"
        elif score >= MODEL_SPAM_THRESHOLD:
            label = "spam"
        elif score >= REVIEW_THRESHOLD:
            label = "review"
        else:
            label = "normal"
        result["label"] = label
        result["model_version"] = self.model_path.stem
        result["rule_score"] = rule_score
        result["model_score"] = round(model_score, 4)
        if model_score >= REVIEW_THRESHOLD:
            reasons = list(result["reasons"])
            categories = list(result["categories"])
            reasons.append("模型识别出垃圾文本模式")
            categories.append("统计模型")
            result["reasons"] = list(dict.fromkeys(reasons))
            result["categories"] = list(dict.fromkeys(categories))
        self._apply_abuse(result, text)
        return result
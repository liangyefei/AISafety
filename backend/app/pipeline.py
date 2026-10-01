"""检测流水线的唯一事实来源。

把此前散落在 classifier.py / train.py 的规则权重、特征权重、融合系数与阈值
集中到一处定义，运行时与前端展示共用同一份配置，避免「文档与实现不一致」。

配置可通过 SPAM_PIPELINE_PATH 指向外部 JSON 覆盖默认值，
运行时的调参结果也写入该文件，形成「配置 -> 行为」的闭环。
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

PIPELINE_VERSION = "1.2.0"

# 运行时保存调参结果的默认位置。
DEFAULT_OVERRIDE = Path(__file__).resolve().parents[1] / "pipeline.json"

# 特征权重固化在已训练的模型里：改它等价于对未经训练的缩放，
# 必须重新训练才有效，因此界面上标记为「需重训」。
REQUIRES_RETRAIN = frozenset({"feature_weights"})

# 俚语规则权重。不宜过高：词表含「小黑盒」「男娘」等中性搭配，
# 权重 0.45 时会把正常的技术/学术讨论推入 review（实测 86 条）。
SLUR_RULE_WEIGHT = 0.3


@dataclass(frozen=True)
class RuleSpec:
    """单条规则：正则、类别与权重。"""

    id: str
    category: str
    reason: str
    pattern: str
    weight: float
    flags: int = 0

    def compiled(self) -> re.Pattern[str]:
        return re.compile(self.pattern, self.flags)


@dataclass(frozen=True)
class FeatureSpec:
    """模型内部的特征通道及其权重。"""

    name: str
    label: str
    analyzer: str
    ngram_range: tuple[int, int]
    max_features: int
    weight: float
    note: str


@dataclass(frozen=True)
class ModelSpec:
    """一个判别模型：特征并联 -> 线性分类器。"""

    id: str
    label: str
    dimension: str
    path: str
    estimator: str
    features: tuple[FeatureSpec, ...]
    trained_rows: int
    note: str


@dataclass(frozen=True)
class FusionSpec:
    """规则分与模型分的融合方式。"""

    strategy: str
    rule_boost_factor: float
    aggregation: str
    long_text_bonus: float
    long_text_threshold: int


@dataclass(frozen=True)
class Thresholds:
    """各判定档位的分数线。"""

    rule_spam: float
    model_spam: float
    review: float
    abuse_spam: float
    abuse_suspicious: float
    abuse_review: float


@dataclass(frozen=True)
class Pipeline:
    version: str
    models: tuple[ModelSpec, ...]
    spam_rules: tuple[RuleSpec, ...]
    abuse_rules: tuple[RuleSpec, ...]
    fusion: FusionSpec
    thresholds: Thresholds
    normalization: tuple[str, ...] = field(default_factory=tuple)

    def model(self, model_id: str) -> ModelSpec:
        for spec in self.models:
            if spec.id == model_id:
                return spec
        raise KeyError(model_id)

    def describe(self) -> dict[str, Any]:
        """导出为可直接序列化的描述，供 API 与前端展示。"""
        return {
            "version": self.version,
            "normalization": list(self.normalization),
            "models": [
                {
                    "id": spec.id,
                    "label": spec.label,
                    "dimension": spec.dimension,
                    "path": spec.path,
                    "estimator": spec.estimator,
                    "trained_rows": spec.trained_rows,
                    "note": spec.note,
                    "features": [
                        {
                            "name": feature.name,
                            "label": feature.label,
                            "analyzer": feature.analyzer,
                            "ngram_range": list(feature.ngram_range),
                            "max_features": feature.max_features,
                            "weight": feature.weight,
                            "note": feature.note,
                        }
                        for feature in spec.features
                    ],
                }
                for spec in self.models
            ],
            "rules": {
                "spam": [
                    asdict(rule) | {"pattern": _abbreviate(rule.pattern)}
                    for rule in self.spam_rules
                ],
                "abuse": [
                    asdict(rule) | {"pattern": _abbreviate(rule.pattern)}
                    for rule in self.abuse_rules
                ],
            },
            "fusion": {
                "strategy": self.fusion.strategy,
                "rule_boost_factor": self.fusion.rule_boost_factor,
                "aggregation": self.fusion.aggregation,
                "long_text_bonus": self.fusion.long_text_bonus,
                "long_text_threshold": self.fusion.long_text_threshold,
            },
            "thresholds": {
                "rule_spam": self.thresholds.rule_spam,
                "model_spam": self.thresholds.model_spam,
                "review": self.thresholds.review,
                "abuse_spam": self.thresholds.abuse_spam,
                "abuse_suspicious": self.thresholds.abuse_suspicious,
                "abuse_review": self.thresholds.abuse_review,
            },
        }


def _abbreviate(pattern: str, limit: int = 72) -> str:
    """规则正则在界面上只展示摘要，避免整屏正则。"""
    return pattern if len(pattern) <= limit else pattern[: limit - 1] + "…"


# 两个模型共用同一套特征结构：字符级 TF-IDF 为主，拼音通道为辅。
# 拼音通道让「微信 / 薇信 / wei xin」落到同一特征空间，权重低于原字面以抑制噪声。
_SHARED_FEATURES = (
    FeatureSpec(
        name="characters",
        label="字符 TF-IDF",
        analyzer="char",
        ngram_range=(2, 5),
        max_features=80_000,
        weight=1.0,
        note="原字面 n-gram，主通道",
    ),
    FeatureSpec(
        name="pinyin",
        label="拼音 TF-IDF",
        analyzer="char",
        ngram_range=(2, 6),
        max_features=60_000,
        weight=0.7,
        note="谐音变体归一，权重低于原字面",
    ),
)

SPAM_RULES = (
    RuleSpec("gambling", "赌博诈骗", "包含赌博或博彩推广", r"博彩|下注|稳赚不赔|包赢|送彩金|棋牌代理", 0.72),
    RuleSpec("prize", "诱导诈骗", "包含中奖或领款诱导", r"恭喜.{0,8}(?:中奖|获奖)|点击.{0,8}(?:领奖|领取)|账户已被冻结", 0.58),
    RuleSpec("loan", "金融营销", "包含高风险贷款推广", r"无抵押|秒到账|黑户可贷|贷款额度|低息借款|强制下款", 0.55),
    RuleSpec("adult-lure", "色情引流", "包含色情内容引流", r"看片|福利视频|约炮|裸聊|成人视频|黄片|纯爱|小说", 0.55),
    RuleSpec("contact-wechat", "引流营销", "包含微信引流信息", r"(?:微信|薇信|微\s*信|v\s*x)\s*[:：]?\s*[a-z0-9_-]{5,}", 0.42, re.I),
    RuleSpec("contact-numeric-id", "联系方式", "客服或联系信息包含长数字标识", r"(?:客服|加我|联系|咨询)\s*[:：]?\s*\d{6,20}", 0.42),
    RuleSpec("contact-phone", "联系方式", "包含联系方式和行动诱导", r"(?:咨询|联系|加我|客服).{0,8}1[3-9]\d{9}", 0.36),
    RuleSpec("external-link", "外链推广", "包含可疑外部链接", r"(?:https?://|www\.|[a-z0-9-]+\.(?:com|cn|net|top|xyz))(?:\S*)", 0.28, re.I),
    RuleSpec("urgency", "诱导营销", "使用限时或名额诱导", r"限时|最后\d+个名额|仅限今天|过期不候|立即抢购", 0.20),
)

ABUSE_RULES = (
    RuleSpec("abuse-family", "辱骂攻击", "辱骂他人亲属", r"(?:你|他|她|它)(?:妈|母亲|爹|父亲)(?:的)?(?:死|滚|是狗|是猪|没屁眼|婊)|(?:妈|爹|爷|奶)死了|断子绝孙|不得好死|全家火葬|父母死了|没父母", 0.72),
    RuleSpec("abuse-insult", "辱骂攻击", "包含人身攻击或侮辱性称谓", r"傻[逼bi]|煞笔|沙比|智障|脑残|弱智|废物|垃圾人|滚[蛋开]?|去死|死全家|你妈|他妈|狗东西|畜生|王八|婊子|贱[人货]|贱种|杂种|蠢猪|猪狗", 0.60),
    RuleSpec("abuse-dehumanize", "辱骂攻击", "使用侮辱性贬损称谓", r"肥猪|脑瘫|畜生|禽兽|下三滥|人渣|贱人|舔狗|狗奴|窝囊废|没出息的东西|喷子|嘴贱|败犬|恶臭男|出生|见货|死狗|司马", 0.50),
    RuleSpec("abuse-obfuscated", "辱骂攻击", "包含谐音或缩写形式的辱骂", r"木琴|史学家|矢山|浮木|春竹|若至|唐氏|我来自你|sz东西|你个cs|真cs|见货|马勒戈壁|狗杂种", 0.50),
    RuleSpec("abuse-advocacy", "正当表达", "包含反网暴或维权表达", r"不要网暴|反对网暴|抵制|依法维权|举报|曝光|理性讨论|关爱(?:孤儿|残障)", -0.60),
)

DEFAULT_PIPELINE = Pipeline(
    version=PIPELINE_VERSION,
    models=(
        ModelSpec(
            id="spam",
            label="营销垃圾模型",
            dimension="spam",
            path="models/spam_pipeline_pinyin.joblib",
            estimator="LogisticRegression(class_weight=balanced)",
            features=_SHARED_FEATURES,
            trained_rows=21_946,
            note="判别诈骗、博彩、色情引流、黑产广告等营销类垃圾",
        ),
        ModelSpec(
            id="abuse",
            label="辱骂歧视模型",
            dimension="abuse",
            path="models/abuse_pipeline_pinyin.joblib",
            estimator="LogisticRegression(class_weight=balanced)",
            features=_SHARED_FEATURES,
            trained_rows=14_898,
            note="判别人身攻击与隐式歧视言论，含 STATE-ToxiCN 俚语特征",
        ),
    ),
    spam_rules=SPAM_RULES,
    abuse_rules=ABUSE_RULES + (
        (
            RuleSpec(
                "abuse-slur",
                "歧视言论",
                "包含隐式歧视俚语",
                "",
                SLUR_RULE_WEIGHT,
            ),
        )
    ),
    fusion=FusionSpec(
        strategy="规则分对模型分单向加权，模型分不可反向压制规则分",
        rule_boost_factor=0.5,
        aggregation="max(rule_score, boosted_model_score)",
        long_text_bonus=0.08,
        long_text_threshold=500,
    ),
    thresholds=Thresholds(
        rule_spam=0.72,
        model_spam=0.65,
        review=0.35,
        abuse_spam=0.65,
        abuse_suspicious=0.40,
        abuse_review=0.35,
    ),
    normalization=(
        "NFKC 规范化并转小写",
        "去除零宽与方向控制字符",
        "重复字符压缩（5 次及以上折叠为 2 次）",
        "带圈数字还原为阿拉伯数字",
        "连续中文数字串转阿拉伯数字",
        "英文数词序列转数字",
        "空白折叠",
    ),
)


def load_pipeline() -> Pipeline:
    """加载流水线配置；SPAM_PIPELINE_PATH 指向的 JSON 会覆盖默认值。"""
    override = os.getenv("SPAM_PIPELINE_PATH")
    if not override:
        return DEFAULT_PIPELINE
    path = Path(override)
    if not path.exists():
        return DEFAULT_PIPELINE
    payload = json.loads(path.read_text(encoding="utf-8"))
    base = DEFAULT_PIPELINE
    models = tuple(
        ModelSpec(
            id=item["id"],
            label=item["label"],
            dimension=item["dimension"],
            path=item["path"],
            estimator=item["estimator"],
            features=tuple(
                FeatureSpec(
                    name=feature["name"],
                    label=feature["label"],
                    analyzer=feature["analyzer"],
                    ngram_range=tuple(feature["ngram_range"]),  # type: ignore[arg-type]
                    max_features=feature["max_features"],
                    weight=feature["weight"],
                    note=feature.get("note", ""),
                )
                for feature in item["features"]
            ),
            trained_rows=item["trained_rows"],
            note=item.get("note", ""),
        )
        for item in payload.get("models", [asdict(spec) for spec in base.models])
    )
    rules_patch = payload.get("rules") or {}
    thresholds = payload.get("thresholds", {})
    fusion = payload.get("fusion", {})

    def with_weights(specs: tuple[RuleSpec, ...], patch: dict[str, float]) -> tuple[RuleSpec, ...]:
        """按 id 覆盖规则权重，未提及的保持默认。"""
        if not patch:
            return specs
        return tuple(
            replace(spec, weight=float(patch[spec.id])) if spec.id in patch else spec
            for spec in specs
        )

    return Pipeline(
        version=payload.get("version", base.version),
        models=models,
        spam_rules=with_weights(base.spam_rules, rules_patch.get("spam") or {}),
        abuse_rules=with_weights(base.abuse_rules, rules_patch.get("abuse") or {}),
        fusion=FusionSpec(
            strategy=fusion.get("strategy", base.fusion.strategy),
            rule_boost_factor=fusion.get("rule_boost_factor", base.fusion.rule_boost_factor),
            aggregation=fusion.get("aggregation", base.fusion.aggregation),
            long_text_bonus=fusion.get("long_text_bonus", base.fusion.long_text_bonus),
            long_text_threshold=fusion.get("long_text_threshold", base.fusion.long_text_threshold),
        ),
        thresholds=Thresholds(
            **{key: getattr(base.thresholds, key) for key in Thresholds.__annotations__}
            | {k: v for k, v in thresholds.items() if k in Thresholds.__annotations__}
        ),
        normalization=base.normalization,
    )


# 模块级默认配置，供 classifier / train / API 共享。
PIPELINE = DEFAULT_PIPELINE


class ConfigError(ValueError):
    """调参请求不合法。"""


def _bounded(value: object, name: str, low: float = -1.0, high: float = 1.0) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} 必须是数字，收到 {value!r}") from exc
    if not low <= number <= high:
        raise ConfigError(f"{name} 需在 [{low}, {high}] 之间，收到 {number}")
    return round(number, 4)


def apply_config(pipeline: Pipeline, patch: dict[str, Any]) -> Pipeline:
    """校验并应用调参补丁，返回新的 Pipeline（不修改入参）。"""

    thresholds = patch.get("thresholds") or {}
    updated = pipeline
    if thresholds:
        unknown = set(thresholds) - set(Thresholds.__annotations__)
        if unknown:
            raise ConfigError(f"未知阈值: {sorted(unknown)}")
        merged = {
            key: _bounded(value, f"thresholds.{key}", 0.0, 1.0)
            for key, value in asdict(pipeline.thresholds).items()
        }
        merged.update(
            {key: _bounded(value, f"thresholds.{key}", 0.0, 1.0) for key, value in thresholds.items()}
        )
        # review <= model_spam 是主判定链，必须递增。
        # rule_spam 是「规则分独立拦截」的旁路，只要求不低于 review，
        # 不必高于 model_spam——否则调低纯规则线就会被误判为非法。
        if merged["review"] > merged["model_spam"]:
            raise ConfigError("阈值必须递增：review <= model_spam")
        if merged["abuse_review"] > merged["abuse_suspicious"] or merged[
            "abuse_suspicious"
        ] > merged["abuse_spam"]:
            raise ConfigError("辱骂阈值必须递增：abuse_review <= abuse_suspicious <= abuse_spam")
        updated = replace(updated, thresholds=Thresholds(**merged))

    fusion_patch = patch.get("fusion") or {}
    if fusion_patch:
        base = asdict(updated.fusion)
        base["rule_boost_factor"] = _bounded(
            fusion_patch.get("rule_boost_factor", base["rule_boost_factor"]),
            "fusion.rule_boost_factor",
            0.0,
            1.0,
        )
        base["long_text_bonus"] = _bounded(
            fusion_patch.get("long_text_bonus", base["long_text_bonus"]),
            "fusion.long_text_bonus",
            0.0,
            0.5,
        )
        base["long_text_threshold"] = int(
            _bounded(
                fusion_patch.get("long_text_threshold", base["long_text_threshold"]),
                "fusion.long_text_threshold",
                1.0,
            )
        )
        updated = replace(updated, fusion=FusionSpec(**base))

    for group in ("spam", "abuse"):
        rule_patch = (patch.get("rules") or {}).get(group)
        if not rule_patch:
            continue
        # Pipeline 是 frozen dataclass，必须整体替换而不是 setattr。
        current = getattr(updated, f"{group}_rules")
        by_id = {spec.id: spec for spec in current}
        unknown = set(rule_patch) - set(by_id)
        if unknown:
            raise ConfigError(f"未知规则 ID（{group}）: {sorted(unknown)}")
        for rule_id, value in rule_patch.items():
            spec = by_id[rule_id]
            by_id[rule_id] = replace(
                spec, weight=_bounded(value, f"rules.{group}.{rule_id}", -1.0, 1.0)
            )
        updated = replace(updated, **{f"{group}_rules": tuple(by_id[spec.id] for spec in current)})

    feature_patch = (patch.get("models") or {})
    if feature_patch:
        models = []
        for spec in updated.models:
            override = feature_patch.get(spec.id)
            if not override:
                models.append(spec)
                continue
            features = tuple(
                replace(
                    feature,
                    weight=_bounded(
                        override.get("features", {}).get(feature.name, feature.weight),
                        f"models.{spec.id}.features.{feature.name}",
                        0.0,
                        5.0,
                    ),
                )
                for feature in spec.features
            )
            models.append(replace(spec, features=features))
        updated = replace(updated, models=tuple(models))

    return updated


def save_config(pipeline: Pipeline) -> Path:
    """把流水线写回 override 文件，供下次启动加载。

    规则权重必须一并落盘，否则重启后调参结果会丢失。
    """
    path = Path(os.getenv("SPAM_PIPELINE_PATH", DEFAULT_OVERRIDE))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "version": pipeline.version,
                "models": [asdict(spec) for spec in pipeline.models],
                "thresholds": asdict(pipeline.thresholds),
                "fusion": asdict(pipeline.fusion),
                "rules": {
                    "spam": {spec.id: spec.weight for spec in pipeline.spam_rules},
                    "abuse": {spec.id: spec.weight for spec in pipeline.abuse_rules},
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return path

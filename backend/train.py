from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from app.classifier import normalize_text
from app.features import phonetic_text
from app.pipeline import PIPELINE

# 拼音通道需要先做音译预处理，其余通道直接交给 TfidfVectorizer。
FEATURE_PREPROCESSORS = {"pinyin": phonetic_text}


def _with_preprocessor(spec) -> TfidfVectorizer:
    vectorizer = TfidfVectorizer(
        analyzer=spec.analyzer,
        ngram_range=spec.ngram_range,
        min_df=2,
        max_features=spec.max_features,
        sublinear_tf=True,
    )
    preprocessor = FEATURE_PREPROCESSORS.get(spec.name)
    if preprocessor is not None:
        vectorizer.set_params(preprocessor=preprocessor, lowercase=False)
    return vectorizer


LABEL_MAP = {"0": 0, "normal": 0, "ham": 0, "正常": 0, "1": 1, "spam": 1, "垃圾": 1}

# 留出集统一使用 text/label 列名；prepare_datasets.py 生成的文件即为此格式。
HOLDOUT_COLUMNS = ("text", "label")


def read_csv(path: Path, text_column: str, label_column: str) -> list[tuple[str, int]]:
    """读取单个 CSV，返回归一化后的 (文本, 标签) 列表。"""
    samples: list[tuple[str, int]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        header = set(reader.fieldnames or [])
        missing = {text_column, label_column} - header
        if missing:
            raise ValueError(f"{path.name} 缺少列 {sorted(missing)}，实际表头为 {sorted(header)}")
        for row_number, row in enumerate(reader, start=2):
            text = normalize_text(row.get(text_column) or "")
            raw_label = (row.get(label_column) or "").strip().lower()
            if not text:
                continue
            if raw_label not in LABEL_MAP:
                raise ValueError(f"{path.name} 第 {row_number} 行缺少有效的 {label_column}")
            samples.append((text, LABEL_MAP[raw_label]))
    if not samples:
        raise ValueError(f"{path.name} 没有可用样本")
    return samples


def load_rows(path: Path, text_column: str, label_column: str) -> tuple[list[str], list[int]]:
    samples, conflicts = resolve_conflicts(read_csv(path, text_column, label_column))
    return unpack(samples, conflicts, path.name)


def resolve_conflicts(
    rows: list[tuple[str, int]],
) -> tuple[dict[str, int], set[str]]:
    """跨源去重；同一文本出现矛盾标签时整条剔除。"""
    samples: dict[str, int] = {}
    conflicts: set[str] = set()
    for text, label in rows:
        if text not in samples:
            samples[text] = label
        elif samples[text] != label:
            conflicts.add(text)
    for text in conflicts:
        samples.pop(text, None)
    return samples, conflicts


def unpack(samples: dict[str, int], conflicts: set[str], name: str) -> tuple[list[str], list[int]]:
    texts = list(samples)
    labels = list(samples.values())
    if len(texts) < 100 or min(labels.count(0), labels.count(1)) < 30:
        raise ValueError(f"{name} 至少需要 100 条数据，且每类不少于 30 条")
    print(f"{name}：清洗后样本 {len(texts)}，冲突样本已剔除 {len(conflicts)}")
    return texts, labels


def build_pipeline(model_id: str = "spam") -> Pipeline:
    """按流水线配置构建特征与分类器，保证训练与推理使用同一套权重。"""
    spec = PIPELINE.model(model_id)
    features = FeatureUnion(
        [
            (
                feature.name,
                _with_preprocessor(feature),
            )
            for feature in spec.features
        ],
        transformer_weights={feature.name: feature.weight for feature in spec.features},
    )
    return Pipeline(
        [
            ("features", features),
            ("classifier", LogisticRegression(max_iter=1_000, class_weight="balanced", random_state=42)),
        ]
    )


def evaluate(
    pipeline: Pipeline,
    holdout_parts: list[tuple[Path, int | None]],
    train_samples: dict[str, int],
    positive_key: str,
) -> tuple[dict[str, object], int]:
    """在独立留出集上评估。

    holdout_parts 为 (文件, 强制标签) 列表；强制标签为 None 时沿用文件内的标签。
    这样可以把纯正样本的辱骂集与含正常样本的垃圾留出集合并评估，
    同时得到可比的查准率与查全率。
    """
    holdout_map: dict[str, int] = {}
    for path, override in holdout_parts:
        if not path.exists():
            print(f"  跳过不存在的留出集 {path}")
            continue
        for text, label in read_csv(path, *HOLDOUT_COLUMNS):
            holdout_map[text] = label if override is None else override
    if not holdout_map:
        raise ValueError("没有可用的留出样本")
    overlap = sum(1 for text in holdout_map if text in train_samples)
    texts = list(holdout_map)
    labels = [holdout_map[text] for text in texts]
    report = classification_report(labels, pipeline.predict(texts), output_dict=True, zero_division=0)
    print(
        f"外部留出集 {len(texts)} 条（与训练集重叠 {overlap} 条）"
        f" {positive_key} 查准={report['1']['precision']:.4f}"
        f" 查全={report['1']['recall']:.4f}"
        f" F1={report['1']['f1-score']:.4f}"
        f" 准确率={report['accuracy']:.4f}"
    )
    return report, overlap


def train_task(
    rows: list[tuple[str, int]],
    holdout_parts: list[tuple[Path, int | None]],
    output: Path,
    task_name: str,
    positive_key: str,
    model_id: str = "spam",
) -> None:
    samples, conflicts = resolve_conflicts(rows)
    texts, labels = unpack(samples, conflicts, f"{task_name}训练集")
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    pipeline = build_pipeline(model_id)
    pipeline.fit(train_texts, train_labels)
    report = classification_report(test_labels, pipeline.predict(test_texts), output_dict=True, zero_division=0)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output)
    metrics: dict[str, object] = {"internal_split": report}
    print(f"{task_name}内部留出 {positive_key} F1: {report['1']['f1-score']:.4f}")

    if holdout_parts:
        holdout_report, overlap = evaluate(pipeline, holdout_parts, samples, positive_key)
        metrics["holdout"] = holdout_report
        metrics["holdout_overlap_with_train"] = overlap

    output.with_suffix(".metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"模型已保存到 {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="训练中文垃圾文本与辱骂识别分类器")
    parser.add_argument("data", type=Path, nargs="+", help="一个或多个 UTF-8 CSV 文件")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--holdout", type=Path, help="独立测试集 CSV（text/label 列），不参与训练")
    parser.add_argument("--output", type=Path, default=Path("models/spam_pipeline_pinyin.joblib"))
    parser.add_argument(
        "--abuse-data",
        type=Path,
        nargs="*",
        default=None,
        help="辱骂正样本 CSV；同时会并入垃圾语料的正常样本作为负样本",
    )
    parser.add_argument("--abuse-holdout", type=Path, help="辱骂模型独立留出集 CSV")
    parser.add_argument(
        "--hate-data",
        type=Path,
        nargs="*",
        default=None,
        help="隐式歧视正样本 CSV（STATE-ToxiCN train）",
    )
    parser.add_argument(
        "--hate-negative",
        type=Path,
        nargs="*",
        default=None,
        help="困难负样本 CSV（非毒性讨论），用于压低误报",
    )
    parser.add_argument(
        "--hate-holdout", type=Path, help="隐式歧视留出集 CSV，与 train 同一来源"
    )
    parser.add_argument(
        "--benign-holdout",
        type=Path,
        nargs="*",
        default=None,
        help="困难负样本留出集（非毒性讨论），未参与训练",
    )
    parser.add_argument(
        "--abuse-output", type=Path, default=Path("models/abuse_pipeline_pinyin.joblib")
    )
    args = parser.parse_args()

    rows: list[tuple[str, int]] = []
    for path in args.data:
        rows.extend(read_csv(path, args.text_column, args.label_column))
    train_task(
        rows,
        [(args.holdout, None)] if args.holdout else [],
        args.output,
        "垃圾",
        "spam",
    )

    if not args.abuse_data and not args.hate_data:
        return

    abuse_rows: list[tuple[str, int]] = []
    for path in args.abuse_data or []:
        abuse_rows.extend(read_csv(path, args.text_column, args.label_column))
    hate_rows: list[tuple[str, int]] = []
    for path in args.hate_data or []:
        hate_rows.extend(read_csv(path, args.text_column, args.label_column))
    hard_negatives: list[tuple[str, int]] = []
    for path in args.hate_negative or []:
        hard_negatives.extend(read_csv(path, args.text_column, args.label_column))

    # 负样本含两类：通用正常文本，以及同主题的非毒性讨论（困难负样本）。
    # 后者对压低「讨论歧视话题 ≠ 歧视言论」的误报至关重要。
    negatives = [(text, 0) for text, label in rows if label == 0]
    negatives += [(text, 0) for text, label in hard_negatives]
    print(
        f"\n辱骂模型：显式辱骂 {len(abuse_rows)} 条 + 隐式歧视 {len(hate_rows)} 条正样本，"
        f"负样本 {len(negatives)} 条（含困难负样本 {len(hard_negatives)} 条）"
    )
    # 留出集为纯正样本，需并入未参与训练的负样本才能同时衡量查准与查全。
    # 注意：--hate-negative 已用于训练，不能再进留出集，否则查准率会虚高。
    abuse_holdout = (
        ([(args.abuse_holdout, 1)] if args.abuse_holdout else [])
        + ([(args.hate_holdout, 1)] if args.hate_holdout else [])
        + ([(args.holdout, 0)] if args.holdout else [])
        + ([(path, 0) for path in (args.benign_holdout or [])])
    )
    train_task(
        abuse_rows + hate_rows + negatives, abuse_holdout, args.abuse_output, "辱骂", "abuse",
        model_id="abuse",
    )


if __name__ == "__main__":
    main()
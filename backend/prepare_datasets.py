"""从 Hugging Face 公开仓库拉取并转换为统一的两分类 CSV 训练语料。

映射原则：只有「营销/引流/诈骗/色情资源/博彩」类才属于本项目的 spam 语义。
辱骂攻击、越狱问句、选择题、LLM 对话等不映射为 spam，避免污染分类器。

各源许可证由上传方声明，商用前需自行确认；本脚本只做技术性转换。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import urllib.parse
from collections import Counter
from pathlib import Path

import requests

from app.classifier import normalize_text

sys.stdout.reconfigure(encoding="utf-8")

HF_BASE = "https://huggingface.co/datasets/"

# ChineseHarm-bench 的有害类别 -> 本项目二分类标签。
# 「谩骂引战」是人身攻击而非营销垃圾，因此单独导出为 abuse.csv，
# 由独立的辱骂模型消费，避免污染营销垃圾的二分类边界。
HARM_LABEL_MAP = {
    "低俗色情": 1,
    "博彩": 1,
    "欺诈": 1,
    "黑产广告": 1,
    "不违规": 0,
}
HARM_ABUSE_LABEL = "谩骂引战"

# 明确的辱骂特征（含语料中常见的谐音、拼音缩写规避写法）
INSULT_PATTERN = re.compile(
    r"傻[逼bi]|煞笔|沙比|智障|脑残|弱智|废物|滚[蛋开]?|去死|死全家|你妈|他妈|你爹"
    r"|狗东西|畜生|王八|婊子|贱[人货]|贱种|杂种|蠢猪|肥猪|脑瘫|禽兽|人渣|舔狗"
    r"|窝囊废|没出息|下三滥|死狗|喷子|嘴贱|败犬|恶臭男|见货|司马|马勒戈壁"
    r"|木琴|史学家|矢山|浮木|春竹|若至|唐氏|sz东西|你个cs|真cs|我来自你|父母死了|没父母"
)
# 正当表达：反网暴、维权、举报等，不应计入辱骂
ADVOCACY_PATTERN = re.compile(
    r"不要网暴|反对网暴|抵制|依法维权|理性讨论|关爱(?:孤儿|残障)|批评教育"
)

# STATE-ToxiCN：中文隐式仇恨言论语料与俚语词表
# 该语料覆盖性别/种族/地域/性向歧视，是显式脏话模型的能力盲区。
TOXICN_RAW = "https://raw.githubusercontent.com/shenmeyemeifashengguo/STATE-ToxiCN/main/data/"
# 词表中大量单字（默/婊/艹/它）与纯 ASCII 缩写（eg/bt/nt）会造成严重误伤，
# 因此只采用长度 >= 2 且含中文的词条。
MIN_SLUR_LENGTH = 2


def load_toxicn_slurs(cache: Path) -> list[str]:
    """读取 STATE-ToxiCN 词表，筛出可用于规则匹配的隐式歧视俚语。"""
    path = download(TOXICN_RAW + "annotated%20lexicon.json", cache / "toxicn" / "lexicon.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    terms = {
        str(item.get("term", "")).strip()
        for item in payload.get("terms", [])
        if isinstance(item, dict)
    }
    return sorted(
        term
        for term in terms
        if len(term) >= MIN_SLUR_LENGTH and re.search(r"[\u4e00-\u9fff]", term)
    )


def rows_from_toxicn(cache: Path) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """STATE-ToxiCN train 集：返回 (毒性样本, 非毒性样本)。

    仅使用 train.json；test.json 保留给评测集，避免数据泄漏。
    """
    path = download(TOXICN_RAW + "train.json", cache / "toxicn" / "train.json")
    toxic: list[tuple[str, int]] = []
    benign: list[tuple[str, int]] = []
    for item in json.loads(path.read_text(encoding="utf-8")):
        text = (item.get("content") or "").strip()
        if not text:
            continue
        (toxic if str(item.get("sen_hate")) == "1" else benign).append((text, int(str(item.get("sen_hate")))))
    return toxic, benign


def download(url: str, destination: Path, attempts: int = 4) -> Path:
    """下载到本地缓存，已存在则复用；网络抖动时指数退避重试。"""
    if destination.exists() and destination.stat().st_size > 0:
        print(f"  复用缓存 {destination.name}（{destination.stat().st_size} 字节）")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, attempts + 1):
        print(f"  下载 {url}（第 {attempt}/{attempts} 次）")
        try:
            response = requests.get(url, timeout=120, headers={"User-Agent": "spam-filter-dataset/1.0"})
            if response.status_code in {401, 403, 404}:
                raise SystemExit(f"下载失败 {url}: HTTP {response.status_code}")
            response.raise_for_status()
            destination.write_bytes(response.content)
            return destination
        except requests.RequestException as error:
            print(f"  网络错误 {type(error).__name__}，准备重试")
        time.sleep(min(2**attempt, 15))
    raise SystemExit(f"重试 {attempts} 次后仍无法下载 {url}")


def rows_from_harm(directory: Path) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """ChineseHarm-bench：按类别分文件存放，字段为「文本」「标签」。

    返回 (营销类二分类样本, 辱骂样本)。未在映射表中的类别会被跳过。
    """
    spam_rows: list[tuple[str, int]] = []
    abuse_rows: list[tuple[str, int]] = []
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for item in payload:
            label = (item.get("标签") or "").strip()
            text = item.get("文本") or ""
            if label == HARM_ABUSE_LABEL:
                abuse_rows.append((text, 1))
            elif label in HARM_LABEL_MAP:
                spam_rows.append((text, HARM_LABEL_MAP[label]))
    return spam_rows, abuse_rows


def rows_from_csv(path: Path) -> list[tuple[str, int]]:
    """Olivi-9 / Moricoo：两列 CSV，label 列为 0/1。"""
    samples: list[tuple[str, int]] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            raw_label = (row.get("label") or "").strip()
            if raw_label not in {"0", "1"}:
                continue
            samples.append((row.get("text") or "", int(raw_label)))
    return samples


def clean_abuse(samples: list[tuple[str, int]]) -> tuple[dict[str, int], int]:
    """过滤谩骂语料中的标签噪声。

    ChineseHarm-bench 的「谩骂引战」含有大量无辱骂特征的口语、
    游戏梗与反网暴表达，直接训练会让模型把「就是」「这种人」等
    中性词学成辱骂信号。这里只保留含明确辱骂特征且非正当表达的正样本。
    """
    kept: dict[str, int] = {}
    dropped = 0
    for text, label in samples:
        cleaned = normalize_text(text)
        if not cleaned:
            dropped += 1
            continue
        if ADVOCACY_PATTERN.search(cleaned) or not INSULT_PATTERN.search(cleaned):
            dropped += 1
            continue
        kept[cleaned] = label
    return kept, dropped


def collect(samples: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """归一化、去空、精确去重；同文本标签冲突时以 spam 优先。"""
    merged: dict[str, int] = {}
    for text, label in samples:
        cleaned = normalize_text(text)
        if not cleaned:
            continue
        if merged.get(cleaned, label) != label:
            label = 1
        merged[cleaned] = label
    return list(merged.items())


def write_csv(path: Path, samples: list[tuple[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["text", "label"])
        writer.writerows(samples)
    counts = Counter(label for _, label in samples)
    print(
        f"写出 {path}：{len(samples)} 条"
        f"（normal={counts.get(0, 0)}，spam={counts.get(1, 0)}）"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="下载并转换公开中文垃圾文本数据集")
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("datasets/raw"),
        help="原始文件缓存目录",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/augmented.csv"),
        help="合并后的训练集输出路径",
    )
    parser.add_argument(
        "--holdout",
        type=Path,
        default=Path("datasets/holdout.csv"),
        help="外部独立测试集输出路径",
    )
    parser.add_argument(
        "--abuse",
        type=Path,
        default=Path("datasets/abuse.csv"),
        help="辱骂样本输出路径，供独立辱骂模型训练",
    )
    parser.add_argument(
        "--abuse-holdout",
        type=Path,
        default=Path("datasets/abuse_holdout.csv"),
        help="辱骂模型的独立留出集输出路径",
    )
    parser.add_argument(
        "--hate",
        type=Path,
        default=Path("datasets/hate.csv"),
        help="STATE-ToxiCN 隐式歧视言论输出路径",
    )
    parser.add_argument(
        "--hate-holdout",
        type=Path,
        default=Path("datasets/hate_holdout.csv"),
        help="隐式歧视留出集输出路径",
    )
    parser.add_argument(
        "--slurs",
        type=Path,
        default=Path("datasets/slurs.txt"),
        help="隐式歧视俚语词表输出路径，供分类器规则使用",
    )
    parser.add_argument(
        "--abuse-negative",
        type=Path,
        default=Path("datasets/hate_benign.csv"),
        help="辱骂模型的困难负样本（非毒性讨论）训练部分",
    )
    parser.add_argument(
        "--benign-holdout",
        type=Path,
        default=Path("datasets/hate_benign_holdout.csv"),
        help="困难负样本的留出部分，用于评估查准率",
    )
    args = parser.parse_args()
    cache = args.cache

    print("[1/5] ChineseHarm-bench（浙大，6 类有害内容）")
    harm_dir = cache / "chineseharm"
    for name in ("低俗色情", "博彩", "欺诈", "谩骂引战", "黑产广告", "不违规"):
        download(
            HF_BASE + "zjunlp/ChineseHarm-bench/resolve/main/"
            + urllib.parse.quote(name) + ".json",
            harm_dir / f"{name}.json",
        )
    harm, abuse = rows_from_harm(harm_dir)
    harm = collect(harm)
    abuse = collect(abuse)
    print(f"  营销类映射后 {len(harm)} 条；辱骂类 {len(abuse)} 条（单独建模）")

    print("[2/5] Olivi-9/Chinese-Spam-Dataset（真实群聊垃圾）")
    olivi_path = download(
        HF_BASE + "Olivi-9/Chinese-Spam-Dataset/resolve/main/train.csv",
        cache / "olivi" / "train.csv",
    )
    olivi = collect(rows_from_csv(olivi_path))

    print("[3/5] Moricoo/chinese_spam_test（外部留出集）")
    holdout_path = download(
        HF_BASE + "Moricoo/chinese_spam_test/resolve/main/test.csv",
        cache / "moricoo" / "test.csv",
    )
    holdout = collect(rows_from_csv(holdout_path))

    print("[4/5] STATE-ToxiCN（隐式歧视言论）")
    hate_rows, hate_benign = rows_from_toxicn(cache)
    hate_train_map = dict(collect(hate_rows))
    hate_texts = sorted(hate_train_map)
    hate_holdout_count = len(hate_texts) // 5
    hate_holdout = [(t, hate_train_map[t]) for t in hate_texts[:hate_holdout_count]]
    hate_train = [(t, hate_train_map[t]) for t in hate_texts[hate_holdout_count:]]

    # 非毒性样本必须拆成两半：一半作训练负样本，一半作留出集。
    # 若整份都进训练，留出集上的查准率将失去意义（数据泄漏）。
    benign_map = dict(collect(hate_benign))
    benign_texts = sorted(benign_map)
    benign_holdout_count = len(benign_texts) // 5
    benign_holdout = [(t, 0) for t in benign_texts[:benign_holdout_count]]
    benign_train = [(t, 0) for t in benign_texts[benign_holdout_count:]]
    print(
        f"  毒性 {len(hate_train_map)} 条；非毒性 {len(benign_map)} 条"
        f"（训练负样本 {len(benign_train)}，留出 {len(benign_holdout)}）"
    )

    slurs = load_toxicn_slurs(cache)
    args.slurs.parent.mkdir(parents=True, exist_ok=True)
    args.slurs.write_text("\n".join(slurs) + "\n", encoding="utf-8")
    print(f"  俚语词表 {len(slurs)} 条（已剔除单字与纯 ASCII 缩写）")

    print("[5/5] 清洗辱骂语料并划分留出集")
    abuse_map, dropped = clean_abuse(abuse)
    abuse_texts = sorted(abuse_map)
    abuse_holdout_count = len(abuse_texts) // 5
    abuse_holdout = [(text, abuse_map[text]) for text in abuse_texts[:abuse_holdout_count]]
    abuse_train = [(text, abuse_map[text]) for text in abuse_texts[abuse_holdout_count:]]
    if dropped:
        print(f"  剔除 {dropped} 条不含辱骂特征或属于反网暴表达的噪声样本")

    write_csv(args.output, harm + olivi)
    write_csv(args.holdout, holdout)
    write_csv(args.abuse, abuse_train)
    write_csv(args.abuse_holdout, abuse_holdout)
    write_csv(args.hate, hate_train)
    write_csv(args.hate_holdout, hate_holdout)
    write_csv(args.abuse_negative, benign_train)
    write_csv(args.benign_holdout, benign_holdout)


if __name__ == "__main__":
    main()

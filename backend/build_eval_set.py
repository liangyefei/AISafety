"""从多个公开中文安全数据集构建统一评测集。

覆盖五个场景：营销引流、垃圾诈骗、低俗色情、暴力威胁、辱骂攻击，
外加正常文本作为负样本对照。

数据来源（均为公开可访问仓库）：
- zjunlp/ChineseHarm-bench        低俗色情 / 博彩 / 欺诈 / 黑产广告 / 不违规
- zjunlp/ChineseHarm-bench 谩骂引战  辱骂攻击
- STATE-ToxiCN                    歧视性仇恨言论（zhihu / tieba 真实语料）
- Olivi-9/Chinese-Spam-Dataset    真实群聊垃圾
- Moricoo/chinese_spam_test       同源独立测试集

评测集与训练集严格隔离：训练使用 base/augmented/abuse，本脚本只从
其余来源采样，并在输出前做精确去重。
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import requests

sys.stdout.reconfigure(encoding="utf-8")

GITHUB_RAW = "https://raw.githubusercontent.com/shenmeyemeifashengguo/STATE-ToxiCN/main/"

# 暴力威胁：需「暴力词 + 威胁语境」共现，避免把「砍价」「枪击报道」等误判。
# 仅含暴力词不足以判定暴力威胁。
VIOLENCE_PATTERN = re.compile(
    r"杀|捅|砍|弄死|打死|砍死|爆头|见血|血溅|割喉|勒死|踹死|去死|死全家"
    r"|人肉|弄残|废了你|打爆|血债|全家死光|不得好死|想死|去死吧"
)
# 威胁语境：指向「你/他/她/你们」等具体对象，或明确的行动意图
VIOLENCE_CONTEXT = re.compile(
    r"(?:你|您|他|她|它|你们|他们|她们|这货|这人|那货|全家|一家|有种|敢|试试|弄|砍|打|杀|捅)"
    r"|威胁|报复|弄死|整死|搞死|废了|见血|血溅|给我|等着|收拾"
)


def is_violence(text: str) -> bool:
    """暴力词与威胁语境共现才判定为暴力威胁。"""
    return bool(VIOLENCE_PATTERN.search(text) and VIOLENCE_CONTEXT.search(text))
# 色情：源语料大量使用暗语与规避写法。
# 采用「强信号词表」+「弱信号需共现」两级判定，避免把普通营销文本误标为色情。
PORN_STRONG = re.compile(
    r"看片|黄片|裸聊|约炮|成人视频|福利视频|做爱|性交|口交|无码|有码|磁力"
    r"|番号|里番|本子|涩图|涩涩|色色|一夜情|包夜|楼凤|援交|特殊服务|全套服务"
    r"|原味丝袜|出来喝茶|➕我出来|艹我|下体|发情|交配|自慰|手活|口活|3p"
)
# 弱信号需与联系方式/交易类共现才判定为色情
PORN_WEAK = re.compile(r"丝袜|御姐|萝莉|嫩模|调教|母狗|公狗|脱衣|开房|大雷|寂寞|空虚|资源")
PORN_CONTEXT = re.compile(r"微信|加我|扣|企鹅|qq|联系|私聊|上门|服务|价格|小时|全套|资源|➕|\+|滴滴")


def is_porn(text: str) -> bool:
    """强信号直接判定；弱信号需同时出现交易/联系语境。"""
    if PORN_STRONG.search(text):
        return True
    return bool(PORN_WEAK.search(text) and PORN_CONTEXT.search(text))


# 明确辱骂特征：含常见脏话、亲属攻击或语料中的谐音缩写
INSULT_PATTERN = re.compile(
    r"傻[逼bi]|煞笔|沙比|智障|脑残|弱智|废物|滚[蛋开]?|去死|死全家|你妈|他妈|你爹"
    r"|狗东西|畜生|王八|婊子|贱[人货]|贱种|杂种|蠢猪|肥猪|脑瘫|禽兽|人渣|舔狗"
    r"|窝囊废|没出息|下三滥|死狗|喷子|嘴贱|败犬|恶臭男|见货|司马|马勒戈壁"
    r"|木琴|史学家|矢山|浮木|春竹|若至|唐氏|sz东西|你个cs|真cs|我来自你|父母死了|没父母"
)

Sample = tuple[str, str, str]  # (文本, 来源, 类别)


def download(url: str, destination: Path) -> Path:
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=120, headers={"User-Agent": "spam-filter-eval/1.0"})
    response.raise_for_status()
    destination.write_bytes(response.content)
    return destination


def load_harm(cache: Path) -> dict[str, list[str]]:
    """ChineseHarm-bench：按类别读取原始 JSON。"""
    buckets: dict[str, list[str]] = defaultdict(list)
    for path in sorted((cache / "chineseharm").glob("*.json")):
        for item in json.loads(path.read_text(encoding="utf-8")):
            text = (item.get("文本") or "").strip()
            label = (item.get("标签") or "").strip()
            if text and label:
                buckets[label].append(text)
    return buckets


def load_toxicn(cache: Path) -> tuple[list[str], list[str]]:
    """STATE-ToxiCN：返回 (毒性样本, 非毒性样本)。"""
    path = download(GITHUB_RAW + "data/test.json", cache / "toxicn" / "test.json")
    toxic: list[str] = []
    benign: list[str] = []
    for item in json.loads(path.read_text(encoding="utf-8")):
        text = (item.get("content") or "").strip()
        if not text:
            continue
        (toxic if str(item.get("sen_hate")) == "1" else benign).append(text)
    return toxic, benign


def load_spam_csv(path: Path) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            text = (row.get("text") or "").strip()
            label = (row.get("label") or "").strip()
            if text and label in {"0", "1"}:
                rows.append((text, int(label)))
    return rows

def is_abuse(text: str) -> bool:
    """辱骂判定：含明确辱骂特征，或属于 STATE-ToxiCN 的歧视性仇恨言论。

    谩骂引战语料噪声很高（仅约 44% 含明确辱骂特征），需按特征过滤，
    否则「这队友没有浮木啊」这类普通吐槽会被计入。
    """
    if INSULT_PATTERN.search(text):
        return True
    return bool(HATE_TOPIC_PATTERN.search(text))


# 歧视性仇恨话题：地域/性别/种族/性向歧视（源自 STATE-ToxiCN 的 topic 分布）
HATE_TOPIC_PATTERN = re.compile(
    r"黑鬼|白鬼|黑人|白人|洋垃圾|黑皮|绿绿|黑人|老黑|黑猴|泰女"
    r"|男同|女同|同性恋|弯|直男癌|普信男|田园女|上海人|东北人|河南人|南方人|北方人"
)


def balanced(pools: dict[str, list[Sample]], total: int, seed: int) -> list[dict[str, str]]:
    """按类别配额采样：先给每类一个保底量，再按候选量比例分配剩余额度。

    等量采样会让候选最少的类别（色情/暴力）无法参与均衡评测，
    因此改为配额 + 比例分配，并在结果中保留真实类别比例。
    """
    rng = random.Random(seed)
    counts = {name: len(items) for name, items in pools.items()}
    smallest = min(counts.values())
    floor = min(smallest, total // (len(pools) * 2))
    remaining = total - floor * len(pools)
    pool_total = sum(max(0, count - floor) for count in counts.values())

    picked: list[dict[str, str]] = []
    for name, items in pools.items():
        pool = list(items)
        rng.shuffle(pool)
        quota = floor
        if pool_total > 0:
            quota += round(remaining * max(0, len(items) - floor) / pool_total)
        quota = min(quota, len(items))
        picked.extend({"text": t, "category": name, "source": s} for t, s, _ in pool[:quota])
    rng.shuffle(picked)
    return picked


def build(cache: Path, total: int, seed: int) -> list[dict[str, str]]:
    harm = load_harm(cache)
    toxic_toxic, toxic_benign = load_toxicn(cache)
    olivi = load_spam_csv(cache / "olivi" / "train.csv")
    moricoo = load_spam_csv(cache / "moricoo" / "test.csv")

    olivi_spam = [t for t, label in olivi if label == 1]
    olivi_normal = [t for t, label in olivi if label == 0]
    moricoo_spam = [t for t, label in moricoo if label == 1]
    moricoo_normal = [t for t, label in moricoo if label == 0]

    pools: dict[str, list[Sample]] = {
        "营销引流": [(t, "ChineseHarm-bench/黑产广告", "营销引流") for t in harm.get("黑产广告", [])]
        + [(t, "Olivi-9/Chinese-Spam-Dataset", "营销引流") for t in olivi_spam],
        "垃圾诈骗": [(t, "ChineseHarm-bench/博彩", "垃圾诈骗") for t in harm.get("博彩", [])]
        + [(t, "ChineseHarm-bench/欺诈", "垃圾诈骗") for t in harm.get("欺诈", [])]
        + [(t, "Moricoo/chinese_spam_test", "垃圾诈骗") for t in moricoo_spam],
        "低俗色情": [
            (t, "ChineseHarm-bench/低俗色情", "低俗色情")
            for t in harm.get("低俗色情", [])
            if is_porn(t)
        ],
        "暴力威胁": [
            (t, "ChineseHarm-bench/谩骂引战", "暴力威胁")
            for t in harm.get("谩骂引战", [])
            if is_violence(t)
        ]
        + [
            (t, "STATE-ToxiCN", "暴力威胁")
            for t in toxic_toxic
            if is_violence(t)
        ]
        + [
            (t, "ChineseHarm-bench/博彩", "暴力威胁")
            for t in harm.get("博彩", [])
            if is_violence(t)
        ]
        + [
            (t, "ChineseHarm-bench/欺诈", "暴力威胁")
            for t in harm.get("欺诈", [])
            if is_violence(t)
        ],
        "辱骂攻击": [
            (t, "ChineseHarm-bench/谩骂引战", "辱骂攻击")
            for t in harm.get("谩骂引战", [])
            if not is_violence(t) and is_abuse(t)
        ]
        + [(t, "STATE-ToxiCN", "辱骂攻击") for t in toxic_toxic if not is_violence(t)],
        "正常文本": [
            (t, "ChineseHarm-bench/不违规", "正常文本")
            for t in harm.get("不违规", [])
            if not is_porn(t) and not is_violence(t) and not is_abuse(t)
        ]
        + [
            (t, "STATE-ToxiCN", "正常文本")
            for t in toxic_benign
            if not is_porn(t) and not is_violence(t) and not is_abuse(t)
        ]
        + [
            (t, "Olivi-9/Chinese-Spam-Dataset", "正常文本")
            for t in olivi_normal
            if not is_porn(t) and not is_violence(t) and not is_abuse(t)
        ]
        + [
            (t, "Moricoo/chinese_spam_test", "正常文本")
            for t in moricoo_normal
            if not is_porn(t) and not is_violence(t) and not is_abuse(t)
        ],
    }

    for name, items in pools.items():
        print(f"  {name:8s} 候选 {len(items):6d} 条")
    smallest = min(len(v) for v in pools.values())
    if smallest * len(pools) < total:
        print(f"  提示: 最少类别仅 {smallest} 条，等量采样上限 {smallest * len(pools)}")

    samples = balanced(pools, total, seed)
    seen: set[str] = set()
    unique: list[dict[str, str]] = []
    for item in samples:
        if item["text"] not in seen:
            seen.add(item["text"])
            unique.append(item)
    print(f"去重后 {len(unique)} 条")
    return unique


def main() -> None:
    parser = argparse.ArgumentParser(description="构建中文安全评测集")
    parser.add_argument("--cache", type=Path, default=Path("datasets/raw"))
    parser.add_argument("--output", type=Path, default=Path("datasets/eval/eval_set.csv"))
    parser.add_argument("--total", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20240501)
    args = parser.parse_args()

    print("加载数据源:")
    samples = build(args.cache, args.total, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text", "category", "source"])
        writer.writeheader()
        writer.writerows(samples)

    counts: dict[str, int] = defaultdict(int)
    for item in samples:
        counts[item["category"]] += 1
    print(f"\n评测集 {len(samples)} 条已写入 {args.output}")
    for name, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {name:8s} {count:5d}")


if __name__ == "__main__":
    main()
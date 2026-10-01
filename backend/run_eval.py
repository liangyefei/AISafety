"""对评测集调用 /api/filter 主接口，计算各类别指标并导出 Excel。

判定口径：
- 营销垃圾维度：营销引流 / 垃圾诈骗 / 低俗色情 / 暴力威胁 期望 label=spam
- 辱骂维度：辱骂攻击 期望 abuse_label=spam
- 正常文本：期望 label=normal 且 abuse_label=normal
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

SPAM_CATEGORIES = {"营销引流", "垃圾诈骗", "低俗色情", "暴力威胁"}
ABUSE_CATEGORIES = {"辱骂攻击"}
NORMAL_CATEGORIES = {"正常文本"}
BATCH_SIZE = 50

def call_api(base: str, path: str, payload: dict[str, object]) -> list[dict[str, object]]:
    request = urllib.request.Request(
        base + path,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        body = json.loads(response.read().decode("utf-8"))
    return body["results"] if "results" in body else [body]


def predict_all(base: str, texts: list[str], retries: int = 3) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        chunk = texts[start : start + BATCH_SIZE]
        for attempt in range(1, retries + 1):
            try:
                results.extend(call_api(base, "/api/filter/batch", {"texts": chunk}))
                break
            except (urllib.error.URLError, TimeoutError, KeyError) as error:
                if attempt == retries:
                    raise RuntimeError(f"批次 {start} 重试 {retries} 次仍失败: {error}") from error
                time.sleep(1.5 * attempt)
        if (start // BATCH_SIZE) % 10 == 0:
            print(f"  已完成 {len(results)}/{len(texts)}", flush=True)
    return results


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def main() -> None:
    parser = argparse.ArgumentParser(description="执行评测并导出 Excel 报告")
    parser.add_argument("--eval-set", type=Path, default=Path("datasets/eval/eval_set.csv"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8787")
    parser.add_argument("--output", type=Path, default=Path("datasets/eval/eval_report.xlsx"))
    args = parser.parse_args()

    with args.eval_set.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    print(f"载入评测集 {len(rows)} 条，开始调用 {args.base_url}/api/filter/batch")
    started = time.time()
    results = predict_all(args.base_url, [row["text"] for row in rows])
    elapsed = time.time() - started
    print(f"接口调用完成，耗时 {elapsed:.1f}s（{len(results) / elapsed:.1f} 条/秒）")

    records: list[dict[str, object]] = []
    for row, result in zip(rows, results, strict=True):
        category = row["category"]
        if category in SPAM_CATEGORIES:
            expected = "spam"
            correct = result["label"] == "spam"
        elif category in ABUSE_CATEGORIES:
            expected = "abuse_spam"
            correct = result["abuse_label"] == "spam"
        else:
            expected = "normal"
            correct = result["label"] == "normal" and result["abuse_label"] == "normal"
        records.append(
            {
                "text": row["text"],
                "category": category,
                "source": row["source"],
                "expected": expected,
                "label": result["label"],
                "abuse_label": result["abuse_label"],
                "abuse_suspicious": result.get("abuse_suspicious", False),
                "score": result["score"],
                "abuse_score": result["abuse_score"],
                "rule_score": result["rule_score"],
                "model_score": result["model_score"],
                "correct": "✓" if correct else "✗",
                "categories": "|".join(result["categories"]),
                "reasons": "；".join(result["reasons"][:3]),
                "abuse_reasons": "；".join(result["abuse_reasons"][:2]),
            }
        )

    report = summarize(records)
    write_excel(args.output, records, report, elapsed, args.base_url)
    print_summary(report)
    print(f"\nExcel 报告已写入 {args.output}")


def summarize(records: list[dict[str, object]]) -> dict[str, object]:
    """按类别与总体计算准确率、查准、查全、F1。"""
    by_category: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "correct": 0})
    for record in records:
        bucket = by_category[str(record["category"])]
        bucket["n"] += 1
        bucket["correct"] += record["correct"] == "✓"

    # 垃圾维度二分类混淆矩阵
    tp = sum(
        1 for r in records
        if r["category"] in SPAM_CATEGORIES and r["label"] == "spam"
    )
    fp = sum(
        1 for r in records
        if r["category"] in NORMAL_CATEGORIES and r["label"] == "spam"
    )
    fn = sum(
        1 for r in records
        if r["category"] in SPAM_CATEGORIES and r["label"] != "spam"
    )
    tn = sum(
        1 for r in records
        if r["category"] in NORMAL_CATEGORIES and r["label"] == "normal"
    )
    spam_metrics = prf(tp, fp, fn)

    # 3) 辱骂维度：spam 与「可疑灰区」分开统计，便于评估人工复核的覆盖面
    ab_tp = sum(
        1 for r in records
        if r["category"] in ABUSE_CATEGORIES and r["abuse_label"] == "spam"
    )
    ab_fp = sum(
        1 for r in records
        if r["category"] in NORMAL_CATEGORIES and r["abuse_label"] == "spam"
    )
    ab_fn = sum(
        1 for r in records
        if r["category"] in ABUSE_CATEGORIES and r["abuse_label"] != "spam"
    )
    ab_tn = sum(
        1 for r in records
        if r["category"] in NORMAL_CATEGORIES and r["abuse_label"] == "normal"
    )
    abuse_metrics = prf(ab_tp, ab_fp, ab_fn)
    # 含灰区：abuse_label != normal 即视为被系统捕获
    caught = sum(
        1 for r in records
        if r["category"] in ABUSE_CATEGORIES and r["abuse_label"] != "normal"
    )
    abuse_total = sum(1 for r in records if r["category"] in ABUSE_CATEGORIES)
    abuse_flagged_normal = sum(
        1 for r in records
        if r["category"] in NORMAL_CATEGORIES and r["abuse_label"] != "normal"
    )

    total = len(records)
    correct = sum(1 for r in records if r["correct"] == "✓")
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "by_category": dict(by_category),
        "spam": spam_metrics | {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "abuse": abuse_metrics
        | {
            "tp": ab_tp, "fp": ab_fp, "fn": ab_fn, "tn": ab_tn,
            "caught": caught,
            "caught_rate": caught / abuse_total if abuse_total else 0.0,
            "flagged_normal": abuse_flagged_normal,
        },
    }
# openpyxl 拒绝 XML 1.0 不允许的控制字符，写入前需清洗
_ILLEGAL_XML = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff\ufffe\uffff]")


def cell_safe(value: object) -> object:
    """把字符串中的非法控制字符剔除，避免 openpyxl 写入失败。"""
    if isinstance(value, str):
        return _ILLEGAL_XML.sub("", value)
    return value


def write_excel(
    path: Path,
    records: list[dict[str, object]],
    report: dict[str, object],
    elapsed: float,
    base_url: str,
) -> None:
    """导出四个工作表：汇总、分类别指标、逐条明细、误判明细。"""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    workbook = Workbook()
    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill("solid", fgColor="2F5496")
    title_font = Font(bold=True, size=13)

    def style_header(sheet, columns: int) -> None:
        for index in range(1, columns + 1):
            cell = sheet.cell(row=1, column=index)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal="center")
        sheet.freeze_panes = "A2"

    def autosize(sheet, widths: dict[int, int]) -> None:
        for index, width in widths.items():
            sheet.column_dimensions[get_column_letter(index)].width = width

    # 1) 汇总
    summary = workbook.active
    summary.title = "汇总"
    spam = report["spam"]
    abuse = report["abuse"]
    rows = [
        ("评测接口", f"{base_url}/api/filter"),
        ("评测时间", time.strftime("%Y-%m-%d %H:%M:%S")),
        ("样本总量", report["total"]),
        ("判定正确", report["correct"]),
        ("整体准确率", f"{report['accuracy']:.2%}"),
        ("调用耗时(秒)", round(elapsed, 1)),
        ("平均吞吐(条/秒)", round(report["total"] / elapsed, 1) if elapsed else 0),
        ("", ""),
        ("【营销垃圾维度】", "期望: 营销/垃圾/色情/暴力 → label=spam；正常 → label=normal"),
        ("查准率", f"{spam['precision']:.2%}"),
        ("查全率", f"{spam['recall']:.2%}"),
        ("F1", f"{spam['f1']:.4f}"),
        ("TP/FP/FN/TN", f"{spam['tp']} / {spam['fp']} / {spam['fn']} / {spam['tn']}"),
        ("", ""),
        ("【辱骂维度】", "期望: 辱骂 → abuse_label=spam；正常 → abuse_label=normal"),
        ("查准率", f"{abuse['precision']:.2%}"),
        ("查全率", f"{abuse['recall']:.2%}"),
        ("F1", f"{abuse['f1']:.4f}"),
        ("TP/FP/FN/TN", f"{abuse['tp']} / {abuse['fp']} / {abuse['fn']} / {abuse['tn']}"),
    ]
    summary.append(["指标", "值"])
    for key, value in rows:
        summary.append([cell_safe(key), cell_safe(value)])
    style_header(summary, 2)
    summary["A1"].font = title_font
    autosize(summary, {1: 22, 2: 62})

    # 2) 分类别指标
    category_sheet = workbook.create_sheet("分类别指标")
    category_sheet.append(["类别", "样本数", "判定正确", "准确率", "主要期望"])
    expectation = {
        "营销引流": "label=spam",
        "垃圾诈骗": "label=spam",
        "低俗色情": "label=spam",
        "暴力威胁": "label=spam",
        "辱骂攻击": "abuse_label=spam",
        "正常文本": "label=normal 且 abuse=normal",
    }
    for name, bucket in report["by_category"].items():
        accuracy = bucket["correct"] / bucket["n"] if bucket["n"] else 0.0
        category_sheet.append(
            [name, bucket["n"], bucket["correct"], round(accuracy, 4), expectation.get(name, "")]
        )
    style_header(category_sheet, 5)
    autosize(category_sheet, {1: 14, 2: 10, 3: 12, 4: 10, 5: 32})

    # 3) 逐条明细
    detail = workbook.create_sheet("逐条明细")
    columns = [
        ("序号", 8), ("文本", 52), ("类别", 12), ("来源", 26), ("期望", 12),
        ("label", 9), ("abuse_label", 12), ("辱骂可疑", 10), ("垃圾分", 9), ("辱骂分", 9),
        ("规则分", 9), ("模型分", 9), ("是否正确", 10),
        ("命中类别", 24), ("垃圾依据", 34), ("辱骂依据", 24),
    ]
    detail.append([name for name, _ in columns])
    for index, record in enumerate(records, start=1):
        detail.append(
            [
                cell_safe(value)
                for value in (
                    index, record["text"], record["category"], record["source"], record["expected"],
                    record["label"], record["abuse_label"], "是" if record["abuse_suspicious"] else "",
                    record["score"], record["abuse_score"],
                    record["rule_score"], record["model_score"], record["correct"],
                    record["categories"], record["reasons"], record["abuse_reasons"],
                )
            ]
        )
    style_header(detail, len(columns))
    autosize(detail, {index: width for index, (_, width) in enumerate(columns, start=1)})

    # 4) 误判明细
    wrong = workbook.create_sheet("误判明细")
    wrong.append(["序号", "类别", "期望", "实际", "文本", "垃圾分", "辱骂分", "命中依据"])
    for index, record in enumerate(records, start=1):
        if record["correct"] == "✓":
            continue
        actual = f"{record['label']}/{record['abuse_label']}"
        wrong.append(
            [
                cell_safe(value)
                for value in (
                    index, record["category"], record["expected"],
                    f"{record['label']}/{record['abuse_label']}", record["text"],
                    record["score"], record["abuse_score"],
                    "；".join(filter(None, [record["reasons"], record["abuse_reasons"]])),
                )
            ]
        )
    style_header(wrong, 8)
    autosize(wrong, {1: 8, 2: 12, 3: 12, 4: 16, 5: 56, 6: 9, 7: 9, 8: 40})

    path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(path)


def print_summary(report: dict[str, object]) -> None:
    spam = report["spam"]
    abuse = report["abuse"]
    print("\n" + "=" * 62)
    print(f"整体准确率 {report['accuracy']:.2%}  ({report['correct']}/{report['total']})")
    print("-" * 62)
    print("分类别准确率:")
    for name, bucket in sorted(report["by_category"].items(), key=lambda kv: -kv[1]["n"]):
        accuracy = bucket["correct"] / bucket["n"] if bucket["n"] else 0.0
        print(f"  {name:8s} {bucket['correct']:5d}/{bucket['n']:5d}  {accuracy:7.2%}")
    print("-" * 62)
    print(
        f"营销垃圾维度  查准={spam['precision']:.2%} 查全={spam['recall']:.2%} "
        f"F1={spam['f1']:.4f}"
    )
    print(
        f"辱骂维度      查准={abuse['precision']:.2%} 查全={abuse['recall']:.2%} "
        f"F1={abuse['f1']:.4f}"
    )
    print(
        f"  含灰区覆盖   {abuse['caught_rate']:.2%}（{abuse['caught']}/{report['by_category'].get('辱骂攻击', {}).get('n', 0)}），"
        f"正常文本被标记 {abuse['flagged_normal']} 条"
    )
    print("=" * 62)


if __name__ == "__main__":
    main()
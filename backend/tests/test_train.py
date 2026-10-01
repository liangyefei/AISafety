import csv
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from app.pipeline import (
    DEFAULT_PIPELINE,
    PIPELINE,
    PIPELINE_VERSION,
    ConfigError,
    apply_config,
    load_pipeline,
)
from prepare_datasets import HARM_ABUSE_LABEL, HARM_LABEL_MAP, collect, rows_from_harm
from train import build_pipeline, load_rows, read_csv, resolve_conflicts


class PipelineConfigTests(unittest.TestCase):
    """流水线配置是唯一事实来源，训练与推理都必须读它。"""

    def test_describe_exposes_model_weights(self) -> None:
        described = DEFAULT_PIPELINE.describe()
        self.assertEqual(described["version"], PIPELINE_VERSION)
        self.assertEqual(len(described["models"]), 2)
        for model in described["models"]:
            weights = {f["name"]: f["weight"] for f in model["features"]}
            self.assertEqual(weights, {"characters": 1.0, "pinyin": 0.7}, model["id"])
            self.assertTrue(model["estimator"])
            self.assertGreater(model["trained_rows"], 0)

    def test_describe_exposes_rule_weights_and_thresholds(self) -> None:
        described = DEFAULT_PIPELINE.describe()
        spam_weights = {r["id"]: r["weight"] for r in described["rules"]["spam"]}
        self.assertEqual(spam_weights["gambling"], 0.72)
        self.assertEqual(spam_weights["urgency"], 0.20)
        abuse_weights = {r["id"]: r["weight"] for r in described["rules"]["abuse"]}
        # 负权重代表正当表达，应压过俚语命中
        self.assertLess(abuse_weights["abuse-advocacy"], 0)
        thresholds = described["thresholds"]
        self.assertEqual(thresholds["model_spam"], 0.65)
        self.assertEqual(thresholds["abuse_suspicious"], 0.40)
        self.assertEqual(described["fusion"]["rule_boost_factor"], 0.5)

    def test_classifier_reads_thresholds_from_pipeline(self) -> None:
        from app import classifier

        self.assertEqual(classifier.MODEL_SPAM_THRESHOLD, PIPELINE.thresholds.model_spam)
        self.assertEqual(classifier.RULE_BOOST_FACTOR, PIPELINE.fusion.rule_boost_factor)
        self.assertEqual(len(classifier.RULES), len(PIPELINE.spam_rules))

    def test_training_pipeline_uses_configured_feature_weights(self) -> None:
        built = build_pipeline("spam")
        union = built.named_steps["features"]
        self.assertEqual(union.transformer_weights, {"characters": 1.0, "pinyin": 0.7})

    def test_missing_override_file_falls_back_to_default(self) -> None:
        import os

        original = os.environ.get("SPAM_PIPELINE_PATH")
        os.environ["SPAM_PIPELINE_PATH"] = "no/such/file.json"
        try:
            self.assertIs(load_pipeline(), DEFAULT_PIPELINE)
        finally:
            if original is None:
                os.environ.pop("SPAM_PIPELINE_PATH", None)
            else:
                os.environ["SPAM_PIPELINE_PATH"] = original

    def test_override_file_changes_thresholds(self) -> None:
        import os

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pipeline.json"
            path.write_text(
                json.dumps({"thresholds": {"review": 0.5}}, ensure_ascii=False),
                encoding="utf-8",
            )
            original = os.environ.get("SPAM_PIPELINE_PATH")
            os.environ["SPAM_PIPELINE_PATH"] = str(path)
            try:
                loaded = load_pipeline()
                self.assertEqual(loaded.thresholds.review, 0.5)
                    # 未覆盖的字段保持默认
                self.assertEqual(loaded.thresholds.model_spam, 0.65)
            finally:
                if original is None:
                    os.environ.pop("SPAM_PIPELINE_PATH", None)
                else:
                    os.environ["SPAM_PIPELINE_PATH"] = original

    def test_api_exposes_pipeline(self) -> None:
        import tempfile as tf
        from pathlib import Path as P

        from fastapi.testclient import TestClient

        from app import main as app_main
        from app.storage import ResultStore

        app_main.store = ResultStore(P(tf.mkdtemp()) / "t.db")
        client = TestClient(app_main.app)
        try:
            payload = client.get("/api/pipeline").json()
            self.assertEqual(payload["version"], PIPELINE_VERSION)
            self.assertEqual(len(payload["models"]), 2)
            self.assertIn("thresholds", payload)
        finally:
            client.close()


class ConfigTuningTests(unittest.TestCase):
    """在线调参：校验、热加载与落盘。"""

    def test_threshold_patch_applies(self) -> None:
        updated = apply_config(PIPELINE, {"thresholds": {"review": 0.5}})
        self.assertEqual(updated.thresholds.review, 0.5)
        self.assertEqual(PIPELINE.thresholds.review, 0.35, "不得修改入参")

    def test_rule_weight_patch_applies(self) -> None:
        updated = apply_config(PIPELINE, {"rules": {"spam": {"gambling": 0.9}}})
        weights = {r.id: r.weight for r in updated.spam_rules}
        self.assertEqual(weights["gambling"], 0.9)
        self.assertEqual(weights["urgency"], 0.20, "未提及的规则应保持不变")

    def test_negative_rule_weight_is_allowed(self) -> None:
        updated = apply_config(PIPELINE, {"rules": {"abuse": {"abuse-insult": -0.3}}})
        weights = {r.id: r.weight for r in updated.abuse_rules}
        self.assertEqual(weights["abuse-insult"], -0.3)

    def test_unknown_keys_are_rejected(self) -> None:
        with self.assertRaises(ConfigError):
            apply_config(PIPELINE, {"thresholds": {"nope": 0.5}})
        with self.assertRaises(ConfigError):
            apply_config(PIPELINE, {"rules": {"spam": {"not_a_rule": 0.5}}})

    def test_out_of_range_is_rejected(self) -> None:
        with self.assertRaises(ConfigError):
            apply_config(PIPELINE, {"thresholds": {"review": 5}})
        with self.assertRaises(ConfigError):
            apply_config(PIPELINE, {"rules": {"spam": {"gambling": 99}}})

    def test_non_monotonic_thresholds_are_rejected(self) -> None:
        with self.assertRaises(ConfigError):
            apply_config(PIPELINE, {"thresholds": {"review": 0.9}})
        with self.assertRaises(ConfigError):
            apply_config(PIPELINE, {"thresholds": {"abuse_review": 0.9}})

    def test_rule_spam_may_sit_below_model_spam(self) -> None:
        """rule_spam 是独立旁路，调低它不应被判为非法配置。"""
        updated = apply_config(
            PIPELINE, {"thresholds": {"model_spam": 0.9, "rule_spam": 0.5}}
        )
        self.assertEqual(updated.thresholds.rule_spam, 0.5)
        self.assertEqual(updated.thresholds.model_spam, 0.9)

    def test_hot_reload_changes_classifier_behavior(self) -> None:
        """改阈值后无需重启即可改变判定结果。"""
        from app import classifier

        model_path = Path(__file__).resolve().parents[1] / "models" / "spam_pipeline_pinyin.joblib"
        if not model_path.exists():
            self.skipTest("模型文件不存在，跳过热加载行为验证")
        original = classifier.REVIEW_THRESHOLD
        try:
            # 阈值必须递增，因此同时抬高 rule_spam，保持 review <= model <= rule
            updated = apply_config(
                PIPELINE, {"thresholds": {"model_spam": 0.9, "rule_spam": 0.95}}
            )
            classifier.apply_pipeline(updated)
            self.assertEqual(classifier.MODEL_SPAM_THRESHOLD, 0.9)
            result = classifier.SpamClassifier(model_path).predict("博彩送彩金立即下注")
            self.assertEqual(result["label"], "review", "阈值抬高后不应再判 spam")
        finally:
            classifier.apply_pipeline(PIPELINE)
            self.assertEqual(classifier.REVIEW_THRESHOLD, original)

    def test_hot_reload_preserves_compiled_patterns(self) -> None:
        from app import classifier

        before = {r.rule_id: r.pattern for r in classifier.RULES}
        updated = apply_config(PIPELINE, {"rules": {"spam": {"gambling": 0.95}}})
        classifier.apply_pipeline(updated)
        try:
            after = {r.rule_id: (r.weight, r.pattern) for r in classifier.RULES}
            self.assertEqual(after["gambling"][0], 0.95)
            # 调参不应重新编译正则：对象应保持同一实例
            self.assertIs(after["gambling"][1], before["gambling"])
            self.assertTrue(after["gambling"][1].search("博彩送彩金"))
        finally:
            classifier.apply_pipeline(PIPELINE)

    def test_api_rejects_invalid_config_with_422(self) -> None:
        import tempfile as tf
        from pathlib import Path as P

        from fastapi.testclient import TestClient

        from app import main as app_main
        from app.storage import ResultStore

        app_main.store = ResultStore(P(tf.mkdtemp()) / "t.db")
        client = TestClient(app_main.app)
        try:
            bad = client.put("/api/pipeline", json={"thresholds": {"review": 9}})
            self.assertEqual(bad.status_code, 422)
            self.assertIn("thresholds.review", bad.json()["detail"])
        finally:
            client.close()

    def test_rule_weights_survive_save_and_reload(self) -> None:
        """调参结果必须落盘并在重启后保留。"""
        import os

        from app.pipeline import save_config

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pipeline.json"
            os.environ["SPAM_PIPELINE_PATH"] = str(path)
            try:
                tuned = apply_config(
                    PIPELINE,
                    {
                        "rules": {"spam": {"gambling": 0.88}, "abuse": {"abuse-insult": 0.44}},
                        "thresholds": {"review": 0.42},
                    },
                )
                save_config(tuned)
                reloaded = load_pipeline()
                spam = {r.id: r.weight for r in reloaded.spam_rules}
                abuse = {r.id: r.weight for r in reloaded.abuse_rules}
                self.assertEqual(spam["gambling"], 0.88)
                self.assertEqual(abuse["abuse-insult"], 0.44)
                self.assertEqual(reloaded.thresholds.review, 0.42)
                # 未调整的规则与阈值保持默认
                self.assertEqual(spam["urgency"], 0.20)
                self.assertEqual(reloaded.thresholds.model_spam, 0.65)
            finally:
                os.environ.pop("SPAM_PIPELINE_PATH", None)



class TrainingDataTests(unittest.TestCase):
    def test_loader_removes_empty_duplicates_and_conflicts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.csv"
            rows = [
                {"text": f"正常消息 {index}", "label": "0"}
                for index in range(60)
            ] + [
                {"text": f"垃圾消息 {index}", "label": "1"}
                for index in range(60)
            ]
            rows.extend(
                [
                    {"text": "正常消息 1", "label": "0"},
                    {"text": "  ", "label": "1"},
                    {"text": "标签冲突", "label": "0"},
                    {"text": "标签冲突", "label": "1"},
                ]
            )
            with path.open("w", encoding="utf-8", newline="") as target:
                writer = csv.DictWriter(target, fieldnames=["text", "label"])
                writer.writeheader()
                writer.writerows(rows)

            texts, labels = load_rows(path, "text", "label")

        self.assertEqual(len(texts), 120)
        self.assertEqual(labels.count(0), 60)
        self.assertEqual(labels.count(1), 60)
        self.assertNotIn("标签冲突", texts)

    def test_read_csv_rejects_missing_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrong.csv"
            with path.open("w", encoding="utf-8", newline="") as target:
                writer = csv.writer(target)
                writer.writerow(["input_ids", "labels"])
                writer.writerow(["文本", "0"])

            with self.assertRaises(ValueError) as context:
                read_csv(path, "text", "label")

        self.assertIn("text", str(context.exception))

    def test_read_csv_rejects_unknown_label(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad_label.csv"
            with path.open("w", encoding="utf-8", newline="") as target:
                writer = csv.writer(target)
                writer.writerow(["text", "label"])
                writer.writerow(["正常文本", "review"])

            with self.assertRaises(ValueError):
                read_csv(path, "text", "label")

    def test_cross_source_merge_dedups_and_drops_conflicts(self) -> None:
        first = [("共享样本", 1), ("正常样本", 0), ("标签冲突", 0)]
        second = [("共享样本", 1), ("冲突样本", 0), ("标签冲突", 1)]

        samples, conflicts = resolve_conflicts(first + second)

        self.assertIn("共享样本", samples)
        self.assertEqual(samples["共享样本"], 1)
        self.assertEqual(samples["正常样本"], 0)
        self.assertIn("标签冲突", conflicts)
        self.assertNotIn("标签冲突", samples)
        self.assertIn("冲突样本", samples)


class HarmMappingTests(unittest.TestCase):
    def test_abuse_category_is_tracked_separately_from_spam(self) -> None:
        self.assertEqual(HARM_ABUSE_LABEL, "谩骂引战")
        self.assertNotIn(HARM_ABUSE_LABEL, HARM_LABEL_MAP)

    def test_marketing_categories_map_to_spam(self) -> None:
        for category in ("低俗色情", "博彩", "欺诈", "黑产广告"):
            self.assertEqual(HARM_LABEL_MAP[category], 1)
        self.assertEqual(HARM_LABEL_MAP["不违规"], 0)

    def test_rows_from_harm_splits_abuse_from_marketing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "低俗色情.json"
            path.write_text(
                json.dumps(
                    [
                        {"文本": "加微信看片", "标签": "低俗色情"},
                        {"文本": "你妈就是狗", "标签": "谩骂引战"},
                        {"文本": "未知类别", "标签": "新增类别"},
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            spam_rows, abuse_rows = rows_from_harm(Path(directory))

        self.assertEqual(spam_rows, [("加微信看片", 1)])
        self.assertEqual(abuse_rows, [("你妈就是狗", 1)])


class CollectTests(unittest.TestCase):
    def test_collect_dedups_and_prefers_spam_on_label_conflict(self) -> None:
        samples = collect(
            [
                ("加微信", 0),
                ("加微信", 1),
                ("", 1),
                ("   ", 0),
                ("正常聊天", 0),
            ]
        )
        collected = dict(samples)

        self.assertEqual(collected["加微信"], 1)
        self.assertEqual(collected["正常聊天"], 0)
        self.assertEqual(len(samples), 2)

    def test_collect_dedups_text_differing_only_by_normalization(self) -> None:
        samples = collect([("加微信", 1), ("加微信  ", 1), ("﻿加微信", 1), ("无抵押秒到账", 1)])

        self.assertEqual(len(samples), 2)
        self.assertEqual(dict(samples)["加微信"], 1)


if __name__ == "__main__":
    unittest.main()
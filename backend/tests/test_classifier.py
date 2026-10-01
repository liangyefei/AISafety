import unittest
from pathlib import Path

from app.classifier import (
    ABUSE_REVIEW_THRESHOLD,
    ABUSE_RULES,
    SLUR_PATH,
    SLUR_RULE_WEIGHT,
    SpamClassifier,
    classify_text,
    load_slurs,
    normalize_text,
    score_abuse_rules,
)


class FixedProbabilityModel:
    def __init__(self, spam_probability: float) -> None:
        self.spam_probability = spam_probability

    def predict_proba(self, texts: list[str]) -> list[list[float]]:
        return [[1.0 - self.spam_probability, self.spam_probability] for _ in texts]

class ClassifierTests(unittest.TestCase):
    def test_normal_message_is_allowed(self) -> None:
        result = classify_text("会议改到下午三点，请带上最新的项目材料。")
        self.assertEqual(result["label"], "normal")
        self.assertEqual(result["matched_rules"], [])

    def test_multiple_signals_are_blocked(self) -> None:
        result = classify_text("限时低息借款，无抵押秒到账，咨询13800138000")
        self.assertEqual(result["label"], "spam")
        self.assertGreaterEqual(result["score"], 0.72)

    def test_single_weak_signal_goes_to_review(self) -> None:
        result = classify_text("加我微信：service_888")
        self.assertEqual(result["label"], "review")

    def test_obfuscation_is_normalized(self) -> None:
        self.assertEqual(normalize_text("Ｖ Ｘ：ABCDE888\u200b"), "v x:abcde888")

    def test_repeated_characters_are_collapsed(self) -> None:
        self.assertEqual(normalize_text("限时时时时时"), "限时时")

    def test_numeric_glyph_variants_are_canonicalized(self) -> None:
        self.assertEqual(normalize_text("①②③ ⑴⑵⑶ 𝟙𝟚𝟛"), "123 123 123")

    def test_chinese_digit_sequence_is_canonicalized(self) -> None:
        self.assertEqual(normalize_text("①③⑧〇〇①③⑧〇〇〇"), "13800138000")

    def test_english_digit_sequence_is_canonicalized(self) -> None:
        self.assertEqual(normalize_text("one2three"), "123")
        self.assertEqual(normalize_text("one three eight-00138000"), "13800138000")

    def test_english_words_containing_number_names_are_unchanged(self) -> None:
        self.assertEqual(normalize_text("someone has a phone"), "someone has a phone")

    def test_obfuscated_phone_with_contact_context_is_blocked(self) -> None:
        result = classify_text("加我①③⑧〇〇①③⑧〇〇〇")
        self.assertEqual(result["label"], "review")
        self.assertIn("contact-phone", result["matched_rules"])

    def test_numeric_variants_after_customer_service_are_reviewed(self) -> None:
        result = classify_text("客服①②③　⑴⑵⑶　𝟙𝟚𝟛　one2three")
        self.assertEqual(result["label"], "review")
        self.assertIn("contact-numeric-id", result["matched_rules"])

    def test_adult_lure_lexicon_includes_pure_love_and_novel(self) -> None:
        for text in ("纯爱视频，忝加薇看片", "小说，全本免费，罒叩获取"):
            result = classify_text(text)
            self.assertEqual(result["label"], "review")
            self.assertIn("adult-lure", result["matched_rules"])

    def test_model_only_high_probability_is_blocked(self) -> None:
        classifier = SpamClassifier()
        classifier.model = FixedProbabilityModel(0.9)
        classifier.model_path = Path("models/test-model.joblib")

        result = classifier.predict("未命中任何规则的模型垃圾样本")

        self.assertEqual(result["label"], "spam")
        self.assertEqual(result["score"], 0.9)
        self.assertEqual(result["model_score"], 0.9)
        self.assertEqual(result["rule_score"], 0.0)

    def test_model_only_medium_probability_is_reviewed(self) -> None:
        classifier = SpamClassifier()
        classifier.model = FixedProbabilityModel(0.6)
        classifier.model_path = Path("models/test-model.joblib")

        result = classifier.predict("模型分数位于灰区的样本")

        self.assertEqual(result["label"], "review")


class SlurRuleTests(unittest.TestCase):
    """俚语规则来自 STATE-ToxiCN 词表，覆盖模型难以学到的隐式歧视表达。"""

    def test_slur_rule_is_loaded(self) -> None:
        slurs = load_slurs()
        self.assertTrue(slurs, "俚语词表未载入，隐式歧视规则失效")
        self.assertTrue(any(rule.rule_id == "abuse-slur" for rule in ABUSE_RULES))

    def test_implicit_slur_is_scored(self) -> None:
        """单条俚语命中应产生可观测分数；叠加显式辱骂词才足以判为辱骂。"""
        score, matches = score_abuse_rules("黑鬼智商确实不行")
        self.assertGreaterEqual(score, SLUR_RULE_WEIGHT)
        self.assertIn("abuse-slur", [match.rule_id for match in matches])

        combined, _ = score_abuse_rules("这女虫脑瘫了吧")
        self.assertGreaterEqual(combined, ABUSE_REVIEW_THRESHOLD)

    def test_single_slur_alone_stays_below_review(self) -> None:
        """词表含大量中性搭配，单条命中不足以触发复核，模型分仍可做二次判断。"""
        score, _ = score_abuse_rules("小黑盒")
        self.assertLess(score, ABUSE_REVIEW_THRESHOLD)

    def test_advocacy_overrides_slur_match(self) -> None:
        """「关爱孤儿，不要网暴他」会命中俚语词表，但属正当表达应归零。"""
        score, matches = score_abuse_rules("关爱孤儿，不要网暴他")
        self.assertEqual(score, 0.0)
        self.assertIn("abuse-advocacy", [match.rule_id for match in matches])

    def test_slur_weight_stays_low_to_limit_false_positives(self) -> None:
        """词表含中性搭配（如「小黑盒」），权重过高会把正常讨论推入复核。"""
        self.assertLessEqual(SLUR_RULE_WEIGHT, 0.35)

    def test_missing_slur_file_is_tolerated(self) -> None:
        import tempfile
        from pathlib import Path as _Path

        from app import classifier as module

        original = module.SLUR_PATH
        with tempfile.TemporaryDirectory() as directory:
            try:
                module.SLUR_PATH = _Path(directory) / "missing.txt"
                self.assertEqual(module.load_slurs(), ())
            finally:
                module.SLUR_PATH = original


    def test_moderate_abuse_score_is_flagged_suspicious(self) -> None:
        """灰区（0.4-0.6）应标记为可疑，便于人工复核而非直接放行。"""
        classifier = SpamClassifier()
        classifier.abuse_model = FixedProbabilityModel(0.5)
        result = classifier.predict("某种委婉的歧视性表达")
        self.assertEqual(result["abuse_label"], "review")
        self.assertTrue(result["abuse_suspicious"])

    def test_high_abuse_score_is_not_marked_suspicious(self) -> None:
        classifier = SpamClassifier()
        classifier.abuse_model = FixedProbabilityModel(0.8)
        result = classifier.predict("明显的辱骂")
        self.assertEqual(result["abuse_label"], "spam")
        self.assertFalse(result["abuse_suspicious"])

    def test_clean_text_is_not_suspicious(self) -> None:
        classifier = SpamClassifier()
        result = classifier.predict("会议改到下午三点")
        self.assertFalse(result["abuse_suspicious"])


class RuleThresholdTests(unittest.TestCase):
    """RULE_SPAM_THRESHOLD 必须真实生效，不能只出现在纯规则路径。"""

    def setUp(self) -> None:
        model_path = Path(__file__).resolve().parents[1] / "models" / "spam_pipeline_pinyin.joblib"
        if not model_path.exists():
            self.skipTest("模型文件不存在")
        self.classifier = SpamClassifier(model_path)
        self.classifier.model_path = model_path

    def test_rule_threshold_can_block_independently(self) -> None:
        from app import classifier as module

        text = "博彩送彩金立即下注"
        rule_score = module.score_text(text)[0]
        self.assertGreaterEqual(rule_score, 0.7)

        original = (module.RULE_SPAM_THRESHOLD, module.MODEL_SPAM_THRESHOLD)
        try:
            module.RULE_SPAM_THRESHOLD, module.MODEL_SPAM_THRESHOLD = 0.7, 0.99
            self.assertEqual(self.classifier.predict(text)["label"], "spam")

            module.RULE_SPAM_THRESHOLD, module.MODEL_SPAM_THRESHOLD = 0.99, 0.99
            self.assertNotEqual(self.classifier.predict(text)["label"], "spam")
        finally:
            module.RULE_SPAM_THRESHOLD, module.MODEL_SPAM_THRESHOLD = original


class AbuseRuleTests(unittest.TestCase):
    def test_insult_is_scored_as_abuse(self) -> None:
        for text in ("你这种废物也配说话", "傻逼东西滚远点", "你妈就是狗"):
            score, matches = score_abuse_rules(text)
            self.assertGreaterEqual(score, ABUSE_REVIEW_THRESHOLD, text)
            self.assertTrue(matches, text)

    def test_obfuscated_insult_is_detected(self) -> None:
        """语料中大量使用谐音与拼音缩写规避，规则需能覆盖。"""
        for text in ("你个cs", "sz东西我知道你是cs", "我来自你🐴", "你没父母女"):
            score, _ = score_abuse_rules(text)
            self.assertGreaterEqual(score, ABUSE_REVIEW_THRESHOLD, text)

    def test_neutral_text_is_not_abuse(self) -> None:
        for text in ("会议改到下午三点", "今天天气不错", "这个bug已经修复了"):
            score, _ = score_abuse_rules(text)
            self.assertEqual(score, 0.0, text)

    def test_anti_harassment_expression_reduces_abuse_score(self) -> None:
        """反网暴属于正当表达，不应因含脏字被判为辱骂。"""
        plain, _ = score_abuse_rules("这个人真是傻逼")
        advocacy, _ = score_abuse_rules("这个人真是傻逼，我要举报他")
        self.assertGreater(plain, advocacy)
        self.assertEqual(score_abuse_rules("关爱孤儿，不要网暴他")[0], 0.0)

    def test_marketing_text_is_not_claimed_as_abuse(self) -> None:
        score, _ = score_abuse_rules("无抵押秒到账，咨询13800138000")
        self.assertEqual(score, 0.0)

    def test_abuse_result_appears_without_model(self) -> None:
        result = classify_text("你这种废物也配说话")
        self.assertEqual(result["abuse_label"], "normal")
        self.assertEqual(result["abuse_score"], 0.0)

    def test_abuse_rules_apply_when_no_models_loaded(self) -> None:
        classifier = SpamClassifier()
        result = classifier.predict("你这种废物也配说话，滚远点")
        self.assertNotEqual(result["abuse_label"], "normal")
        self.assertIn("辱骂攻击", result["categories"])

    def test_abuse_model_score_drives_label_when_loaded(self) -> None:
        classifier = SpamClassifier()
        classifier.abuse_model = FixedProbabilityModel(0.93)
        result = classifier.predict("完全没有任何脏字的普通句子")
        self.assertEqual(result["abuse_label"], "spam")
        self.assertEqual(result["abuse_score"], 0.93)
        self.assertIn("模型识别出辱骂或攻击性表达", result["abuse_reasons"])

    def test_abuse_signal_does_not_change_spam_label(self) -> None:
        """辱骂维度与营销维度正交：高辱骂不应被算作营销垃圾。"""
        classifier = SpamClassifier()
        result = classifier.predict("你这种废物也配说话，滚远点")
        self.assertEqual(result["label"], "normal")
        self.assertNotEqual(result["abuse_label"], "normal")


if __name__ == "__main__":
    unittest.main()
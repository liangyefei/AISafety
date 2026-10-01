import unittest

from app.features import phonetic_text


class PhoneticFeatureTests(unittest.TestCase):
    def test_homophones_share_the_same_feature(self) -> None:
        self.assertEqual(phonetic_text("微信"), "weixin")
        self.assertEqual(phonetic_text("薇信"), "weixin")
        self.assertEqual(phonetic_text("维信"), "weixin")

    def test_romanized_text_ignores_separators_and_case(self) -> None:
        self.assertEqual(phonetic_text("Wei Xin"), "weixin")
        self.assertEqual(phonetic_text("WEI-XIN"), "weixin")

    def test_full_width_and_mixed_text_are_canonicalized(self) -> None:
        self.assertEqual(phonetic_text("加ＶＸ888"), "jiavx888")


if __name__ == "__main__":
    unittest.main()
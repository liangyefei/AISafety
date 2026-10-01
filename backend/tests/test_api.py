import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from app import main
from app.storage import ResultStore


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        main.store = ResultStore(Path(self.temp_directory.name) / "test.db")
        self.client = TestClient(main.app)

    def tearDown(self) -> None:
        self.client.close()
        self.temp_directory.cleanup()

    def test_health_reports_rule_fallback(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_detection_and_stats(self) -> None:
        response = self.client.post("/api/filter", json={"text": "博彩送彩金，立即下注稳赚不赔"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["label"], "spam")
        self.assertNotIn("text", self.client.get("/api/stats").json())
        self.assertEqual(self.client.get("/api/stats").json()["total"], 1)

    def test_empty_text_is_rejected(self) -> None:
        response = self.client.post("/api/filter", json={"text": ""})
        self.assertEqual(response.status_code, 422)

    def test_feedback_is_accepted(self) -> None:
        response = self.client.post(
            "/api/feedback",
            json={
                "text": "这是误判样本",
                "predicted_label": "review",
                "corrected_label": "normal",
                "note": "内部通知",
            },
        )
        self.assertEqual(response.status_code, 201)
        self.assertEqual(self.client.get("/api/stats").json()["feedback"], 1)

    def test_abuse_detection_is_recorded(self) -> None:
        response = self.client.post("/api/filter", json={"text": "你这种废物也配说话，滚远点"})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("abuse_label", payload)
        self.assertIn("abuse_score", payload)
        self.assertEqual(self.client.get("/api/stats").json()["abuse"], 1)

    def test_abuse_feedback_label_is_stored(self) -> None:
        response = self.client.post(
            "/api/feedback",
            json={
                "text": "辱骂样本",
                "predicted_label": "normal",
                "corrected_label": "normal",
                "corrected_abuse_label": "abuse",
            },
        )
        self.assertEqual(response.status_code, 201)
        self.assertEqual(self.client.get("/api/stats").json()["feedback"], 1)

    def test_health_reports_abuse_model_state(self) -> None:
        payload = self.client.get("/health").json()
        self.assertIn("abuse_model_loaded", payload)


if __name__ == "__main__":
    unittest.main()
"""
End-to-end unit and integration tests for API and ML engine.
Run with: python3 -m unittest tests/test_api_and_ml.py
"""
import io
import unittest
from fastapi.testclient import TestClient
from api.index import app

SAMPLE_URL = "https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv"

class TestApiAndMLEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_01_health_check(self):
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get("status"), "ok")
        self.assertEqual(data.get("engine"), "scikit-learn")

    def test_02_preview_valid_url(self):
        resp = self.client.post("/api/dataset/preview", json={"url": SAMPLE_URL})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["row_count"], 50)
        self.assertIn("Profit", data["candidate_targets"])
        self.assertEqual(data["default_target"], "Profit")
        self.assertEqual(len(data["preview_rows"]), 5)

    def test_03_ssrf_protection_private_ip(self):
        # 127.0.0.1
        resp = self.client.post("/api/dataset/preview", json={"url": "https://127.0.0.1/data.csv"})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("安全阻擋", resp.json()["detail"])

        # http:// (non-https)
        resp2 = self.client.post("/api/dataset/preview", json={"url": "http://example.com/data.csv"})
        self.assertEqual(resp2.status_code, 400)
        self.assertIn("HTTPS", resp2.json()["detail"])

    def test_04_train_startups_url(self):
        payload = {
            "url": SAMPLE_URL,
            "target": "Profit",
            "folds": 3,
            "quick": True
        }
        resp = self.client.post("/api/train", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["target"], "Profit")
        self.assertGreater(data["winner_test_r2"], 0.70)
        self.assertGreater(len(data["winner_selected_features"]), 0)
        self.assertEqual(len(data["predictions"]), 10)
        self.assertEqual(len(data["qq_plot"]), 10)
        self.assertIn("durbin_watson", data["diagnostics"])
        self.assertIn("cv_summary", data)
        self.assertIn("test_summary", data)

    def test_05_train_file_upload(self):
        csv_content = (
            "feature_x,feature_z,y_target\n" +
            "\n".join([f"{i * 1.5},{i * 0.8},{i * 3.2 + 5.0}" for i in range(25)])
        )
        file_obj = io.BytesIO(csv_content.encode("utf-8"))
        files = {"file": ("toy_data.csv", file_obj, "text/csv")}
        data = {"target": "y_target", "folds": "3", "quick": "true"}

        resp = self.client.post("/api/train", data=data, files=files)
        self.assertEqual(resp.status_code, 200)
        res = resp.json()
        self.assertEqual(res["target"], "y_target")
        self.assertGreater(res["winner_test_r2"], 0.95)

    def test_06_train_invalid_target_error(self):
        resp = self.client.post("/api/train", json={
            "url": SAMPLE_URL,
            "target": "NonExistentColumn",
            "folds": 3
        })
        self.assertEqual(resp.status_code, 400)
        self.assertIn("未找到指定的目標欄位", resp.json()["detail"])

    def test_07_train_insufficient_rows_error(self):
        csv_content = "x,y\n1,2\n3,4\n5,6\n"
        resp = self.client.post("/api/train", json={
            "csv_text": csv_content,
            "target": "y",
            "folds": 3
        })
        self.assertEqual(resp.status_code, 400)
        self.assertIn("列數", resp.json()["detail"])

if __name__ == "__main__":
    unittest.main()

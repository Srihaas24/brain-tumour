import os
import sys
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from fastapi.testclient import TestClient
from backend.app.main import app
import numpy as np
import cv2

class NeuroVisionAPITestCase(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_check(self):
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("models_configured", data)

    def test_list_models(self):
        res = self.client.get("/api/models")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("models", data)
        self.assertGreaterEqual(len(data["models"]), 2)

    def test_demo_scan(self):
        res = self.client.get("/api/demo/glioma")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["sample_id"], "glioma")
        self.assertTrue(data["image_base64"].startswith("data:image/"))

    def test_analyze_pipeline(self):
        sample_path = os.path.join(BASE_DIR, 'app', 'static', 'samples', 'sample_glioma.jpg')
        if not os.path.exists(sample_path):
            self.skipTest("Sample file missing")

        with open(sample_path, 'rb') as f:
            files = {'file': ('sample_glioma.jpg', f, 'image/jpeg')}
            data = {'model_choice': 'nnunet-2d', 'mc_dropout_passes': 4}
            res = self.client.post("/api/analyze", files=files, data=data)

        self.assertEqual(res.status_code, 200)
        resp = res.json()
        self.assertIn("result_id", resp)
        self.assertIn("classification", resp)
        self.assertIn("segmentation", resp)
        self.assertIn("uncertainty", resp)
        self.assertIn("quality", resp)
        self.assertIn("mesh_3d", resp)
        self.assertIn("human_review", resp)
        self.assertTrue(resp["mesh_3d"]["has_mesh"])

    def test_benchmark_endpoint(self):
        res = self.client.post("/api/benchmark")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("models", data)
        self.assertIn("speedup", data)

if __name__ == "__main__":
    unittest.main()

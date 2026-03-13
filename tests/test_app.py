"""
Unit and integration tests for the Plant Disease Detector.

Run with:
    pytest tests/ -v
"""

import io
import importlib
import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image


# ── helpers ─────────────────────────────────────────────────────────────────

def _make_green_leaf_image(width: int = 224, height: int = 224) -> Image.Image:
    """Create a simple solid-green PIL image that mimics a leaf scan."""
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[:, :, 1] = 160  # green channel
    return Image.fromarray(arr, mode="RGB")


def _image_to_bytes(image: Image.Image, fmt: str = "JPEG") -> bytes:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    buf.seek(0)
    return buf.read()


# ── predictor module tests ───────────────────────────────────────────────────

class TestClassLabels:
    def test_exactly_38_classes(self):
        from model.predictor import CLASS_LABELS
        assert len(CLASS_LABELS) == 38

    def test_each_label_has_description(self):
        from model.predictor import CLASS_LABELS, DISEASE_INFO
        for label in CLASS_LABELS:
            assert label in DISEASE_INFO, f"Missing description for: {label}"

    def test_no_duplicate_labels(self):
        from model.predictor import CLASS_LABELS
        assert len(CLASS_LABELS) == len(set(CLASS_LABELS))


class TestPreprocessImage:
    def test_output_shape(self):
        from model.predictor import preprocess_image
        img = _make_green_leaf_image(300, 400)
        result = preprocess_image(img)
        assert result.shape == (1, 224, 224, 3)

    def test_values_in_mobilenet_range(self):
        """MobileNetV2 preprocessing maps pixel values to [-1, 1]."""
        from model.predictor import preprocess_image
        img = _make_green_leaf_image()
        arr = preprocess_image(img)
        assert arr.min() >= -1.0 - 1e-6
        assert arr.max() <= 1.0 + 1e-6

    def test_accepts_rgba_image(self):
        """RGBA images (e.g. PNG with transparency) should be converted to RGB."""
        from model.predictor import preprocess_image
        rgba = Image.new("RGBA", (100, 100), (0, 200, 0, 128))
        result = preprocess_image(rgba)
        assert result.shape == (1, 224, 224, 3)


class TestPredict:
    """Test the predict() function using a mocked Keras model."""

    def _mock_predict(self, num_classes: int = 38, best_class: int = 37):
        """Return a mock that produces a one-hot-ish softmax output."""
        preds = np.full((1, num_classes), 0.01 / (num_classes - 1))
        preds[0, best_class] = 0.99
        mock_model = MagicMock()
        mock_model.predict.return_value = preds
        return mock_model

    def test_predict_returns_required_keys(self):
        from model import predictor
        with patch.object(predictor, "load_model", return_value=self._mock_predict()):
            result = predictor.predict(_make_green_leaf_image())
        assert {"label", "confidence", "description", "top5"} <= result.keys()

    def test_confidence_between_0_and_100(self):
        from model import predictor
        with patch.object(predictor, "load_model", return_value=self._mock_predict()):
            result = predictor.predict(_make_green_leaf_image())
        assert 0 <= result["confidence"] <= 100

    def test_top5_has_five_entries(self):
        from model import predictor
        with patch.object(predictor, "load_model", return_value=self._mock_predict()):
            result = predictor.predict(_make_green_leaf_image())
        assert len(result["top5"]) == 5

    def test_best_label_is_top5_first(self):
        from model import predictor
        with patch.object(predictor, "load_model", return_value=self._mock_predict()):
            result = predictor.predict(_make_green_leaf_image())
        assert result["label"] == result["top5"][0]["label"]

    def test_description_is_non_empty_string(self):
        from model import predictor
        with patch.object(predictor, "load_model", return_value=self._mock_predict()):
            result = predictor.predict(_make_green_leaf_image())
        assert isinstance(result["description"], str)
        assert len(result["description"]) > 0


# ── Flask app tests ───────────────────────────────────────────────────────────

@pytest.fixture()
def client():
    """Flask test client with mocked predict()."""
    import app as flask_app
    flask_app.app.config["TESTING"] = True
    flask_app.app.config["WTF_CSRF_ENABLED"] = False
    with flask_app.app.test_client() as c:
        yield c


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_json(self, client):
        resp = client.get("/health")
        data = resp.get_json()
        assert data["status"] == "ok"


class TestIndexPage:
    def test_get_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_upload_form_present(self, client):
        resp = client.get("/")
        assert b"upload-form" in resp.data or b"file" in resp.data


class TestPredictEndpoint:
    def _fake_result(self):
        return {
            "label": "Tomato – Healthy",
            "confidence": 95.0,
            "description": "No disease detected.",
            "top5": [
                {"label": "Tomato – Healthy", "confidence": 95.0},
                {"label": "Tomato – Early Blight", "confidence": 2.5},
                {"label": "Tomato – Late Blight", "confidence": 1.0},
                {"label": "Tomato – Bacterial Spot", "confidence": 0.8},
                {"label": "Tomato – Leaf Mold", "confidence": 0.7},
            ],
        }

    def test_no_file_returns_error(self, client):
        resp = client.post("/predict", data={})
        assert resp.status_code == 200
        assert b"No file" in resp.data

    def test_empty_filename_returns_error(self, client):
        resp = client.post(
            "/predict",
            data={"file": (io.BytesIO(b""), "")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        assert b"No file selected" in resp.data

    def test_unsupported_extension_returns_error(self, client):
        resp = client.post(
            "/predict",
            data={"file": (io.BytesIO(b"data"), "leaf.bmp")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        assert b"Unsupported" in resp.data

    def test_valid_image_returns_result(self, client):
        import app as flask_app
        img_bytes = _image_to_bytes(_make_green_leaf_image())
        with patch.object(flask_app, "predict", return_value=self._fake_result()):
            resp = client.post(
                "/predict",
                data={"file": (io.BytesIO(img_bytes), "leaf.jpg")},
                content_type="multipart/form-data",
            )
        assert resp.status_code == 200
        assert b"Tomato" in resp.data

    def test_model_error_returns_error_page(self, client):
        import app as flask_app
        img_bytes = _image_to_bytes(_make_green_leaf_image())
        with patch.object(flask_app, "predict", side_effect=RuntimeError("model crash")):
            resp = client.post(
                "/predict",
                data={"file": (io.BytesIO(img_bytes), "leaf.jpg")},
                content_type="multipart/form-data",
            )
        assert resp.status_code == 200
        assert b"Prediction failed" in resp.data

    def test_png_file_accepted(self, client):
        import app as flask_app
        img_bytes = _image_to_bytes(_make_green_leaf_image(), fmt="PNG")
        with patch.object(flask_app, "predict", return_value=self._fake_result()):
            resp = client.post(
                "/predict",
                data={"file": (io.BytesIO(img_bytes), "leaf.png")},
                content_type="multipart/form-data",
            )
        assert resp.status_code == 200
        assert b"Tomato" in resp.data

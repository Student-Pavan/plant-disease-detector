"""
Flask web application for the Plant Disease Detector.

Run locally:
    python app.py

Or with a WSGI server:
    gunicorn app:app
"""

import io
import os

from flask import Flask, redirect, render_template, request, url_for
from PIL import Image
from werkzeug.utils import secure_filename

from model.predictor import predict

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload limit
app.secret_key = os.environ.get("SECRET_KEY", "change-me-in-production")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}


def _allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


@app.route("/", methods=["GET"])
def index():
    """Landing page with the upload form."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_disease():
    """Accept an uploaded leaf image and return disease predictions."""
    if "file" not in request.files:
        return render_template("index.html", error="No file part in the request.")

    file = request.files["file"]

    if file.filename == "":
        return render_template("index.html", error="No file selected.")

    if not _allowed_file(file.filename):
        return render_template(
            "index.html",
            error="Unsupported file type. Please upload a PNG, JPG, JPEG, or WEBP image.",
        )

    try:
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes))
        result = predict(image)
    except Exception as exc:  # noqa: BLE001
        return render_template("index.html", error=f"Prediction failed: {exc}")

    return render_template("result.html", result=result, filename=secure_filename(file.filename))


@app.route("/health")
def health():
    """Simple health-check endpoint."""
    return {"status": "ok"}, 200


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)

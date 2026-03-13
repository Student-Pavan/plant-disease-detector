"""
Plant disease predictor using MobileNetV2 transfer learning.

The model is trained on the PlantVillage dataset and can classify
38 plant disease / healthy categories.
"""

import os
import numpy as np
from PIL import Image

# ── disease class labels (PlantVillage, 38 classes) ─────────────────────────
CLASS_LABELS = [
    "Apple – Apple Scab",
    "Apple – Black Rot",
    "Apple – Cedar Apple Rust",
    "Apple – Healthy",
    "Blueberry – Healthy",
    "Cherry – Powdery Mildew",
    "Cherry – Healthy",
    "Corn – Cercospora / Gray Leaf Spot",
    "Corn – Common Rust",
    "Corn – Northern Leaf Blight",
    "Corn – Healthy",
    "Grape – Black Rot",
    "Grape – Esca (Black Measles)",
    "Grape – Leaf Blight (Isariopsis)",
    "Grape – Healthy",
    "Orange – Huanglongbing (Citrus Greening)",
    "Peach – Bacterial Spot",
    "Peach – Healthy",
    "Pepper – Bacterial Spot",
    "Pepper – Healthy",
    "Potato – Early Blight",
    "Potato – Late Blight",
    "Potato – Healthy",
    "Raspberry – Healthy",
    "Soybean – Healthy",
    "Squash – Powdery Mildew",
    "Strawberry – Leaf Scorch",
    "Strawberry – Healthy",
    "Tomato – Bacterial Spot",
    "Tomato – Early Blight",
    "Tomato – Late Blight",
    "Tomato – Leaf Mold",
    "Tomato – Septoria Leaf Spot",
    "Tomato – Spider Mites",
    "Tomato – Target Spot",
    "Tomato – Yellow Leaf Curl Virus",
    "Tomato – Mosaic Virus",
    "Tomato – Healthy",
]

# Map each label to a brief description shown in the UI
DISEASE_INFO = {
    "Apple – Apple Scab": "Fungal disease causing dark, scabby lesions on leaves and fruit.",
    "Apple – Black Rot": "Fungal infection causing brown rot on fruit and leaf spots.",
    "Apple – Cedar Apple Rust": "Fungal rust disease with orange spots on leaves.",
    "Apple – Healthy": "No disease detected. The plant appears healthy.",
    "Blueberry – Healthy": "No disease detected. The plant appears healthy.",
    "Cherry – Powdery Mildew": "White powdery fungal growth on leaf surfaces.",
    "Cherry – Healthy": "No disease detected. The plant appears healthy.",
    "Corn – Cercospora / Gray Leaf Spot": "Fungal disease with rectangular gray/brown lesions.",
    "Corn – Common Rust": "Small, powdery orange-brown pustules on both leaf surfaces.",
    "Corn – Northern Leaf Blight": "Cigar-shaped grayish-green lesions on leaves.",
    "Corn – Healthy": "No disease detected. The plant appears healthy.",
    "Grape – Black Rot": "Fungal disease causing brown lesions and shriveled fruit.",
    "Grape – Esca (Black Measles)": "Complex fungal disease with interveinal chlorosis.",
    "Grape – Leaf Blight (Isariopsis)": "Angular brown lesions on older leaves.",
    "Grape – Healthy": "No disease detected. The plant appears healthy.",
    "Orange – Huanglongbing (Citrus Greening)": "Bacterial disease causing blotchy yellowing of leaves.",
    "Peach – Bacterial Spot": "Bacterial infection causing water-soaked lesions on leaves.",
    "Peach – Healthy": "No disease detected. The plant appears healthy.",
    "Pepper – Bacterial Spot": "Bacterial infection with raised, dark brown lesions.",
    "Pepper – Healthy": "No disease detected. The plant appears healthy.",
    "Potato – Early Blight": "Fungal disease with dark concentric rings on older leaves.",
    "Potato – Late Blight": "Water mold causing dark, water-soaked lesions.",
    "Potato – Healthy": "No disease detected. The plant appears healthy.",
    "Raspberry – Healthy": "No disease detected. The plant appears healthy.",
    "Soybean – Healthy": "No disease detected. The plant appears healthy.",
    "Squash – Powdery Mildew": "White powdery fungal coating on leaf surfaces.",
    "Strawberry – Leaf Scorch": "Fungal disease causing purple to dark brown lesions.",
    "Strawberry – Healthy": "No disease detected. The plant appears healthy.",
    "Tomato – Bacterial Spot": "Bacterial infection with small, dark water-soaked lesions.",
    "Tomato – Early Blight": "Fungal disease with concentric ring lesions on lower leaves.",
    "Tomato – Late Blight": "Water mold causing greasy dark lesions on leaves and fruit.",
    "Tomato – Leaf Mold": "Fungal disease with yellow patches and olive-green mold.",
    "Tomato – Septoria Leaf Spot": "Circular spots with dark borders and grey centres.",
    "Tomato – Spider Mites": "Tiny mites causing stippled, yellowing leaves.",
    "Tomato – Target Spot": "Fungal disease with concentric ring lesions on leaves.",
    "Tomato – Yellow Leaf Curl Virus": "Viral disease causing leaf curling and yellowing.",
    "Tomato – Mosaic Virus": "Viral disease with mosaic pattern of light and dark green.",
    "Tomato – Healthy": "No disease detected. The plant appears healthy.",
}

IMG_SIZE = (224, 224)

# Path where fine-tuned weights are expected to be placed by the user.
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "plant_disease_model.keras")


def _build_model():
    """Build the MobileNetV2-based classification model."""
    import tensorflow as tf

    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    model = tf.keras.Sequential(
        [
            base,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(len(CLASS_LABELS), activation="softmax"),
        ],
        name="plant_disease_detector",
    )
    return model


_model = None  # lazy singleton


def load_model():
    """Return the (cached) Keras model, loading fine-tuned weights when available."""
    global _model
    if _model is None:
        _model = _build_model()
        if os.path.exists(WEIGHTS_PATH):
            _model.load_weights(WEIGHTS_PATH)
    return _model


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize and normalise a PIL image for MobileNetV2 inference."""
    import tensorflow as tf

    img = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def predict(image: Image.Image) -> dict:
    """
    Run inference on *image* and return a prediction dictionary:

        {
            "label":       str,   # human-readable disease label
            "confidence":  float, # 0–100 %
            "description": str,   # brief disease description
            "top5":        list[dict],  # top-5 predictions [{label, confidence}]
        }
    """
    model = load_model()
    processed = preprocess_image(image)
    preds = model.predict(processed, verbose=0)[0]

    top5_idx = np.argsort(preds)[::-1][:5]
    top5 = [
        {"label": CLASS_LABELS[i], "confidence": round(float(preds[i]) * 100, 2)}
        for i in top5_idx
    ]

    best_idx = int(np.argmax(preds))
    label = CLASS_LABELS[best_idx]
    return {
        "label": label,
        "confidence": round(float(preds[best_idx]) * 100, 2),
        "description": DISEASE_INFO.get(label, ""),
        "top5": top5,
    }

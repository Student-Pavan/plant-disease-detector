# 🌿 Plant Disease Detector

An AI-powered web application that diagnoses plant diseases from leaf photographs.
Upload a photo and receive an instant prediction with confidence scores across
**38 disease / healthy categories** covering 14 popular crop types.

---

## Features

- **MobileNetV2** transfer-learning backbone fine-tunable on the PlantVillage dataset
- **38 disease classes** – Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato
- Drag-and-drop or click-to-browse image upload
- Confidence bar + top-5 predictions displayed in the browser
- Simple REST `/predict` endpoint – easy to extend or call from other services
- `/health` endpoint for uptime checks

---

## Requirements

| Dependency   | Version  |
|--------------|----------|
| Python       | ≥ 3.9    |
| Flask        | ≥ 2.3    |
| TensorFlow   | ≥ 2.13   |
| Pillow       | ≥ 10.0   |
| NumPy        | ≥ 1.24   |

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/Student-Pavan/plant-disease-detector.git
cd plant-disease-detector

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Place fine-tuned weights
#    Download or train plant_disease_model.keras and put it in the model/ folder.
#    Without weights the model uses ImageNet pre-training (predictions will be random).

# 5. Run the development server
python app.py
```

Open <http://localhost:5000> in your browser.

---

## Project Structure

```
plant-disease-detector/
├── app.py                      # Flask application
├── requirements.txt
├── model/
│   ├── __init__.py
│   └── predictor.py            # Model loading, pre-processing, inference
├── static/
│   ├── css/style.css
│   └── js/main.js
├── templates/
│   ├── index.html              # Upload page
│   └── result.html             # Prediction results page
└── tests/
    └── test_app.py             # Pytest test suite
```

---

## Using a Pre-trained Model

To get accurate predictions you need weights trained on the
[PlantVillage dataset](https://github.com/spMohanty/PlantVillage-Dataset).

1. Train the model (see the `model/predictor.py` for the architecture).
2. Save the weights:
   ```python
   model.save("model/plant_disease_model.keras")
   ```
3. The app will automatically load them on the next startup.

---

## Running Tests

```bash
pytest tests/ -v
```

---

## API

### `POST /predict`

Accepts a `multipart/form-data` POST with a `file` field containing the leaf image.
Supported formats: `png`, `jpg`, `jpeg`, `webp`. Max size: 10 MB.

Returns an HTML page with the prediction. To consume the prediction
programmatically, extend the route to return JSON instead of a template.

### `GET /health`

Returns `{"status": "ok"}` with HTTP 200.

---

## License

MIT
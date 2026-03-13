# 🌿 Plant Disease Detection using Deep Learning

A web-based AI application that detects plant leaf diseases using a deep learning model trained on the PlantVillage dataset.
Users can upload a leaf image, and the system predicts the plant disease with high accuracy.

---

## 📌 Project Overview

Plant diseases cause significant losses in agriculture. Early detection helps farmers take preventive actions and improve crop productivity.

This project uses a **Convolutional Neural Network (CNN)** based on **MobileNetV2** to classify plant diseases from leaf images.
The trained model is integrated into a **Streamlit web application** for easy usage.

---

## 🚀 Features

* Upload plant leaf images for disease detection
* Real-time predictions using a trained deep learning model
* Simple and interactive web interface
* High accuracy using transfer learning
* Lightweight and deployable AI system

---

## 🧠 Model Information

| Component          | Details                           |
| ------------------ | --------------------------------- |
| Model Architecture | MobileNetV2 (Transfer Learning)   |
| Framework          | TensorFlow / Keras                |
| Dataset            | PlantVillage Dataset              |
| Input Image Size   | 224 × 224                         |
| Classes            | Multiple plant disease categories |
| Training Platform  | Kaggle GPU                        |

---

## 📂 Project Structure

```
plant-disease-detector
│
├── app.py                    # Streamlit web application
├── plant_disease_model.h5   # Trained deep learning model
├── requirements.txt         # Required Python packages
└── README.md                # Project documentation
```

---

## ⚙️ Installation

### 1️⃣ Clone the repository

```
git clone https://github.com/yourusername/plant-disease-detector.git
```

### 2️⃣ Navigate to the project folder

```
cd plant-disease-detector
```

### 3️⃣ Install dependencies

```
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start the Streamlit server:

```
streamlit run app.py
```

Then open your browser:

```
http://localhost:8501
```

Upload a plant leaf image to get disease predictions.

---

## 🌍 Deployment

The application can be deployed using **Streamlit Community Cloud**.

Steps:

1. Upload project to GitHub
2. Connect GitHub repo to Streamlit Cloud
3. Deploy the `app.py` file

After deployment you will receive a public URL like:

```
https://plant-disease-detector.streamlit.app
```

---

## 📊 Dataset

This project uses the **PlantVillage Dataset**, which contains thousands of labeled plant leaf images for disease classification.

Dataset includes diseases from plants such as:

* Apple
* Potato
* Tomato
* Pepper

---

## 🖼 Example Workflow

```
Upload Leaf Image
        ↓
Image Preprocessing
        ↓
Deep Learning Model (MobileNetV2)
        ↓
Disease Prediction
        ↓
Result Displayed on Web Interface
```

---

## 🛠 Technologies Used

* Python
* TensorFlow
* Keras
* Streamlit
* NumPy
* Pillow
* Kaggle GPU

---

## 📈 Future Improvements

* Add more plant species
* Provide disease treatment recommendations
* Improve mobile compatibility
* Deploy as a mobile application
* Integrate real-time camera detection

---

## 👨‍💻 Author

**Pavan Kumar**

B.Tech Student | AI & Full Stack Enthusiast

---

## ⭐ Support

If you like this project, consider giving it a ⭐ on GitHub!

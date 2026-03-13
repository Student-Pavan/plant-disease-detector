import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model

model = tf.keras.models.load_model("plant_disease_model.h5")

# Full PlantVillage class list (38 classes)

class_names = [
"Apple___Apple_scab",
"Apple___Black_rot",
"Apple___Cedar_apple_rust",
"Apple___healthy",
"Blueberry___healthy",
"Cherry_(including_sour)__*Powdery_mildew",
"Cherry*(including_sour)__*healthy",
"Corn*(maize)__*Cercospora_leaf_spot Gray_leaf_spot",
"Corn*(maize)_**Common_rust*",
"Corn*(maize)__*Northern_Leaf_Blight",
"Corn*(maize)***healthy",
"Grape___Black_rot",
"Grape___Esca*(Black_Measles)",
"Grape___Leaf_blight*(Isariopsis_Leaf_Spot)",
"Grape___healthy",
"Orange___Haunglongbing*(Citrus_greening)",
"Peach___Bacterial_spot",
"Peach___healthy",
"Pepper,_bell___Bacterial_spot",
"Pepper,_bell___healthy",
"Potato___Early_blight",
"Potato___Late_blight",
"Potato___healthy",
"Raspberry___healthy",
"Soybean___healthy",
"Squash___Powdery_mildew",
"Strawberry___Leaf_scorch",
"Strawberry___healthy",
"Tomato___Bacterial_spot",
"Tomato___Early_blight",
"Tomato___Late_blight",
"Tomato___Leaf_Mold",
"Tomato___Septoria_leaf_spot",
"Tomato___Spider_mites Two-spotted_spider_mite",
"Tomato___Target_Spot",
"Tomato___Tomato_Yellow_Leaf_Curl_Virus",
"Tomato___Tomato_mosaic_virus",
"Tomato___healthy"
]

st.title("🌿 AI Plant Disease Detection")

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg","jpeg","png"])

if uploaded_file is not None:


    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Leaf", use_container_width=True)

    # Preprocess image
    img = image.resize((224,224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img)
    class_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction)) * 100

    predicted_class = class_names[class_index]

    # Separate plant and disease
    plant, disease = predicted_class.split("___")

    # Detect health condition
    if "healthy" in disease.lower():
        health_status = "Healthy Leaf ✅"
    else:
        health_status = "Diseased Leaf ⚠️"

    # Display results
    st.subheader("Prediction Result")

    st.write("🌱 **Plant:**", plant)
    st.write("🦠 **Condition:**", disease)
    st.write("📊 **Confidence:**", f"{confidence:.2f}%")

    if health_status == "Healthy Leaf ✅":
        st.success(health_status)
    else:
        st.error(health_status)

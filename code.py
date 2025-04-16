from google.colab import files
uploaded = files.upload()
%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
import pytesseract
from PIL import Image
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- Text Cleaning Function --- #
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

# --- Streamlit UI --- #
st.title("📄 AI-Powered Document Verifier")
st.markdown("Upload a document image (JPG/PNG), and this app will verify its validity using OCR and a trained ML model.")

# --- Train Model Function --- #
@st.cache_data
def load_and_train_model():
    df = pd.read_csv("ai_document_verification_dataset.csv")
    df.dropna(subset=["comments"], inplace=True)

    # Generate labels from 'comments'
    df['label'] = df['comments'].apply(lambda x: 'real' if 'valid' in x.lower() else 'fake')
    df['comments'] = df['comments'].apply(clean_text)

    X = df['comments']
    y = df['label']

    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.success(f"✅ Model trained successfully with accuracy: {acc*100:.2f}%")

    return model, vectorizer

# --- Load the Model --- #
model, vectorizer = load_and_train_model()

# --- Image Upload --- #
st.markdown("### 📤 Upload Document Image")
image_file = st.file_uploader("Choose a document image", type=["jpg", "jpeg", "png"])

if image_file is not None:
    img = Image.open(image_file)
    st.image(img, caption="Uploaded Document", use_column_width=True)

    with st.spinner("🔍 Extracting text from image..."):
        extracted_text = pytesseract.image_to_string(img)
        cleaned_text = clean_text(extracted_text)

        if cleaned_text:
            st.markdown("### 📝 Extracted Text")
            st.write(cleaned_text)

            vectorized_input = vectorizer.transform([cleaned_text])
            prediction = model.predict(vectorized_input)[0]
            probability = model.predict_proba(vectorized_input).max()

            st.markdown("### 🔎 Prediction")
            st.info(f"This document is **{prediction.upper()}**")
            st.write(f"Confidence: **{probability * 100:.2f}%**")
        else:
            st.warning("⚠️ Could not extract readable text from the image.")
from pyngrok import ngrok
ngrok.set_auth_token("2vSUvFuY7N5N0o6JopVukrZRmNG_2ccbn1fzYbCsdfPEpejXn")
from pyngrok import ngrok
import os

# Kill existing tunnels
ngrok.kill()

# Run Streamlit app
os.system('streamlit run app.py &')

# Open tunnel on correct port
public_url = ngrok.connect(addr="8501", proto="http")
print(f"🌐 Click here to open your app: {public_url}")

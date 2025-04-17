# import streamlit as st
# import pytesseract
# import cv2
# import numpy as np
# from PIL import Image
# import re
# import sqlite3
# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score, classification_report
# import seaborn as sns
# import matplotlib.pyplot as plt

# Setup tesseract path
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'  # Adjust if needed

# Connect to your database (create or load)
conn = sqlite3.connect('documents.db')
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS documents (
    doc_number TEXT, name TEXT, email TEXT
)''')
conn.commit()

# --- Streamlit App ---
st.title("🧾 OCR and Document Verification App")

# 1. Image Upload
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image for OpenCV
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # OCR Extraction
    text = pytesseract.image_to_string(thresh)
    st.subheader("🔍 Extracted Text")
    st.text(text)

    # Clean text
    def clean_text(text):
        text = ' '.join(text.split())
        text = re.sub(r'[^A-Za-z0-9\s]', '', text)
        return text.lower()

    cleaned_text = clean_text(text)

    # Extract phone numbers
    phone_pattern = r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
    phone_numbers = re.findall(phone_pattern, cleaned_text)
    st.write("📞 Extracted Phone Numbers:", phone_numbers)

# 2. Document Verification
st.subheader("📄 Document Verification")
doc_number = st.text_input("Document Number", value="DOC1234")
name = st.text_input("Name", value="John Doe")
email = st.text_input("Email", value="john.doe@example.com")

if st.button("Verify Document"):
    cursor.execute("SELECT * FROM documents WHERE doc_number=? AND name=? AND email=?", (doc_number, name, email))
    result = cursor.fetchone()
    if result:
        st.success("✅ Document is valid.")
    else:
        st.error("❌ Document not found in database.")

# 3. Train & Evaluate Model
st.subheader("📊 Model Training & Evaluation")

if st.button("Train & Evaluate Model"):
    X, y = make_classification(n_samples=1000, n_features=20,
                               n_informative=15, n_redundant=5,
                               n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    st.subheader("Confusion Matrix")
    fig1, ax1 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
    st.pyplot(fig1)

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    ap_score = average_precision_score(y_test, y_probs)

    fig2, ax2 = plt.subplots()
    ax2.plot(recall, precision, marker='.')
    ax2.set_title(f'Precision-Recall Curve (AP={ap_score:.2f})')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    st.pyplot(fig2)

    # Classification Report
    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))

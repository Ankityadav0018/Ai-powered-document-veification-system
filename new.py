from google.colab import files
uploaded = files.upload()
!apt-get install tesseract-ocr
!pip install pytesseract opencv-python
import pytesseract
import os

# Set Tesseract path manually
# Get the path to the tesseract executable, which might vary depending on your system
tesseract_path = '/usr/bin/tesseract'  # Update with your actual path if needed

# Set the path for pytesseract
pytesseract.pytesseract.tesseract_cmd = tesseract_path
# Verify if the path is set correctly
print(f"Tesseract path: {pytesseract.pytesseract.tesseract_cmd}")
# Check if the file exists
if not os.path.exists(tesseract_path):
  print(f"Warning: Tesseract executable not found at {tesseract_path}")

import cv2
import pytesseract
from PIL import Image
import io

# Get the uploaded file name
image_path = list(uploaded.keys())[0]

# Read the image using OpenCV
img = cv2.imread(image_path)

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Optional: Apply thresholding (useful for clearer text extraction)
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

# Use pytesseract to extract text from the image
text = pytesseract.image_to_string(thresh)

# Display the extracted text
print(text)
import re

# Clean up the text by removing unnecessary characters, extra spaces, and newlines
def clean_text(text):
    # Remove extra spaces and newlines
    text = ' '.join(text.split())

    # Remove any non-alphanumeric characters except spaces
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)

    # Convert text to lowercase for consistency
    text = text.lower()

    return text

# Clean the extracted text
cleaned_text = clean_text(text)

# Display the cleaned text
print(cleaned_text)
# Regex pattern for phone numbers
phone_pattern = r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
phone_numbers = re.findall(phone_pattern, cleaned_text)

# Display extracted phone numbers
print("Extracted Phone Numbers:", phone_numbers)
def verify_document_data(doc_number, name, email):
    cursor.execute("SELECT * FROM documents WHERE doc_number=? AND name=? AND email=?", (doc_number, name, email))
    result = cursor.fetchone()  # If match is found, it will return a tuple with data
    return result is not None

# Example: Verify extracted document number, name, and email
doc_number = "DOC1234"
name = "John Doe"
email = "john.doe@example.com"

is_valid = verify_document_data(doc_number, name, email)
print("Is the document valid?", is_valid)
# 1. Generate sample data
X, y = make_classification(n_samples=1000, n_features=20,
                           n_informative=15, n_redundant=5,
                           n_classes=2, random_state=42)

# 2. Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)

# 3. Train a classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 4. Predict class labels
y_pred = model.predict(X_test)

# 5. Predict probabilities
y_probs = model.predict_proba(X_test)[:, 1]  # Probabilities for positive class

# 6. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# 7. Recall Curve (Precision-Recall)
precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
ap_score = average_precision_score(y_test, y_probs)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.', label=f'AP={ap_score:.2f}')
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.grid()
plt.show()

# 8. Print predicted probabilities (first 10)
print("Predicted probabilities (first 10 samples):")
print(y_probs[:10])

# 9. Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

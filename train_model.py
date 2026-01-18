import pandas as pd
import numpy as np
import re
import string
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC  # <-- Changed Algorithm
from sklearn.metrics import accuracy_score

print("⏳ Loading Dataset...")

# 1. Load Data
train_data = pd.read_csv('train.txt', sep=';', names=['text', 'emotion'])
test_data = pd.read_csv('test.txt', sep=';', names=['text', 'emotion'])

print(f"✅ Data Loaded! Training on {len(train_data)} rows.")

# 2. Data Cleaning Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

print("🧹 Cleaning Data...")
train_data['clean_text'] = train_data['text'].apply(clean_text)
test_data['clean_text'] = test_data['text'].apply(clean_text)

# 3. Vectorization (UPGRADE: N-grams)
print("🧮 converting text to numbers (using N-grams)...")
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000) 
X_train = vectorizer.fit_transform(train_data['clean_text'])
X_test = vectorizer.transform(test_data['clean_text'])

y_train = train_data['emotion']
y_test = test_data['emotion']

# 4. Model Training (UPGRADE: LinearSVC)
print("🧠 Training Model with LinearSVC (High Accuracy)...")
model = LinearSVC(max_iter=1000, dual='auto') 
model.fit(X_train, y_train)

# 5. Accuracy Check
print("📊 Checking Accuracy...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"🚀 New Model Accuracy: {accuracy * 100:.2f}%")

# 6. Saving Model
print("💾 Saving Model files...")
joblib.dump(model, 'emotion_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("✅ Process Complete! High accuracy model saved.")
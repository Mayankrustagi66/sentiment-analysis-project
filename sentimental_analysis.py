# ==============================
# SENTIMENT ANALYSIS PROJECT
# ==============================

import pandas as pd
import numpy as np
import re
import string
import nltk
import joblib
import os

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.decomposition import LatentDirichletAllocation

# Download NLTK resources (only first time)
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# ==============================
# 1. LOAD DATA
# ==============================

data = pd.read_csv("dataset.csv")

print("Dataset shape:", data.shape)
print(data.head())

# ==============================
# 2. TEXT PREPROCESSING
# ==============================

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

data['clean_text'] = data['text'].apply(clean_text)

# ==============================
# 3. FEATURE ENGINEERING (TF-IDF)
# ==============================

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2)
)

X = vectorizer.fit_transform(data['clean_text'])
y = data['sentiment']

# ==============================
# 4. TRAIN TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==============================
# 5. MODEL TRAINING
# ==============================

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ==============================
# 6. MODEL EVALUATION
# ==============================

y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# ==============================
# 7. TOPIC MODELING (LDA)
# ==============================

print("\nPerforming Topic Modeling...")

count_vectorizer = CountVectorizer(
    max_features=3000,
    stop_words='english'
)

X_counts = count_vectorizer.fit_transform(data['clean_text'])

lda = LatentDirichletAllocation(
    n_components=5,
    random_state=42
)

lda.fit(X_counts)

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"\nTopic {topic_idx + 1}:")
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))

feature_names = count_vectorizer.get_feature_names_out()
display_topics(lda, feature_names, 10)

# ==============================
# 8. SAVE MODEL & VECTORIZER
# ==============================

if not os.path.exists("saved_models"):
    os.makedirs("saved_models")

joblib.dump(model, "saved_models/sentiment_model.pkl")
joblib.dump(vectorizer, "saved_models/tfidf_vectorizer.pkl")
joblib.dump(lda, "saved_models/lda_model.pkl")

print("\nModels saved successfully!")

# ==============================
# 9. PREDICTION FUNCTION
# ==============================

def predict_sentiment(text):
    cleaned = clean_text(text)
    vect = vectorizer.transform([cleaned])
    prediction = model.predict(vect)[0]
    return prediction

# Example
sample_text = "The delivery was very slow and customer support was terrible"
print("\nSample Prediction:", predict_sentiment(sample_text))
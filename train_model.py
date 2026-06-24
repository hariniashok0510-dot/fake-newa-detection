
import os
import sys
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, classification_report
from preprocess import clean_text


def train():
    csv_path = 'news.csv'
    if not os.path.exists(csv_path):
        print("❌  news.csv not found!")
        print("    Download from https://www.kaggle.com/c/fake-news/data")
        print("    Rename train.csv → news.csv and place it here.")
        sys.exit(1)

    print("📂  Loading dataset...")
    df = pd.read_csv(csv_path)

    # Support multiple common column names
    if 'text' not in df.columns:
        # Try to combine title + text
        if 'title' in df.columns and 'text' in df.columns:
            df['text'] = df['title'] + ' ' + df['text']
        elif 'title' in df.columns:
            df['text'] = df['title']

    df.dropna(subset=['text', 'label'], inplace=True)
    df['label'] = df['label'].str.upper().str.strip()

    print(f"📊  Dataset size : {len(df)} rows")
    print(f"📊  Class counts : {df['label'].value_counts().to_dict()}")

    print("🔧  Cleaning text (this may take a minute)...")
    df['cleaned'] = df['text'].apply(clean_text)

    X = df['cleaned']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("🔢  Vectorising with TF-IDF...")
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec  = tfidf.transform(X_test)

    print("🤖  Training Passive Aggressive Classifier...")
    model = PassiveAggressiveClassifier(max_iter=100, random_state=42)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅  Accuracy : {acc * 100:.2f}%")
    print("\n📋  Classification Report:")
    print(classification_report(y_test, y_pred))

    os.makedirs('model', exist_ok=True)
    joblib.dump(model, 'model/model.pkl')
    joblib.dump(tfidf,  'model/vectorizer.pkl')
    print("\n💾  Model saved to  model/model.pkl")
    print("💾  Vectorizer saved to  model/vectorizer.pkl")
    print("\n🚀  Now run:  streamlit run app.py")


if __name__ == '__main__':
    train()
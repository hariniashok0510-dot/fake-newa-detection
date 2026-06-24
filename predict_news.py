# predict_news.py

import joblib
import sys

# Load saved pipeline model
model = joblib.load("fake_news_model.pkl")


def predict_news(text):
    prediction = model.predict([text])[0]
    return "Real News" if prediction == 1 else "Fake News"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_news.py \"Enter news text here\"")
        sys.exit(1)

    news_text = " ".join(sys.argv[1:])
    result = predict_news(news_text)

    print(f"Prediction: {result}")
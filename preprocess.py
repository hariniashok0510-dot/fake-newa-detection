import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords', quiet=True)

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """Clean and preprocess input text for model prediction."""
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)   # Remove URLs
    text = re.sub(r'[^a-z\s]', '', text)                 # Remove special chars
    tokens = text.split()
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)



# predict_model.py

import re
import string
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from tensorflow.keras.models import load_model

# NLTK stopwords'i indirme
nltk.download('punkt')
nltk.download('stopwords')

# Temizleme fonksiyonu
def clean_tweet(tweet):
    if not isinstance(tweet, str):
        return ''
    tweet = tweet.lower()
    tweet = re.sub(r'http\S+|www\S+', '', tweet)
    tweet = re.sub(r'@\w+', '', tweet)
    tweet = re.sub(r'#\w+', '', tweet)
    tweet = re.sub(r'\d+', '', tweet)
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(tweet)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)

# Modeli ve TF-IDF vektörleştiriciyi yükle
model = load_model('sentiment_analysis_model.h5')
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

# Tahmin yapılacak örnek veri
input_data = ["Rocket League is too hard game but I love play this game"]

# Veriyi temizle
cleaned_input = [clean_tweet(tweet) for tweet in input_data]

# TF-IDF ile vektörleştir
input_data_tfidf = tfidf_vectorizer.transform(cleaned_input)

# Tahmin yap
predictions = model.predict(input_data_tfidf)

# Tahmin sonuçlarını yazdır
sentiment_mapping = {1: 'Positive', 0: 'Negative', 2: 'Neutral', 3: 'Irrelevant'}
predicted_labels = [sentiment_mapping[label.argmax()] for label in predictions]

print(f"Tahmin Edilen Duygular: {predicted_labels}")

# train_model.py

import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import pickle

# NLTK stopwords'i indirme
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# Veri dosyasını yükle
train_df = pd.read_csv('data/twitter_training.csv', header=None)
train_df.columns = ['ID', 'Category', 'Sentiment', 'Tweet']

validation_df = pd.read_csv('data/twitter_validation.csv', header=None)
validation_df.columns = ['ID', 'Category', 'Sentiment', 'Tweet']

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

# Eğitim ve doğrulama verilerini temizle
train_df['Cleaned_Tweet'] = train_df['Tweet'].apply(clean_tweet)
validation_df['Cleaned_Tweet'] = validation_df['Tweet'].apply(clean_tweet)

# Sentiment etiketlerini sayısal değerlere dönüştürme
sentiment_mapping = {
    'Positive': 1,
    'Negative': 0,
    'Neutral': 2,
    'Irrelevant': 3
}
train_df['Sentiment_Label'] = train_df['Sentiment'].map(sentiment_mapping)
validation_df['Sentiment_Label'] = validation_df['Sentiment'].map(sentiment_mapping)

# Temizlenmiş veriyi kaydet
train_df.to_csv('twitter_training_cleaned_with_labels.csv', index=False)
validation_df.to_csv('twitter_validation_cleaned_with_labels.csv', index=False)

# TF-IDF vektörleştiriciyi oluştur ve fit et
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(train_df['Cleaned_Tweet'])
X_validation_tfidf = tfidf_vectorizer.transform(validation_df['Cleaned_Tweet'])

# TF-IDF vektörleştiriciyi kaydet
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)

# Modeli oluştur
model = Sequential()
model.add(Dense(512, input_dim=X_train_tfidf.shape[1], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))

# Modeli derle
model.compile(loss='sparse_categorical_crossentropy', 
              optimizer=Adam(), 
              metrics=['accuracy'])

# Modelin özetini yazdır
model.summary()

# Modeli eğit
model.fit(X_train_tfidf, train_df['Sentiment_Label'], 
          epochs=5, 
          batch_size=32, 
          validation_data=(X_validation_tfidf, validation_df['Sentiment_Label']))

# Modeli kaydet
model.save('sentiment_analysis_model.h5')

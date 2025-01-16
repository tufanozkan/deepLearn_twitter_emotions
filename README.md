# Twitter Sentiment Analysis

This project aims to analyze the sentiment of tweets using a deep learning model. It employs data preprocessing, TF-IDF vectorization, and a neural network to classify tweets into four sentiment categories: Positive, Negative, Neutral, and Irrelevant.

## Features
- Preprocesses raw Twitter data by cleaning and tokenizing text.
- Utilizes TF-IDF for feature extraction.
- Implements a deep learning model with dense layers for classification.
- Provides a prediction script for real-time sentiment analysis.

## Dataset
The project uses two datasets:
1. **Training Dataset**: `twitter_training.csv`
2. **Validation Dataset**: `twitter_validation.csv`

Each dataset includes:
- Tweet ID
- Category
- Sentiment (Positive, Negative, Neutral, Irrelevant)
- Raw Tweet text

## Model Architecture
- Input: TF-IDF vectorized tweets with a maximum of 5000 features.
- Hidden Layers:
  - Dense layer with 512 neurons and ReLU activation.
  - Dropout layer with 50% dropout rate.
  - Dense layer with 256 neurons and ReLU activation.
  - Dropout layer with 50% dropout rate.
- Output: Dense layer with 4 neurons and Softmax activation.

## How to Use

### 1. Train the Model
Run the `train_model.py` script to:
- Clean and preprocess the training data.
- Train the sentiment analysis model.
- Save the trained model and TF-IDF vectorizer.

### 2. Predict Sentiments
Run the `python predict_model.py` script to:
- Predict the sentiment of new tweets.

### 3. Input Example
- "input_data = ["I love the new feature on this app!"]" The script outputs the predicted sentiment.

## Requirements
-
	•	Python 3.x
	•	Libraries: pandas, nltk, sklearn, tensorflow, pickle

## Results
The trained model achieved high accuracy in identifying sentiments in tweets. The use of TF-IDF and dropout layers helped prevent overfitting and improved generalization.

## Future Work
- Experiment with advanced models like **BERT** or **GPT-based transformers** for improved accuracy and better contextual understanding.  
- Incorporate **data augmentation** techniques to enhance the diversity and robustness of the training data.  
- Add support for **multilingual sentiment analysis** to analyze tweets in multiple languages.  
- Optimize the preprocessing pipeline for better handling of noisy, sarcastic, or mixed-emotion text.  
- Develop a **web interface or API** for seamless integration and real-time sentiment prediction.  
- Explore **unsupervised learning** methods to classify new or unlabelled tweets dynamically.  
- Integrate visualizations like **word clouds** and **sentiment trends over time** for enhanced reporting.  
- Evaluate and compare performance with different vectorization techniques like **word embeddings (Word2Vec, GloVe)**.  



import streamlit as st
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import string

# Load the dataset
@st.cache_resource
@st.cache_data
def load_data():
    return pd.read_csv("WELFAKE_Dataset.csv", encoding='ISO-8859-1')

# Preprocess the data
def preprocess_data(data):
    data['title'] = data['title'].str.lower()
    data['text'] = data['text'].str.lower()
    data = data.drop(['Unnamed: 0'], axis=1)
    data['transformed_text'] = data.apply(lambda row: transform_text(row['title'] + ' ' + row['text']), axis=1)
    return data

# Text preprocessing function
def transform_text(text):
    ps = PorterStemmer()
    text = nltk.word_tokenize(text)
    text = [ps.stem(word) for word in text if word.isalnum() and word not in stopwords.words('english') and word not in string.punctuation]
    return " ".join(text)

# Train the model
def train_model(X_train, y_train):
    tfidf = TfidfVectorizer(max_features=3000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    mnb = MultinomialNB()
    mnb.fit(X_train_tfidf, y_train)
    return tfidf, mnb

# Main function
def main():
    st.title("Fake News Detection")

    # Load data
    df = load_data()  # Change 'data' to 'df'
    df = preprocess_data(df)  # Change 'data' to 'df'

    # Split data
    X = df['transformed_text']  # Change 'data' to 'df'
    y = df['label']  # Change 'data' to 'df'
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

    # Train the model
    tfidf, mnb = train_model(X_train, y_train)

    # User input
    input_text = st.text_input("Enter news text:")
    if st.button("Classify"):
        input_transformed = transform_text(input_text)
        input_tfidf = tfidf.transform([input_transformed])
        prediction = mnb.predict(input_tfidf)
        if prediction == 1:
            st.write("Predicted Label: Not Fake News")
        else:
            st.write("Predicted Label: Fake News")

    # Evaluation
    st.subheader("Model Evaluation")
    y_pred = mnb.predict(tfidf.transform(X_test))
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    confusion_mat = confusion_matrix(y_test, y_pred)
    st.write(f"Accuracy: {accuracy}")
    st.write(f"Precision: {precision}")
    st.write(f"Recall: {recall}")
    st.write(f"F1-score: {fscore}")
    st.write("Confusion Matrix:")
    st.write(confusion_mat)

if __name__ == "__main__":
    main()

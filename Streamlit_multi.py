import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Load the data
@st.cache
def load_data():
    return pd.read_csv('WELFAKE_Dataset.csv')

# Preprocess the data
def preprocess_data(data):
    data = data.dropna()
    return data

# Train the model
def train_model(X_train, y_train):
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    return model, vectorizer

# Main function
def main():
    st.title("Fake News Detection")

    # Load data
    data = load_data()
    data = preprocess_data(data)

    # Split data
    X = data['title'] + ' ' + data['text']
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model, vectorizer = train_model(X_train, y_train)

    # User input
    input_text = st.text_input("Enter news text:")
    if st.button("Classify"):
        input_tfidf = vectorizer.transform([input_text.lower()])
        prediction = model.predict(input_tfidf)
        if prediction == 1:
            st.write("Predicted Label: Not Fake News")
        else:
            st.write("Predicted Label: Fake News")

    # Evaluation
    st.subheader("Model Evaluation")
    X_test_tfidf = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Accuracy: {accuracy}")

if __name__ == "__main__":
    main()

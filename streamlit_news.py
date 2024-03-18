import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load the data
@st.cache_resource
def load_data():
    return pd.read_csv('WELFake_Dataset.csv')

# Preprocess the data
def preprocess_data(data):
    data = data.dropna()
    return data

# Train the model
def train_model(X_train, y_train):
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    lr_model = LogisticRegression()
    lr_model.fit(X_train_tfidf, y_train)
    return lr_model, vectorizer

# Main function
def main():
    st.title("Fake News Detection")
    
    # Load data
    data = load_data()
    data = preprocess_data(data)

    # Display first few rows and column names
    st.write("Data Preview:")
    st.write(data.head())  # Display first few rows of DataFrame
    st.write("Column Names:", data.columns)  # Print column names to check the correct name

    # Split data
    X = data['title'] + ' ' + data['text']
    y = data['label']  # Use the correct column name for the target variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model_type = st.sidebar.selectbox("Select Model", ["Logistic Regression"])  # Add more models as needed
    if model_type == "Logistic Regression":
        model, vectorizer = train_model(X_train, y_train)
    
    # User input
    input_text = st.text_input("Enter news text:")
    if st.button("Classify"):
        input_tfidf = vectorizer.transform([input_text.lower()])
        prediction = model.predict(input_tfidf)
        if prediction == 1:
            st.write("Predicted label:not Fake News")
        else:
            st.write("Predicted label: Fake News")

    # Evaluation
    st.subheader("Model Evaluation")
    if model_type == "Logistic Regression":
        X_test_tfidf = vectorizer.transform(X_test)
        lr_pred = model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, lr_pred)
        report = classification_report(y_test, lr_pred)
        st.write(f"Accuracy: {accuracy}")
        st.write(f"Classification Report:\n{report}")

if __name__ == "__main__":
    main()

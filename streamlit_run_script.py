import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Function to load and preprocess the data
def load_data():
    data = pd.read_csv('WELFake_Dataset.csv').dropna()
    X = data['title'] + ' ' + data['text']
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Function to preprocess text data
def preprocess_text(X_train, X_test):
    X_train_lower = [text.lower() for text in X_train]
    X_test_lower = [text.lower() for text in X_test]
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train_lower)
    X_test_tfidf = vectorizer.transform(X_test_lower)
    return X_train_tfidf, X_test_tfidf

# Function to train Logistic Regression model
def train_logistic_regression(X_train, y_train):
    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)
    return lr_model

# Function to train Random Forest model
def train_random_forest(X_train, y_train):
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)
    return rf_model

# Function to train SVM model
def train_svm(X_train, y_train):
    svm_model = SVC()
    svm_model.fit(X_train, y_train)
    return svm_model

# Main function
def main():
    st.title("Fake News Detection")

    # Load data
    X_train, X_test, y_train, y_test = load_data()

    # Preprocess text data
    X_train_tfidf, X_test_tfidf = preprocess_text(X_train, X_test)

    # Train models
    lr_model = train_logistic_regression(X_train_tfidf, y_train)
    rf_model = train_random_forest(X_train_tfidf, y_train)
    svm_model = train_svm(X_train_tfidf, y_train)

    # Evaluate models
    lr_pred = lr_model.predict(X_test_tfidf)
    rf_pred = rf_model.predict(X_test_tfidf)
    svm_pred = svm_model.predict(X_test_tfidf)

    # Display results
    st.subheader("Logistic Regression Model:")
    st.write("Accuracy:", accuracy_score(y_test, lr_pred))
    st.write("Classification Report:\n", classification_report(y_test, lr_pred))

    st.subheader("Random Forest Model:")
    st.write("Accuracy:", accuracy_score(y_test, rf_pred))
    st.write("Classification Report:\n", classification_report(y_test, rf_pred))

    st.subheader("Support Vector Machine (SVM) Model:")
    st.write("Accuracy:", accuracy_score(y_test, svm_pred))
    st.write("Classification Report:\n", classification_report(y_test, svm_pred))

if __name__ == "__main__":
    main()

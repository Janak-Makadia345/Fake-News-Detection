import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC  # Import Support Vector Classifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load the data
@st.cache_resource
@st.cache_data
def load_data():
    return pd.read_csv('WELFake_Dataset.csv')

# Preprocess the data
def preprocess_data(data):
    data = data.dropna()
    return data

# Train the model
def train_model(X_train, y_train, model_type='Logistic Regression'):
    if model_type == 'Logistic Regression':
        vectorizer = TfidfVectorizer()
        model = LogisticRegression()
    elif model_type == 'Random Forest':
        vectorizer = TfidfVectorizer()
        model = RandomForestClassifier(n_estimators=100)
    elif model_type == 'SVM':
        vectorizer = TfidfVectorizer()
        model = SVC(kernel='linear')  # Use linear kernel for SVM
    else:
        raise ValueError("Invalid model type. Please choose 'Logistic Regression', 'Random Forest', or 'SVM'.")
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
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
    model_type = st.sidebar.selectbox("Select Model", ["Logistic Regression", "Random Forest", "SVM"])
    model, vectorizer = train_model(X_train, y_train, model_type=model_type)
    
    # User input
    input_text = st.text_input("Enter news text:")
    if st.button("Classify"):
        input_tfidf = vectorizer.transform([input_text.lower()])
        prediction = model.predict(input_tfidf)
        if prediction == 1:
            st.write("Predicted Label: Fake News")
        else:
            st.write("Predicted Label: Not Fake News")

    # Evaluation
    st.subheader("Model Evaluation")
    if model_type == "Logistic Regression":
        X_test_tfidf = vectorizer.transform(X_test)
        lr_pred = model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, lr_pred)
        st.write(f"Accuracy: {accuracy}")
    elif model_type == "Random Forest":
        X_test_tfidf = vectorizer.transform(X_test)
        rf_pred = model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, rf_pred)
        st.write(f"Accuracy: {accuracy}")
    elif model_type == "SVM":
        X_test_tfidf = vectorizer.transform(X_test)
        svm_pred = model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, svm_pred)
        st.write(f"Accuracy: {accuracy}")

if __name__ == "__main__":
    main()

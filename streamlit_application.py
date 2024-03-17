import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Function to preprocess text data
def preprocess_text(text):
    return text.lower()

# Function to train Logistic Regression model
def train_model(X_train_tfidf, y_train):
    lr_model = LogisticRegression()
    lr_model.fit(X_train_tfidf, y_train)
    return lr_model

# Function to evaluate model
def evaluate_model(model, X_test_tfidf, y_test):
    lr_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, lr_pred)
    report = classification_report(y_test, lr_pred)
    cm = confusion_matrix(y_test, lr_pred)
    return accuracy, report, cm

# Main Streamlit app
def main():
    # Title and description
    st.title('Fake News Classifier')
    st.write('This app trains a Logistic Regression model to classify news articles as fake or real.')

    # Read the CSV file
    data = pd.read_csv('WELFake_Dataset.csv')

    # Drop rows with missing values (NaN)
    data = data.dropna()

    # Combine 'title' and 'text' columns
    data['combined_text'] = data['title'] + ' ' + data['text']

    # Preprocess the text data
    data['combined_text'] = data['combined_text'].apply(preprocess_text)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data['combined_text'], data['label'], test_size=0.2, random_state=42)

    # Create a TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()

    # Fit and transform on training data
    X_train_tfidf = vectorizer.fit_transform(X_train)

    # Transform testing data
    X_test_tfidf = vectorizer.transform(X_test)

    # Train the model
    lr_model = train_model(X_train_tfidf, y_train)

    # Evaluate the model
    accuracy, report, cm = evaluate_model(lr_model, X_test_tfidf, y_test)

    # Display results
    st.subheader('Model Evaluation')
    st.write(f'Accuracy: {accuracy}')
    st.write('Classification Report:')
    st.write(report)

    # Plot confusion matrix
    st.subheader('Confusion Matrix')
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={'size': 14})
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    st.pyplot()

if __name__ == '__main__':
    main()

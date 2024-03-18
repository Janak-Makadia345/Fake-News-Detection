# Load data
@st.cache
def load_data():
    return pd.read_csv('WELFake_Dataset.csv')

# Main function
def main():
    st.title("Fake News Detection")
    
    # Load data
    data = load_data()
    data = preprocess_data(data)

    # Check column names
    st.write("Column Names:", data.columns)  # Print column names to check the correct name

    # Split data
    X = data['title'] + ' ' + data['text']
    y = data['your_actual_column_name']  # Adjust this with the correct column name
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
            st.write("Predicted Label: Fake News")
        else:
            st.write("Predicted Label: Not Fake News")

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

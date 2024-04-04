import streamlit as st
import pickle
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import string
import joblib
from tensorflow.keras.models import load_model
import numpy as np

# Load the models from the pkl file
with open('models.pkl', 'rb') as f:
    loaded_models = pickle.load(f)

# Now you can access each loaded model from the dictionary
mnb_loaded = loaded_models['MultinomialNB']
rf_loaded = loaded_models['RandomForest']
lr_loaded = loaded_models['LogisticRegression']
ds_loaded = loaded_models['DecisionTree']
xgb_loaded = loaded_models['XGBoost']

model = load_model("mlp_model.h5")

def transform_text(text):
    ps = PorterStemmer()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

# Streamlit app
st.markdown("<h1 style='color: red; font-size: 50px'>Fake News Detection</h1>", unsafe_allow_html=True)

# User input
user_input = st.text_area("Enter text for prediction", "")

# Preprocess input
preprocessed_input = transform_text(user_input)
tfidf = joblib.load('tfidf_vectorizer.pkl')

# Pass a list containing the preprocessed input string to fit_transform
text = tfidf.transform([preprocessed_input]).toarray()  # Use transform instead of fit_transform

# Model selection
selected_model = st.sidebar.selectbox("Select Model", ["MultinomialNB", "RandomForest", "LogisticRegression", "DecisionTree", "XGBoost", "MLP"])

mnb_metrics = joblib.load('mnb_metrics.pkl')
rf_metrics = joblib.load('rf_metrics.pkl')
lr_metrics = joblib.load('lr_metrics.pkl')
ds_metrics = joblib.load('ds_metrics.pkl')
xg_metrics = joblib.load('xg_metrics.pkl')
mlp_metrics = joblib.load('mlp_metrics.pkl')

# Make prediction
if st.button("Make Prediction"):
    if selected_model == "MultinomialNB":
        prediction = mnb_loaded.predict(text)
        st.markdown("<h1 style='font-size:30px; color: #EAE86F'>Multinomial Naive Bayes Model Evaluation Metrics</h1>", unsafe_allow_html=True)

        # Display evaluation metrics
        st.markdown("<h3 style='font-size: 20px;'>Accuracy:</h3>", unsafe_allow_html=True)
        st.write(mnb_metrics['accuracy'])

        st.markdown("<h3 style='font-size: 20px;'>Precision:</h3>", unsafe_allow_html=True)
        st.write(mnb_metrics['precision'])

        st.markdown("<h3 style='font-size: 20px;'>Classification Report:</h3>", unsafe_allow_html=True)
        st.code(mnb_metrics['classification_report'])

        # Display the confusion matrix plot
        st.markdown("<h3 style='font-size: 20px;'>Confusion Matrix:</h3>", unsafe_allow_html=True)
        st.image("mnb_matrix.png") 

        st.markdown("<h3 style='font-size: 20px;'>MNB (Test set) Analysis:</h3>", unsafe_allow_html=True)
        st.image("mnb_test.png")

    elif selected_model == "RandomForest":
        prediction = rf_loaded.predict(text)
        st.markdown("<h1 style='font-size:30px; color: #EAE86F'>Ranndom Forest Model Evaluation Metrics</h1>", unsafe_allow_html=True)

        # Display evaluation metrics
        st.markdown("<h3 style='font-size: 20px;'>Accuracy:</h3>", unsafe_allow_html=True)
        st.write(rf_metrics['accuracy'])

        st.markdown("<h3 style='font-size: 20px;'>Precision:</h3>", unsafe_allow_html=True)
        st.write(rf_metrics['precision'])

        st.markdown("<h3 style='font-size: 20px;'>Classification Report:</h3>", unsafe_allow_html=True)
        st.code(rf_metrics['classification_report'])

        # Display the confusion matrix plot
        st.markdown("<h3 style='font-size: 20px;'>Confusion Matrix:</h3>", unsafe_allow_html=True)
        st.image("rf_matrix.png") 

        st.markdown("<h3 style='font-size: 20px;'>RF (Test set) Analysis:</h3>", unsafe_allow_html=True)
        st.image("rf_test.png")

    elif selected_model == "LogisticRegression":
        prediction = lr_loaded.predict(text)
        st.markdown("<h1 style='font-size:30px; color: #EAE86F'>Logistic Regression Model Evaluation Metrics</h1>", unsafe_allow_html=True)

        # Display evaluation metrics
        st.markdown("<h3 style='font-size: 20px;'>Accuracy:</h3>", unsafe_allow_html=True)
        st.write(lr_metrics['accuracy'])

        st.markdown("<h3 style='font-size: 20px;'>Precision:</h3>", unsafe_allow_html=True)
        st.write(lr_metrics['precision'])

        st.markdown("<h3 style='font-size: 20px;'>Classification Report:</h3>", unsafe_allow_html=True)
        st.code(lr_metrics['classification_report'])

        # Display the confusion matrix plot
        st.markdown("<h3 style='font-size: 20px;'>Confusion Matrix:</h3>", unsafe_allow_html=True)
        st.image("lr_matrix.png") 

        st.markdown("<h3 style='font-size: 20px;'>LR (Test set) Analysis:</h3>", unsafe_allow_html=True)
        st.image("lr_test.png")

    elif selected_model == "DecisionTree":
        prediction = ds_loaded.predict(text)
        st.markdown("<h1 style='font-size:30px; color: #EAE86F'>Decision Tree Model Evaluation Metrics</h1>", unsafe_allow_html=True)

        # Display evaluation metrics
        st.markdown("<h3 style='font-size: 20px;'>Accuracy:</h3>", unsafe_allow_html=True)
        st.write(ds_metrics['accuracy'])

        st.markdown("<h3 style='font-size: 20px;'>Precision:</h3>", unsafe_allow_html=True)
        st.write(ds_metrics['precision'])

        st.markdown("<h3 style='font-size: 20px;'>Classification Report:</h3>", unsafe_allow_html=True)
        st.code(ds_metrics['classification_report'])

        # Display the confusion matrix plot
        st.markdown("<h3 style='font-size: 20px;'>Confusion Matrix:</h3>", unsafe_allow_html=True)
        st.image("ds_matrix.png") 

        st.markdown("<h3 style='font-size: 20px;'>DST (Test set) Analysis:</h3>", unsafe_allow_html=True)
        st.image("ds_test.png")

    elif selected_model == "XGBoost":
        prediction = xgb_loaded.predict(text)
        st.markdown("<h1 style='font-size:30px; color: #EAE86F'>XG-Boost Model Evaluation Metrics</h1>", unsafe_allow_html=True)

        # Display evaluation metrics
        st.markdown("<h3 style='font-size: 20px;'>Accuracy:</h3>", unsafe_allow_html=True)
        st.write(xg_metrics['accuracy'])

        st.markdown("<h3 style='font-size: 20px;'>Precision:</h3>", unsafe_allow_html=True)
        st.write(xg_metrics['precision'])

        st.markdown("<h3 style='font-size: 20px;'>Classification Report:</h3>", unsafe_allow_html=True)
        st.code(xg_metrics['classification_report'])

        # Display the confusion matrix plot
        st.markdown("<h3 style='font-size: 20px;'>Confusion Matrix:</h3>", unsafe_allow_html=True)
        st.image("xg_matrix.png") 

        st.markdown("<h3 style='font-size: 20px;'>XG-B (Test set) Analysis:</h3>", unsafe_allow_html=True)
        st.image("xg_test.png")

    elif selected_model == "MLP":
        prediction = model.predict(text)
        st.markdown("<h1 style='font-size:30px; color: #EAE86F'>MLP Model Evaluation Metrics</h1>", unsafe_allow_html=True)

        # Display evaluation metrics
        st.markdown("<h3 style='font-size: 20px;'>Accuracy:</h3>", unsafe_allow_html=True)
        st.write(mlp_metrics['accuracy'])

        st.markdown("<h3 style='font-size: 20px;'>Precision:</h3>", unsafe_allow_html=True)
        st.write(mlp_metrics['precision'])

        st.markdown("<h3 style='font-size: 20px;'>Classification Report:</h3>", unsafe_allow_html=True)
        st.code(mlp_metrics['classification_report'])

        # Display the confusion matrix plot
        st.markdown("<h3 style='font-size: 20px;'>Confusion Matrix:</h3>", unsafe_allow_html=True)
        st.image("mlp_matrix.png") 

        st.markdown("<h3 style='font-size: 20px;'>MLP (Test set) Analysis:</h3>", unsafe_allow_html=True)
        st.image("mlp_test.png")
    
    else:
        prediction = None

    if prediction is not None:
        if prediction == 0:
            st.success("Prediction: FAKE NEWS")
        else:
            st.success("Prediction: REAL NEWS")
    else:
        st.error("Please select a model and provide input.")
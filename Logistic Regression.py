import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Read the CSV file
data = pd.read_csv('WELFake_Dataset.csv')

# Drop rows with missing values (NaN)
data = data.dropna()

# Assuming your CSV has 'title', 'text', and 'label' columns
X = data['title'] + ' ' + data['text']  # Combine title and text
y = data['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess the text data: lowercase
X_train_lower = [text.lower() for text in X_train]
X_test_lower = [text.lower() for text in X_test]

# Create a TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform on training data
X_train_tfidf = vectorizer.fit_transform(X_train_lower)

# Transform testing data
X_test_tfidf = vectorizer.transform(X_test_lower)

# Train Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train_tfidf, y_train)

# Evaluate the model
lr_pred = lr_model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, lr_pred)
report = classification_report(y_test, lr_pred)

print(f"Logistic Regression Model:")
print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}\n")

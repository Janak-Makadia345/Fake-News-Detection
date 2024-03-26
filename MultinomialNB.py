# %%
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.model_selection import train_test_split

# %%
df=pd.read_csv("WELFAKE_Dataset.csv",encoding='ISO-8859-1')

# %%
df.head()

# %% [markdown]
# df.sample(5)

# %%
df['title']

# %%
df.title[10]
# df.Label[10]

# %%
df.info

# %%
df.info()

# %%
#checking if there is any null value is there or not or missing value
df.isna().sum()

# %%
#dropping the columns id,images,date,web,category,axis=1 because we are dropping the coloumns not rows
df=df.drop(['Unnamed: 0'],axis=1)

# %%
df.head()

# %% [markdown]
# # DATA PREPROCESSING

# %%
#applying lowercase to statement that will add less complexity in tokenization
df['title']=df['title'].str.lower()
df['text']=df['text'].str.lower()

# %%
df.head()

# %%
df.isna().sum()

# %%
df.shape

# %%
df.head()

# %%
#the label dataset is slightly unbalanced
df['label'].value_counts()

# %%
import matplotlib.pyplot as plt
plt.pie(df['label'].value_counts(),labels=['1','0'],autopct="%0.2f")
plt.show()

# %%
import nltk
#it is used in natural language 

# %%
df['title'] = df['title'].astype(str)
df['text'] = df['text'].astype(str)

df['num_characters'] = df['title'].str.len() + df['text'].str.len()

# %%
df.head()

# %% [markdown]
# df.Statement[0]

# %%
df.title[0]

# %%
df[df['label'] == 0].head()


# %%
# Assuming 'title' and 'text' columns are already converted to string
df['num_words'] = df['title'].apply(lambda x: len(nltk.word_tokenize(x))) + df['text'].apply(lambda x: len(nltk.word_tokenize(x)))


# %%
df.head()

# %%
df.describe()

# %%
# Calculate the number of sentences for the 'title' and 'text' columns combined
df['num_sen'] = df.apply(lambda row: len(nltk.sent_tokenize(row['title']) + nltk.sent_tokenize(row['text'])), axis=1)
df.head()

# %%
df[['num_characters','num_words','num_sen']].describe()

# %%
df[df['label']==0][['num_characters','num_words','num_sen']].describe()

# %%
df[df['label']==1][['num_characters','num_words','num_sen']].describe()

# %%
import seaborn as sns

# %%
sns.histplot(df[df['label']==0]['num_characters'])
sns.histplot(df[df['label']==1]['num_characters'],color='red')

# %%
from nltk.corpus import stopwords
stopwords.words('english')

# %%
import nltk
nltk.download('stopwords')

# %%
import string
string.punctuation

# %%
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
ps.stem('loving')

# %%
def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text=y[:]#cloning
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text=y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
        
    return " ".join(y)

# %%
import swifter

# Apply transform_text function to both 'title' and 'text' columns and combine the results
df['transformed_text'] = df.swifter.apply(lambda row: transform_text(row['title']) + transform_text(row['text']), axis=1)


# %%
from wordcloud import WordCloud
wc = WordCloud(width=600, height=600, min_font_size=10, background_color='white')


# %%
truenews=wc.generate(df[df['label']==1]['transformed_text'].str.cat(sep=" "))

# %%
plt.figure(figsize=(15,6))
plt.imshow(truenews)

# %%
falsenews=wc.generate(df[df['label']==0]['transformed_text'].str.cat(sep=" "))

# %%
plt.figure(figsize=(15,6))
plt.imshow(falsenews)

# %%
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv=CountVectorizer()
tfidf=TfidfVectorizer(max_features=3000)

# %%
X=tfidf.fit_transform(df['transformed_text']).toarray()

# %%
X.shape

# %%
X


# %%
y=df['label'].values

# %%
y

# %%
from sklearn.model_selection import train_test_split

# %%
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)

# %%
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,classification_report

# %%
mnb=MultinomialNB()

# %%
mnb.fit(X_train,y_train)
y_pred2=mnb.predict(X_test)

predictedtext = transform_text("UNESCO declares PM Modi best Prime Minister")
input_tfidf = tfidf.transform([predictedtext])
user_prediction = mnb.predict(input_tfidf)
print("Predicted Label: ", user_prediction)

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Accuracy Score
accuracy = accuracy_score(y_test, y_pred2)
print("Accuracy:", accuracy)

# Precision Score
precision = precision_score(y_test, y_pred2, average='macro')
print("Precision:", precision)

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred2))

# Confusion Matrix
conf_mat = confusion_matrix(y_test, y_pred2)
plt.figure(figsize=(4,3))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap="Blues", xticklabels=["Class 0", "Class 1"])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# %%
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

# Calculate precision, recall, and F1-score
precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred2)

# Plotting precision, recall, and F1-score
labels = ['Class 0', 'Class 1']
x = np.arange(len(labels)) 
width = 0.2  

fig, ax = plt.subplots(figsize=(6,4))

ax.bar(x - width, precision, width, label='Precision')
ax.bar(x, recall, width, label='Recall')
ax.bar(x + width, fscore, width, label='F1-score')

ax.set_ylabel('Scores')
ax.set_title('Precision, Recall, and F1-score')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.show()

# %%
from sklearn.decomposition import PCA

# Get the predicted labels
y_pred = mnb.predict(X_test)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_test)

# Create a boolean mask for correctly and incorrectly predicted points
correctly_predicted = (y_pred == y_test)
incorrectly_predicted = (y_pred != y_test)

# Plotting the data points
plt.scatter(X_pca[correctly_predicted, 0], X_pca[correctly_predicted, 1], label='Correct', c='green', marker='o')
plt.scatter(X_pca[incorrectly_predicted, 0], X_pca[incorrectly_predicted, 1], label='Incorrect', c='red', marker='x')

plt.title('Multinomial Naive Bayes (Test set)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()

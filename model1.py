import re
import pandas as pd
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import GridSearchCV

import pickle

def preprocess_tweet(tweet):
    """
    Function to preprocess the tweet
    """
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
    tweet = re.sub(r'\@\w+|\#', '', tweet)
    tweet = re.sub(r'[^a-zA-Z0-9 ]+', '', tweet)
    tweet = tweet.lower()
    tweet = tweet.split()
    lemmatizer = WordNetLemmatizer()
    tweet = [lemmatizer.lemmatize(word) for word in tweet if not word in set(stopwords.words('english'))]
    tweet = ' '.join(tweet)
    return tweet

def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity

def getPolarity(text):
    return TextBlob(text).sentiment.polarity

def getAnalysis(score):
    if score <=0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'

# load the dataset and preprocess the tweets
df = pd.read_csv(r"Data_Mixed.csv").sample(n=7500, random_state=42)
df['tweet'] = df['tweet'].apply(preprocess_tweet)

# train the model
X = df['tweet']
y = df['Candidate']
cv = TfidfVectorizer()
X = cv.fit_transform(X)

# Set the hyperparameters for SVC
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf']
}


# Tune the hyperparameters using GridSearchCV
clf = GridSearchCV(SVC(), param_grid)
clf.fit(X, y)

# Save the model as a pickle file
with open('model.pkl', 'wb') as file:
    pickle.dump(clf, file)

# Save the CountVectorizer as a pickle file
with open('cv.pkl', 'wb') as file:
    pickle.dump(cv, file)

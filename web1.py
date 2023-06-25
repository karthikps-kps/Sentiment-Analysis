from flask import Flask, render_template, request
import pickle
import re
import numpy as np
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

# Load the model and CountVectorizer from the pickle files
with open('model.pkl', 'rb') as file:
    clf = pickle.load(file)

with open('cv.pkl', 'rb') as file:
    cv = pickle.load(file)

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

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        tweet = request.form['text']
        tweet = preprocess_tweet(tweet)
        tweet_t = cv.transform([tweet])
        #tweet_t = np.array2string(tweet_t.toarray())
        prediction = clf.predict(tweet_t)
        subjectivity = getSubjectivity(tweet)
        polarity = getPolarity(tweet)
        analysis = getAnalysis(polarity)
        return render_template('result.html', predicted_candidate=prediction[0], subjectivity=subjectivity, polarity=polarity, analysis=analysis)

@app.route('/result')
def result():
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)

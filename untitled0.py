#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 19:36:37 2020

@author: shivaneeprajapati
"""

#Python packages to import 
import pandas as pd
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plot 
import numpy as np
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
import warnings
#Getting CSV file and setting column values
df=pd.read_csv('/Users/shivaneeprajapati/Desktop/TakenMind/Spotle.ai/SentimentAnalysis_IMDB/Imdb_Sentiment_Analyzer_Arnav_Raina/movie_review_data_edited.csv',sep=",", names=['sentiment','reviews'])
df.loc[:, ['sentiment', 'reviews']] = df[['reviews', 'sentiment']].to_numpy()
#Splitting csv into testing and traing data
traindf =df.iloc[0:25000]
testdf = df.iloc[25000:50000]
#Getting info on training and testing data
print(traindf.describe())
print(testdf.describe())
print(traindf.sentiment)

#nullsum = traindf.isnull().sum()
#nullsum_TS = testdf.isnull().sum()

#traindf.fillna(traindf.mean())
#np.nan_to_num(traindf)
#np.nan_to_num(testdf)

#Ignore user and Package Warning if any
warnings.filterwarnings("ignore")

lemmatizer = WordNetLemmatizer()
def lem_tokens(tokens, lemmatizer):
    lemmetized = []
    for item in tokens:
        lemmetized.append(lemmatizer.lemmatize(item))
    return lemmetized

def tokenize(text):
    # remove non letters
    text = re.sub(r'\b\w{1,3}\b', '',text)
    text = re.sub("[^a-zA-Z]", " ", text)
    tokens = nltk.word_tokenize(text)

    # stem
    stems = lem_tokens(tokens, lemmatizer)
    return stems

stopset=set(stopwords.words('english'))
'''
we can create our model using our training data. In creating the model, 
I will use the TF-IDF as the vectorizer and the Stochastic Gradient Descend algorithm as the classifier.
'''
# fit_transform fits the model and learns the vocabulary.Also it transforms our corpus data into feature vectors. 
vectorizer=TfidfVectorizer(use_idf=True,ngram_range=(1,2), lowercase=True,tokenizer=tokenize,strip_accents='ascii',max_features=1000,stop_words=stopset,norm='l1')
#Taking first 25000 Reviews and sentiment for  training 
train_sentiments =traindf.sentiment

train_text = vectorizer.fit_transform(traindf.reviews)
#maps a dictonary for given sparse matrix 
vocab = vectorizer.vocabulary_
#Prints a dictonary vocublary
print(vocab)

test_sentiment=testdf.sentiment
test_text=vectorizer.transform(testdf.reviews)


#Implement Stochalistic Gradient Descent to minimize the loss and updating the model 
classifier = SGDClassifier(alpha=1e-05,max_iter=50,penalty='elasticnet')
#Training our Data Model

classifier = classifier.fit(train_text, train_sentiments)





predictions = classifier.predict(test_text)

# Model Evaluvation of other 25000 reviews
#Examining accuracy precision recall and f1 results
acc = accuracy_score(test_sentiment, predictions, normalize=True)
hit = precision_score(test_sentiment, predictions, average=None)
capture = recall_score(test_sentiment, predictions, average=None)
print('Model Accuracy:%.2f'%acc)
print(classification_report(test_sentiment, predictions))
print(confusion_matrix(test_sentiment, predictions))

#Visualizing Confusion Matrix
ncmat=confusion_matrix(test_sentiment,predictions)
plot.matshow(ncmat)
plot.colorbar()
plot.show()

import seaborn as sn

ax = sn.heatmap(ncmat, annot=True,fmt='d') #notation: "annot" not "annote"
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plot.show()


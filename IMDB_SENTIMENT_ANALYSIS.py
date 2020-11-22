#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 21:53:08 2020

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
df=pd.read_csv('/Users/shivaneeprajapati/Desktop/TakenMind/Spotle.ai/SentimentAnalysis_IMDB/Imdb_Sentiment_Analyzer_Arnav_Raina/movie_review_data_edited.csv',
               sep=",", names=['sentiment','reviews'])

df.loc[:, ['sentiment', 'reviews']] = df[['reviews', 'sentiment']].to_numpy()

print(df.shape)
print(df.head(10))

#Splitting csv into testing and traing data
traindf =df.iloc[0:25000]
testdf = df.iloc[25000:50000]

#Getting info on training and testing data
print(traindf.describe())
print(testdf.describe())
print(traindf.sentiment)

#Ignore user and Package Warning if any
warnings.filterwarnings("ignore")

##################################  SECOND PART  #############################################

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

##################################  THIRD PART  #############################################

import nltk
nltk.download('popular')
nltk.download('stopwords')

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
print(train_text)

#maps a dictonary for given sparse matrix 

vocab = vectorizer.vocabulary_

#Prints a dictonary vocublary

print(vocab)

##################################  FOURTH PART  #############################################
test_sentiment=testdf.sentiment
test_text=vectorizer.transform(testdf.reviews)
    

#Implement Stochalistic Gradient Descent to minimize the loss and updating the model 
classifier = SGDClassifier(alpha=1e-05,max_iter=50,penalty='elasticnet')
#Training our Data Model
classifier = classifier.fit(train_text, train_sentiments)

##################################  FIFTH PART  #############################################

predictions = classifier.predict(test_text)

# Model Evaluvation of other 25000 reviews
#Examining accuracy precision recall and f1 results
acc = accuracy_score(test_sentiment, predictions, normalize=True)
hit = precision_score(test_sentiment, predictions, average=None)
capture = recall_score(test_sentiment, predictions, average=None)
print('Model Accuracy:%.2f'%acc)
print(classification_report(test_sentiment, predictions))
print(confusion_matrix(test_sentiment, predictions))

##################################  SIXTH PART  #############################################


#Visualizing Confusion Matrix
import seaborn as sn
ncmat=confusion_matrix(test_sentiment,predictions)
ax = plot.subplot()
IMG1 = plot.matshow(ncmat)

from matplotlib import cm
# fake up the array of the scalar mappable. Urgh…
#sm = plot.cm.ScalarMappable(cmap=plot.cm.hsv, norm=plot.Normalize(vmin=0, vmax=1))
#sm._A = []
##plot.colorbar(sm)
#plot.colorbar(cm)

plot.colorbar(IMG1)
#####

df_cm = pd.DataFrame(ncmat, range(2), range(2))
#plt.figure(figsize=(10,7))
sn.set(font_scale=1) # for label size
sn.heatmap(df_cm, annot=True, fmt="d", annot_kws={"size": 25}) # font size

plot.show()
############################## SEVENTH STEP  ##############################################
#normalized train reviews
norm_train_reviews=df.reviews[:25000]
norm_train_reviews[0]

#Normalized test reviews
norm_test_reviews=df.reviews[25000:]
norm_test_reviews[30005]

#word cloud for positive review words
#import WordCloud
from wordcloud import WordCloud, STOPWORDS 
plot.figure(figsize=(10,10))
positive_text=norm_train_reviews[1]
WC=WordCloud(width=1000,height=500,max_words=500,min_font_size=5)
positive_words=WC.generate(positive_text)
plot.imshow(positive_words,interpolation='bilinear')
plot.show

#Word cloud for negative review words
plot.figure(figsize=(10,10))
negative_text=norm_train_reviews[8]
WC=WordCloud(width=1000,height=500,max_words=500,min_font_size=5)
negative_words=WC.generate(negative_text)
plot.imshow(negative_words,interpolation='bilinear')
plot.show


#heatmap for confusion matrix
import seaborn as sn

ax = sn.heatmap(ncmat, annot=True,fmt='d') #notation: "annot" not "annote"
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plot.show()
###########################  EIGHTH STEP ##################################################

#AUC-ROC CURVE 

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
    
def plot_roc_curve(fpr, tpr):
    plot.plot(fpr, tpr, color='orange', label='ROC')
    plot.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plot.xlabel('False Positive Rate')
    plot.ylabel('True Positive Rate')
    plot.title('Receiver Operating Characteristic (ROC) Curve')
    plot.legend()
    plot.show()
    
##probs = classifier.predict_proba(x_ts)
##probs = probs[:, 1]
    
#auc = roc_auc_score(test_sentiment,predictions)
#fpr, tpr, thresholds = roc_curve(test_sentiment,predictions)
#plot_roc_curve(fpr, tpr)
#print("AUC-ROC :",auc)





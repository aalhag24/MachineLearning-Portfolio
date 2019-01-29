# Natural Language Processing

# Importing the Librairies
import pandas as pd
import os
import re

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB

# Import the data
BinPath = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'bin'))
FilePath = BinPath + '\Restaurant_Reviews.tsv';
dataset = pd.read_csv(FilePath, delimiter='\t', quoting = 3)

# Cleaning the texts
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.apppend(review)

# Creating the Bag of Words model
CV = CountVectorizer(max_features = 1500)
X = CV.fit_transform(corpus).toarray()
Y = dataset.loc[:, 1].values

# Splitting into Training and Test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)

# Fitting a new result
classifier = GaussianNB()
classifier.fit(X_train, Y_train)

# Predicting the Test set results
Y_pred = classifier.predict(X_test)

# Create the Confusion Matrix
cm = confusion_matrix(Y_test, Y_pred)

# Must Calculate Accuracy
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.feature_extraction.text import HashingVectorizer
import itertools
import numpy as np

# Import dataset using pandas dataframe
df = pd.read_csv('datasets/politifact_data.csv')

# Inspect shape of `df`
df.shape

# Print first lines of `df`
df.head()

# Print first lines of `df`
df.head()

# Separate the labels and set up training and test datasets
y = df.label

# Drop the `label` column
# where numbering of news article is done that column is dropped in dataset
df.drop("label", axis=1)

# Make training and test sets 60-40 train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df['title'], y, test_size=0.6, random_state=53)

# Building the Count and Tfidf Vectors
# Initialize the `count_vectorizer`
count_vectorizer = CountVectorizer(stop_words='english')

# Fit and transform the training data
# Learn the vocabulary dictionary and return term-document matrix.
count_train = count_vectorizer.fit_transform(X_train)

# Transform the test set
count_test = count_vectorizer.transform(X_test)

# Initialize the `tfidf_vectorizer`
# This removes words which appear in more than 70% of the articles
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Fit and transform the training data
tfidf_train = tfidf_vectorizer.fit_transform(X_train)

# Transform the test set
tfidf_test = tfidf_vectorizer.transform(X_test)

# Get the feature names of `tfidf_vectorizer`
# print(tfidf_vectorizer.get_feature_names()[-10:])

# Get the feature names of `count_vectorizer`
# print(count_vectorizer.get_feature_names()[:10])

count_df = pd.DataFrame(
    count_train.A, columns=count_vectorizer.get_feature_names())

tfidf_df = pd.DataFrame(
    tfidf_train.A, columns=tfidf_vectorizer.get_feature_names())

difference = set(count_df.columns) - set(tfidf_df.columns)

print(difference)

# Check whether the DataFrames are equal
print(count_df.equals(tfidf_df))

print(count_df.head())

print(tfidf_df.head())


# Function to plot the confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    See full source and example: 
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# --------------------------------------------------------------
# Naive Bayes classifier for Multinomial model
# --------------------------------------------------------------


clf = MultinomialNB()

# Fit Naive Bayes classifier according to X, y
clf.fit(tfidf_train, y_train)

# Perform classification on an array of test vectors X.
# pred = clf.predict(tfidf_test)
# score = metrics.accuracy_score(y_test, pred)
# print("accuracy:   %0.3f" % score)
# cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
# plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])
# print(cm)


# Perform classification on an array of count vectors X.

clf = MultinomialNB()

clf.fit(count_train, y_train)

pred = clf.predict(count_test)
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])
print(cm)

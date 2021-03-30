import numpy as np  # linear algebra
import pandas as pd  # data processing

import os
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from translator import lang_translate


def get_all_query(title):
    title = [title]
    return title


def remove_punctuation_stopwords_lemma(sentence):
    # google translate if language is not english
    lang_translate(sentence)
    filter_sentence = ''
    stop_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    #sentence = re.sub(r'[^\w\s]', '', sentence)
    sentence = re.sub(r'https?://\S+', '', sentence, flags=re.MULTILINE)
    words = nltk.word_tokenize(sentence)  # tokenization
    words = [w for w in words if not w in stop_words]
    for word in words:
        filter_sentence = filter_sentence + ' ' + \
            str(lemmatizer.lemmatize(word)).lower()
    return filter_sentence

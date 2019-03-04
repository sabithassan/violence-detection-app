from app import app
from flask import render_template
from flask import jsonify
from flask import request

import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
import dill as pickled



def load_models(model_file):
    ''' loads previously trained model'''
    models = []
    with open(model_file, "rb") as f:
        while True:
            try:
                models.append(pickled.load(f))
            except EOFError:
                break
    return models



# loads all models
MNB_unigram_model, unigram_vectorizer = load_models ("./app/static/models/MNB_model_word_1-gram.pckl")
MNB_bigram_model, bigram_vectorizer = load_models ("./app/static/models/MNB_model_word_2-gram.pckl")

SVM_unigram_model, _ = load_models ("./app/static/models/SVM_model_word_1-gram.pckl")
SVM_bigram_model, _ = load_models ("./app/static/models/SVM_model_word_2-gram.pckl")

print ("All models loaded")


@app.route('/')
@app.route('/index')
def index():
    ''' Home page '''
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def detect():
    ''' detects level of offensiveness in text posted'''


    # Gets text and classifier from client
    user_query = [request.form["text"]]
    classifier = request.form["model"]


    # gets the model chosen by client
    model = None
    vectorizer = None

    if (classifier == "Word Unigram with Multinomial Naive Bayes"):
        model = MNB_unigram_model
        vectorizer = unigram_vectorizer
    elif (classifier == "Word Bigram with Multinomial Naive Bayes"):
        model = MNB_bigram_model
        vectorizer = bigram_vectorizer
    elif (classifier == "Word Bigram with Linear SVM"):
        model = SVM_bigram_model
        vectorizer = bigram_vectorizer
    else:
        model = SVM_unigram_model
        vectorizer = unigram_vectorizer

    # gets word n gram features and performs classification using
    # model chosen
    n_gram_features = vectorizer.transform(user_query)
    predicted_labels = model.predict(n_gram_features)
    prediction = str(predicted_labels[0])

    return jsonify({"level": prediction})
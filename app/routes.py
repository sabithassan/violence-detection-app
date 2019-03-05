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
MNB_word_unigram_model, word_unigram_vectorizer = load_models ("./app/static/models/MNB_model_word_1-gram.pckl")
MNB_word_bigram_model, word_bigram_vectorizer = load_models ("./app/static/models/MNB_model_word_2-gram.pckl")

SVM_word_unigram_model, _ = load_models ("./app/static/models/SVM_model_word_1-gram.pckl")
SVM_word_bigram_model, _ = load_models ("./app/static/models/SVM_model_word_2-gram.pckl")

SVM_char_3gram_model, char_3gram_vectorizer = load_models ("./app/static/models/SVM_model_char_3-gram.pckl")
SVM_char_5gram_model, char_5gram_vectorizer = load_models ("./app/static/models/SVM_model_char_5-gram.pckl")

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

    if (classifier == "Multinomial Naive Bayes (Word Unigram)"):
        model = MNB_word_unigram_model
        vectorizer = word_unigram_vectorizer
    elif (classifier == "Multinomial Naive Bayes (Word Bigram)"):
        model = MNB_word_bigram_model
        vectorizer = word_bigram_vectorizer
    elif (classifier == "Linear SVM (Word Unigram)"):
        model = SVM_word_unigram_model
        vectorizer = word_unigram_vectorizer
    elif (classifier == "Linear SVM (Word Bigram)"):
        model = SVM_word_bigram_model
        vectorizer = word_bigram_vectorizer
    elif (classifier == "Linear SVM (Char 3-gram)"):
        model = SVM_char_3gram_model
        vectorizer = char_3gram_vectorizer
    else:
        model = SVM_char_5gram_model
        vectorizer = char_5gram_vectorizer

    print (classifier)
    print (model)
    print (vectorizer)


    # gets word n gram features and performs classification using
    # model chosen
    n_gram_features = vectorizer.transform(user_query)
    predicted_labels = model.predict(n_gram_features)
    prediction = str(predicted_labels[0])

    return jsonify({"level": prediction})
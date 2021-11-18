# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 22:32:19 2021

@author: rasul
"""
import pandas
import csv
from collections import Counter, defaultdict
import re
import script_id as s_id
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, classification_report
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import langid


"""First we limit the size of the corpus due to computational restraints.
Then split the limited corpus into training and testing portions
according to training_pct (default to 80% training)."""
def dev_split(total, limit, training_pct=.8):
    langs = list(total.columns[1:])
    fragment = total.iloc[:limit, 1:]
    rows = len(fragment)
    split = round(rows * training_pct)
    train = fragment.iloc[:split]
    test = fragment.iloc[split:]
    #first label is ids and the rest are the language codes
    return (train, test, langs)


"""Relatively gentle cleaning that gets rid of common punctuation
and normalizes whitespace and numbers."""
def clean_raw(raw):
    raw = re.sub(r"[\.\?\!\",\*\u200B]", "", raw)
    raw = raw.lower()
    #remove punctuation
    #normalize whitespace 
    raw = re.sub("\s+", " ", raw)
    #normalize numbers
    raw = re.sub("[0-9]+", "#", raw)    
    return raw
"""This function groups languages by the dominant script. The underlying
assumption is that a language with multiple writing standards will have different 
labels for each standard. This is true for Chinese, Norwegian, and 
even Portuguese, but not for Serbian."""
def sort_by_script(train, scripts, langs):
    macros = defaultdict(lambda: [])
    macro_langs = defaultdict(lambda: [])  
    for lang in langs:
        #Get a string
        raw = train.loc[:, lang].to_string(index=False)
        clean = clean_raw(raw)
        label = s_id.classify_by_script(clean)[0][0]
        #say what languages the script has
        macro_langs[label].append([lang, len(clean)])
        macros[label].append(clean)
    return macros, macro_langs

"""Takes the dictionary of lists of languages by script and creates
an character-level bag of ngrams vectorizer and modeler
based on the vectorizer and modeler class parameters. Modeling
by script helps to reduce dimensionality and allows for denser
feature representation."""
def get_script_models(script_texts, script_langs, 
                      vectorizer, modeler):
    macro_models = {}
    for script in script_texts:
        #If only one language uses a script modelling is redundant
        if len(script_langs[script]) > 1:
            my_vectorizer = vectorizer(analyzer="char", ngram_range=(1,2), max_features=500, dtype=np.float32)
            vectors = my_vectorizer.fit_transform(script_texts[script])
            labels = [pair[0] for pair in script_langs[script]]
            my_model = modeler().fit(vectors, labels)
            macro_models[script] = [ my_vectorizer, my_model]
    return macro_models

"""Does a quick script check to see how many distinctively East Asian characters
are present. Because Korean and Japanese mix scripts,
and have characters dispersed throughout unicode, it's technically possible
that a text that is mostly in Chinese or unclassified characters 
is actually Japanese or Korean."""
def check_east_asian(script_data):
    counter = script_data[1]
    total = sum(counter.values())
    if counter["hgl"] >= total / 4:
        return "ko"
    elif counter["jpn"] >= total / 4:
        return "ja"
    elif counter["cjk"] >= total/ 4:
        return "zh"
    else:
        return "other"

"""This function generates a model for distinguishing 2 closely related languages
by converting each dialect into an array of vectors and then using the vector representation
to train a dense neural network. The loss function is binary cross entropy. The function
returns an array containing the vectorizer used to convert strings, the model itself,
and a dictionary associating outputs with language codes."""
def binary_dialect_model(train, dialects, ngram_range=(2,5), max_features=1000):
    d1, d2 = dialects[0], dialects[1]
    d1 = [str(entry) for entry in train.loc[:, d1]]
    d1_labels = [0] * len(d1)
    d2 = [str(entry) for entry in train.loc[:, d2]]
    d2_labels = [1] * len(d2)
    docs = d1 + d2
    total_labels = np.array(d1_labels + d2_labels, np.float32)
    label_numbers = {0 :dialects[0], 1: dialects[1]}
    my_vectorizer = CountVectorizer(max_features=max_features, analyzer="char", ngram_range=ngram_range)
    vectors = my_vectorizer.fit_transform(docs)
    feature_count = vectors.shape[1]
    vectors = vectors.toarray()
    callback = [ tf.keras.callbacks.EarlyStopping(
        monitor="loss", min_delta=.0160, patience=6)]
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(64, activation='relu', input_shape=(feature_count,)))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(vectors, total_labels, epochs=30, batch_size=512, callbacks=callback)
    return [my_vectorizer, model, label_numbers]


"""This function generates a model for distinguishing multiple closely related languages
by converting each dialect into an array of vectors and then using the vector representation
to train a dense neural network. The loss function is categorical cross entropy. The function
returns an array containing the vectorizer used to convert strings, the model itself,
and a dictionary associating outputs with language codes."""
def multi_dialect_model(train_set, dialects, ngram_range=(1,3), max_features=1000):
    #One Hot encoder takes list of lists as input
    encoder_dialects = [[dialect] for dialect in dialects]
    X = OneHotEncoder()
    one_hot = X.fit_transform(encoder_dialects).toarray()
    label_numbers = {}
    for dialect, vec in zip(dialects, one_hot):
        label_numbers[vec.argmax()] = dialect
    examples = []
    labels = []
    #Walk through each relevant column and add the text to examples and one hot label to labels
    for i, dialect in enumerate(dialects):
        dialect_examples = [str(entry) for entry in train_set.loc[:, dialect] if len(entry) > 0]
        my_labels = [one_hot[i]]* len(dialect_examples)
        examples += dialect_examples
        labels += my_labels

    labels = np.array(labels, np.float32)
    my_vectorizer = CountVectorizer(max_features=max_features, analyzer="char", ngram_range=ngram_range, dtype=np.int16)
    vectors = my_vectorizer.fit_transform(examples)
    feature_count = vectors.shape[1]
    vectors = vectors.toarray()
    callback = [ tf.keras.callbacks.EarlyStopping(
        monitor="loss", min_delta=.0160, patience=4)]
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(32, activation='tanh', input_shape=(feature_count,)))
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(16, activation='tanh'))
    model.add(tf.keras.layers.Dense(len(dialects), activation="softmax"))
    model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(vectors, labels, epochs=30, batch_size=512, callbacks=callback)
    return [my_vectorizer, model, label_numbers]

#Vectorize the text, feed it to the neural model and convert the output to a label
def id_dialect(text, dialect_model, model_type):
    vector = dialect_model[0].transform([text]).toarray()
    output = dialect_model[1].predict(vector)
    my_label = ""
    if model_type == "multi":
        #Multi model outputs a softmaxed vector
        my_label = output.argmax()
    #model_type == "binary":
    else:
        #Binary model outputs a single number
        my_label = int(output[0] > .5)
    predicted = dialect_model[2][my_label] 
    return predicted

"""This is the main function of the class. Given the string we want to label,
the models for each script, the dialect classifiers, and model number for 
the general classifier (currently just one for Multinomial Naive Bayers),
the function first assigns a script label to the string. If there is more than
one language for the script label, the function then puts the string through
the appropriate script model to get a language label. If the language label
is in one of the pre-identified groups of closely related languages, 
the relevant dialect model is then used to review the assigned language label
and reassigns a new one if necessary."""
def predict(text, models, labels, dialect_id, model_number=1):
    try:
        iber_id, zh_id = dialect_id["iber"], dialect_id["zh"]
        balkan_id, fr_id, nor_id = dialect_id["balkan"], dialect_id["fr"], dialect_id["nor"]
        clean = clean_raw(text)
        #Most common script*
        script_data = s_id.classify_by_script(clean)
        script = script_data[0][0]
        if script in ["cjk", "hgl", "jpn", "other"]:
            tentative = check_east_asian(script_data)
            if tentative not in ["zh", "other"]:
                return tentative
        if len(labels[script]) <= 1:
            try:
                #The first (and only) language label for the script
                return labels[script][0][0]
            #The entire script wasn't seen in training data so just guess the default language
            except:
                return s_id.script_defaults[script]
        else:
            try:
                #The vectorizer used to train the model
                vectorizer = models[script][0]
                #Use transform instead of fit to match dimensionality
                vector = vectorizer.transform([clean])
                predicted = models[script][model_number].predict(vector)
                #remove end brackets
                predicted = str(predicted)[2:-2]
            
                if predicted[:2] == "zh":
                    predicted = id_dialect(text, zh_id, "multi")
                if predicted in ["fr", "fr-ca", "ca", "oc"]:
                    predicted = id_dialect(text, fr_id, "multi")
                if predicted in ["pt", "pt-br", "gl", "es", "ca", "eo", "it"]:
                    predicted = id_dialect(text, iber_id, "multi")
                if predicted in  ["mk", "sr", "bs", "hr", "sl", "bg", "sh"]:
                    predicted =  id_dialect(text, balkan_id, "multi")
                if predicted in  ["da", "nn", "nb", "is", "sv"]:
                    predicted =  id_dialect(text, nor_id, "multi")            
            except:
                return s_id.script_defaults[script]
    except:
        return "unk"
    return predicted


def test(test_set, models, script_labels, dialect_id):
    true = []
    predictions = []
    langid_predictions = []
    test_array = test_set
    languages = test_set.columns
    test_array = test_array.to_numpy()
    for row in test_array:
        for lang, sent in zip(languages, row):
            if len(sent) > 0:
                label = predict(sent, models, script_labels, dialect_id)
                true.append(lang)
                predictions.append(label)
                langid_predictions.append(langid.classify(sent)[0])
    return true, predictions, langid_predictions

"""This function removes dialect distinctions in a list to be able to compare
output with other classifiers."""
def merge_dialects(answers):
    answers = [s if s != "fr-ca" else "fr" for s in answers]
    answers = [s if s != "pt-br" else "pt" for s in answers]
    answers = [s if s[:2] != "zh" else "zh" for s in answers]
    return answers
def main(total, corpus_size=20000, max_neural_features=8000, training_pct=.8):
    
    train_set, test_set, langs = dev_split(total, corpus_size, training_pct)
    scripts = s_id.labelled_scripts
    script_texts, script_labels = sort_by_script(train_set, scripts, langs)
    models = get_script_models(script_texts, script_labels, CountVectorizer, MultinomialNB)
    iber = ["pt", "pt-br", "gl", "es", "ca", "eo", "it"]
    fr = ["fr", "fr-ca", "ca", "oc"]
    zh = ["zh-tw", "zh-cn", "zh"] 
    balkan = ["mk", "sr", "bs", "hr", "sl", "bg"]
    norse = ["da", "nn", "nb", "is", "sv"]
    iber_id = multi_dialect_model(train_set,iber, max_features=max_neural_features)
    fr_id = multi_dialect_model(train_set, fr, max_features=max_neural_features)
    zh_id = multi_dialect_model(train_set, zh,max_features=max_neural_features)
    balkan_id = multi_dialect_model(train_set, balkan, max_features=max_neural_features)
    norse_id = multi_dialect_model(train_set, norse, max_features=max_neural_features)
   
    dialect_models = {"iber"  :iber_id, "zh": zh_id, "balkan": balkan_id, "fr":fr_id, "nor": norse_id}
    gold, predictions, lid_predictions = test(test_set, models, script_labels, dialect_models)
    report = classification_report(gold, predictions)
    lid_report = classification_report(gold, lid_predictions)
    print(report)
    print(lid_report)
    return models, script_labels, dialect_models, gold, predictions, lid_predictions, report


if __name__ == "__main__":
    total = pandas.read_csv('ted2020.tsv', sep='\t', keep_default_na=False, encoding='utf8', quoting=csv.QUOTE_NONE)
    models, script_labels, dialect_models, gold, predictions, lid_predictions, report = main(
        total, corpus_size=50000, max_neural_features=800, training_pct=.9)
    

import sys
# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import re
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords



from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.metrics import accuracy_score

import pickle

import matplotlib.pyplot as plt

def load_data(database_filepath):
    #engine = create_engine('sqlite:///'+database_filepath)
    engine = create_engine('sqlite:///{}'.format('DisasterResponse.db'))
    df = pd.read_sql_table('Message',engine)
    indexNames = df[~(df['related'] != 2)].index 
    df.drop(indexNames , inplace=True)
    X = df ['message']
    y = df.iloc[:, 4:]
    category_names = y.columns
    
    return X, y, category_names


def tokenize(text):
    """Tokenization function.
    Receives as input raw text
    Processes: normalization, stop words removal, stem and lemmatization.
    Returns tokenized text"""
    
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]"," ",text.lower())
    
    #stop words
    stop_words = stopwords.words("english")
    
    #tokenize
    words = word_tokenize(text)
    
    #stemming
    stemmed = [PorterStemmer().stem(w) for w in words]
    
    #lemmatizing
    words_lemmed = [WordNetLemmatizer().lemmatize(w) for w in stemmed if w not in stop_words]
    
    return words_lemmed


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf',  MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = { 
                'clf__estimator__n_estimators': [5]
              }
    
    model = GridSearchCV(pipeline, param_grid = parameters)
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=category_names))
    for  idx, cat in enumerate(Y_test.columns.values):
        print("{} -- {}".format(cat, accuracy_score(Y_test.values[:,idx], y_pred[:, idx])))
      
    print("accuracy = {}".format(accuracy_score(Y_test, y_pred)))


def save_model(model, model_filepath):
    filename = 'classifier.sav'
    pickle.dump(model, open(filename, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
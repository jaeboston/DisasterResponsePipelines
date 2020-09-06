import sys
import pandas as pd
import numpy as np
import re
import pickle
import nltk
nltk.download(['punkt', 'wordnet'])

from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def load_data(database_filepath):
    df = pd.read_sql_table('tbl_diaster_text', 'sqlite:///' + database_filepath)  
    X = df.loc[:,'message'] 
    Y = df.iloc[:, 4:] 
    category_names = Y.columns.values

    return X, Y, category_names

def tokenize(text):
    
    #: tokenize text
    tokens= word_tokenize(text)
    
    #: initiate Lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    #: iterate through each token
    clean_tokens = []
    for tok in tokens:
        
        #: Lemmatize, normalize case, and remove Leading/trailing wite space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens


def build_model():
    pipeline = Pipeline([
        ('vector', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    
    #: model  = pipeline
    y_predict = model.predict(X_test)
    print(classification_report(Y_test.values, y_predict, target_names=category_names))
    return

def save_model(model, model_filepath):
    
    with open(model_filepath, 'wb') as model_file:
        pickle.dump(model, model_file)
    return



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
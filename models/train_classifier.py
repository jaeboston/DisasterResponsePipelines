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
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

def load_data(database_filepath):
    '''
    Load data from a Database file
   
    Parameters:
    database_filepath (string) :  path to SQLite db
    
    Return:
        X (DataFrame) : feature data
        Y (DataFrame) : label data
        category_names (list) : categori names

    '''

    df = pd.read_sql_table('tbl_disaster_message', 'sqlite:///' + database_filepath)  
    X = df.loc[:,'message'] 
    Y = df.iloc[:, 4:] 
    category_names = Y.columns.values

    return X, Y, category_names


def tokenize(text):
    '''
    clean up input text by tokenizing,   lemmatizing and norormalizing  string
    
    Parameter:
    text (string) : String containing message for processing
       
    Return:
    clean_tokens (list of strings) :  List containing normalized and lemmatize word tokens
    
    '''

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

    '''
    Build model with GridSearch to find the optimized parameters
    
    Return:
    model (pipeline): model 
    
    '''

    pipeline = Pipeline([
        ('vector', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
    #    ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ('clf', MultiOutputClassifier(
            AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1, class_weight='balanced'))
        ))
    ])

    parameters ={
        'clf__estimator__learning_rate': [0.2, 0.3]
        #'clf__estimator__n_estimators': [10, 50]
        #'clf__estimator__min_samples_split': [2, 10],
        #'clf__estimator__min_samples_leaf': [1, 4]
        #'clf__estimator__max_depth': [20, None]
    }

    model = GridSearchCV(estimator=pipeline, param_grid=parameters, cv=3, n_jobs = -1, verbose = 3)

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    
    '''
    Evaluate model
    
    Parameter:
    model (pipeline) :  sklearn estimator model
    X_test (numpy.ndarray) :  message test data
    Y_test (numpy.ndarray) :  categories labeled data
    category_names (list) :  category names.
    
    Return:
    None
    '''

    #: model  = pipeline
    y_predict = model.predict(X_test)
    print(classification_report(Y_test.values, y_predict, target_names=category_names))
    return

def save_model(model, model_filepath):
    '''
    Save model into a file

    Parameter:
    model (pipeline) : sklearn estimator model
    model_filepath (string) : filename to save the model

    Return:
    None

    '''

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
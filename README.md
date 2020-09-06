# Disaster Response Project

## Building Machine Learning model to classify text messages


### Overview

This is an End-to-End Machine Learning project. Using disaster text message data from Figure Eight, I built a ML model for an API that classifies the text messages into several categories. This project involves building ETL process, modeling ML classifer, and developing a Flask Web app.

These are a few screenshots of the web app.

I used to the Notebooks to find/optimize the model used in the train_classifier.py file 

### Components

1. ETL Pipeline : a data cleaning pipeline that:

 * Loads the messages and categories datasets
 * Merges the two datasets
 * Cleans the data
 * Stores it in a SQLite database

2. ML Pipeline train_classifiery.py : a machine learning pipeline that:

 * Loads data from the SQLite database
 * Splits the dataset into training and test sets
 * Builds a text processing and machine learning pipeline
 * Trains and tunes a model using GridSearchCV
 * Outputs results on the test set
 * Exports the final model as a pickle file

3. Flask Web App


### Instructions
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Load the data set

    load the message and categoreis text data set

    Parameters:
    messages_filepath (string) : Description of message text file path 
    categories_filepath (string) : Description of categories text file path

    Returns:
    df (DataFrame) : merged data set of message and categoreis text

    '''

    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    messages.head()

    categories = pd.read_csv(categories_filepath)
    categories.head()

    # merge datasets
    df = messages.merge(categories, on='id')
    return df
    
def clean_data(df):
    '''
    Clean the data in the DataFrame

    performs various data cleansing steps including: 
        Splits the single 'category' column into indivdual columns
        Extract category names 
        Trim column values to indicator value then convert to int and set it eigher 0 or 1
        Drop any duplicate
    
    Parameters:
    df (DataFrame) :  message and categories text data
    
    Returns:
    df (DataFrame) : cleaned dataset 
    
    '''

    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    categories.head()

    # select the first row of the categories dataframe
    row = categories.iloc[0,:]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x[:-2])

    #: rename the columns of `categories`
    categories.columns = category_colnames


    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]          
        # convert column from string to numeric
        categories[column] = categories[column].astype(int) 

    #: make sure those values to be just 0 or 1
    for column in categories:
        if len(categories[column].value_counts())>2 :
            categories[column][categories[column]>1] = 1 
        
    
    # drop the original categories column from `df`
    df.drop(columns=['categories'], inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1, sort=False)

    # check number of duplicates
    sum(df.duplicated())

    # drop duplicates
    df = df.drop_duplicates()

    return df

def save_data(df, database_filename):
    '''
    Save data into database file

    Parameters:
    df (DataFrame) : DataFrame to be saved as a table into DB
    database_filename (string) : database file name to store the table

    Return:
    None

    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('tbl_disaster_message', engine, index=False,if_exists = 'replace')  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
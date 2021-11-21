import argparse
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    # read from files
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = messages.merge(categories, how='inner', on='id')
    return df


def clean_data(df):
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(pat=';', expand=True)
    
    # get meaningful column names
    row = categories.iloc[0]
    category_colnames = row.str.slice(stop=-2)
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.slice(start=-1)
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
    # drop the original categories column from `df`
    df = df.drop(['categories'], axis=1)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1, join="inner")
    df = df.drop_duplicates(subset='id')
    return df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('data', engine, index=False)  


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--messages", help="filepath of the messages", default='data/disaster_messages.csv')
    parser.add_argument("-c", "--categories", help="filepath of the categories", default='data/disaster_categories.csv')
    parser.add_argument("-o", "--database", help="filepath of the database to save the cleaned data",
                        default='data/disaster_processed.db')

    args = parser.parse_args()

    print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'.format(args.messages, args.categories))
    df = load_data(args.messages, args.categories)

    print('Cleaning data...')
    df = clean_data(df)

    print('Saving data...\n    DATABASE: {}'.format(args.database))
    save_data(df, args.database)

    print('Cleaned data saved to database!')


if __name__ == '__main__':
    main()

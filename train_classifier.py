import argparse
import re
import pickle
import pandas as pd
from sqlalchemy import create_engine
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# global model
lemmatizer = WordNetLemmatizer()


def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('data', engine)
    X = df.message
    Y = df.drop(['message', 'original', 'genre', 'id'], axis=1).to_numpy()
    category_names = df.drop(['message', 'original', 'genre', 'id'], axis=1).columns.tolist()

    X = X.tolist()

    return X, Y, category_names


def tokenize(text):
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)

    # Tokenize text
    tokens = text.split()

    # Remove stop words
    tokens = [w for w in tokens if w not in stopwords.words("english")]

    # Reduce words to their lemma
    tokens = [lemmatizer.lemmatize(w) for w in tokens]

    return tokens


def build_model():
    pipeline = Pipeline([
        ('text_pipeline', Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer(use_idf=True))
        ])),
        ('clf', MultiOutputClassifier(KNeighborsClassifier()))
    ])

    parameters = {
        'text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'clf__estimator__n_neighbors': [5, 10]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def display_results(y_test, y_pred):
    confusion_mat = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    print("\tConfusion Matrix:\n", confusion_mat)
    print("\tReport:\n", report)


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)

    # best hyper-parameters
    print("\nBest Parameters:", model.best_params_)

    for i in range(len(category_names)):
        print("\nResults for {}:".format(category_names[i]))
        display_results(Y_test[:, i], Y_pred[:, i])


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def load_model(model_filepath):
    return pickle.load(open(model_filepath, 'rb'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--database", help="filepath of the database to save the cleaned data",
                        default='data/disaster_processed.db')
    parser.add_argument("-m", "--model", help="filepath of the pickle file to save the model",
                        default='trained_models/model_v1.0.pkl')
    parser.add_argument("-p", "--phase", help="training or evaluating phase", choices=['train', 'test'],
                        default='train')
    # model training configuration
    parser.add_argument("-r", "--test_size", help="splitting ratio between train and test size", default=0.2)
    parser.add_argument("-s", "--state", help="random state seed", default=42)

    args = parser.parse_args()

    print('Loading data...\n\tDATABASE: {}'.format(args.database))
    X, Y, category_names = load_data(args.database)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=args.test_size, random_state=args.state)

    if args.phase == 'train':
        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)
    else:
        model = load_model(args.model)

    print('Evaluating model...')
    evaluate_model(model, X_test, Y_test, category_names)

    print('Saving model...\n\tMODEL: {}'.format(args.model))
    save_model(model, args.model)

    print('Trained model saved!')


if __name__ == '__main__':
    main()

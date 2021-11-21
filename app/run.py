import json
import plotly
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from flask import Flask
from flask import render_template, request
from plotly.graph_objs import Bar

from train_classifier import load_model, tokenize

app = Flask(__name__)


# load data
engine = create_engine('sqlite:///../data/disaster_processed.db')
df = pd.read_sql_table('data', engine)

# load model
model = load_model('../trained_models/model_v1.0.pkl')


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals
    # genre distribution
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    genre_counts, genre_names = zip(*sorted(zip(genre_counts, genre_names), key=lambda x: x[0], reverse=True))

    # category distribution
    category_df = df.drop(['message', 'original', 'genre', 'id'], axis=1)
    category_counts = category_df.sum(axis=0).tolist()
    category_names = category_df.columns.tolist()
    category_counts, category_names = zip(*sorted(zip(category_counts, category_names), key=lambda x: x[0],
                                                  reverse=True))

    # message length distribution
    bins = list(range(0, 401, 10)) + [12000]
    message_length = np.histogram(df.message.apply(len), bins=bins)[0]

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=bins,
                    y=message_length
                )
            ],

            'layout': {
                'title': 'Distribution of Message Lengths',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Number of Characters"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()

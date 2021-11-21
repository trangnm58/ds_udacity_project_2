# Project 2: Disaster Response Pipelines
Submission of Project 2 in Data Scientist Nanodegree Program - Udacity

# Installations
The code was developed using Python version 3.x. We recommend using [Anaconda](https://docs.anaconda.com/anaconda/install/index.html) for installing **Python 3** as well as other required libraries, although you can install them by other means.

Other required libraries that are not included in the Anaconda package:
- plotly==2.7.0
- sqlalchemy==1.4.13

Trained model can be downloaded [here](https://vnueduvn-my.sharepoint.com/:u:/g/personal/trangnm_58_vnu_edu_vn/EYCCuNj7knNHt6porsVtbCgBgh0rW8RZFVv4pLU_dHnxaw?e=JRlFMY), put it under ```trained_models/```

# Project Motivation
Following a disaster, typically there are millions of communications, either directly or via news and social media. Disaster response organizations might not have enough capacity to handle these messages manually. Furthermore, different organizations will take care of different parts of the problem. Therefore, an automatic disaster response classification system is needed so that the problems are handled fast and effectively by relevant parties.
In this project, the disaster response data prepared by [Figure Eight](https://www.figure-eight.com/) is used to train a multi-output machine learning model. The model is then deployed in a web application with intuitive interface and visualizations. 

# File Descriptions

Within the download you'll find the following directories and files:
```
./
├── data/
│   ├── disaster_categories.csv  # data to process 
│   ├── disaster_messages.csv  # data to process
│   └── disaster_processed.db   # database to save clean data to
├── trained_models/
│   └── model_v1.0.pkl  # saved model
├── app/
│   ├── template/
│   │	├── master.html  # main page of web app
│   │   └── go.html  # classification result page of web app
│   └── run.py  # Flask file that runs app
├── train_classifier.py
├── process_data.py
├── .gitignore
├── LICENSE.txt
└── README.md
```

# How to Use

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
    
       ```python process_data.py -m data/disaster_messages.csv -c data/disaster_categories.csv -o data/disaster_processed.db```
    - To run ML pipeline that trains classifier and saves

        ```python train_classifier.py -d data/disaster_processed.db -m trained_models/model_v1.0.pkl```

3. Run the following command in the app's directory to run your web app.
    
    ```python run.py```

4. Go to http://0.0.0.0:3001/

# Copyright and license
Code released under the [MIT License](https://github.com/trangnm58/ds_udacity_project_2/LICENSE.txt).
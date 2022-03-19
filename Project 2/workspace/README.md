# Disaster Response Pipeline Project

### Summary

This Project is part of Data Science Nanodegree Program by Udacity. The initial dataset contains pre-labelled tweet and messages from real-life disaster. The aim of the project is to build a Natural Language Processing tool that categorize messages. It allows people/organisations to find sources of messages - where are they comming from: News, media, social etc. and also allows to see the categories of comming messages - what are they main topic.

The Project is divided in the following Sections:

- Data Processing, ETL Pipeline to extract data from source, clean data and save them in a proper databse structure
- Machine Learning Pipeline to train a model able to classify text message in categories
- Web App to show model results in real time.


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.sav`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Files

app:
- template
    - master.html # main page of web app
    - go.html # classification result page of web app
- run.py # Flask file that runs app

data:
- disaster_categories.csv # data to process
- disaster_messages.csv # data to process
- process_data.py
- DisasterResponse.db # database to save clean data to

models:
- train_classifier.py

up:
- ETL Pipeline Preparation.ipynb # Jupyter Notebook for ETL
- ML Pipeline Preparation.ipynb # Jupyter Notebook for ML Pipeline
- classifier.sav # saved model
- README.md

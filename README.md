# Sentimeter

An app where users can check ratings or write reviews for a movie which are classified into either positive or negative using ML to get a score called "Sentimeter" for a movie. 
(Sentimeter score is percentage of positive sentiments received for a movie.)

### Contents:
- <b>Notebook</b> - Jupyter Notebook containing data preprocessing, training and evaluating part of the machine learning models.
- <b>FastAPI</b> - Implementation of API using FastAPI for deploying the saved model. Also contains Preprocessing Class that was used while preprocessing.

### Dataset:

- [IMDB Movie Review Dataset - Kaggle](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

### Summary of Implementation:

- Explored the given dataset.
- Preprocessed the text using various techniques.
- Trained various models on the preprocessed data and evaluated their performance.
- Selected the best performing model and saved it for deployment.
- Pickled few objects that were necessary for preprocessing.
- Used FastAPI to create an API that can be used to get predictions.
- Created a mobile app that contains few movies where users can see the sentimeter score for a movie or write reviews. The endpoint of the API for prediction was called to get sentiment of the reviews being posted, through which sentimeter score was calculated for each movie. 

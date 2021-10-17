# importing neccessary libraries
from fastapi import FastAPI
import uvicorn
import pandas as pd
import pickle
from pydantic import BaseModel
# importing preprocessing class
from TextProcessing import Preprocessing

# for fetching data
class GetReview(BaseModel):
    review: str

# creating app object
app = FastAPI()

# loading text preprocessing object and classification model
pre_processing = Preprocessing()
pre_processing.selector = pickle.load(open('selector.pkl', 'rb'))
pre_processing.vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
classifier = pickle.load(open('MNB_Model.sav', 'rb'))

# defining prediction end point
@app.post('/predict')
def predict_sentiment(review: GetReview):
    review = review.dict()
    input = pd.Series(review)

    # preprocessing input data
    input_cleaned, input_vectorized = pre_processing.preprocess(input, training=False)

    # getting prediction
    prediction = classifier.predict(input_vectorized)

    # mapping prediction to sentiment
    if prediction == 1:
        prediction = "positive"
    else:
        prediction = "negative"

    # returning the prediction
    return {'sentiment': prediction}


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000) 

# run the app using command:
# 1. .\mlproj\scripts\activate
# 2. uvicorn app:app --reload

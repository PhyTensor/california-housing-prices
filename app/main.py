import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
model = joblib.load('model.sav')

class HouseFeatures(BaseModel):
    median_income: float
    housing_median_age: float
    room_per_household: float
    bedrooms_per_room: float
    ocean_proximity: str # <1H OCEAN or INLAND


@app.get("/")
def root():
    return {"Hello": "World!"}


@app.post('/predict')
def predict(house_features: HouseFeatures):
    input_data = np.array([[
        house_features.median_income,
        house_features.housing_median_age,
        house_features.room_per_household,
        house_features.bedrooms_per_room,
        1 if house_features.ocean_proximity == '<1H OCEAN' else 0,
        1 if house_features.ocean_proximity == 'INLAND' else 0
        ]])

    prediction = model.predict(input_data)
    return {"predicted_price": round(prediction[0], 2)}


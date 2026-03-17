from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

app = FastAPI()

#load model pipeline
with open("RF_Model_pipeline.pkl",'rb') as file:
    pipe = pickle.load(file)


class HouseData(BaseModel):
    location:str
    total_sqft:float
    bath:int
    bhk:int

@app.post("/predict_price")
def predict(data:HouseData):
    input_data = pd.DataFrame([{
        "location": data.location,
        "total_sqft": data.total_sqft,
        "bath": data.bath,
        "bhk": data.bhk
    }])
    pred = pipe.predict(input_data)
    pred = pred[0]*100000
    return {"predicted_price":round(pred)}

@app.get("/cleaned_dataframe")
def get_data():
    df = pd.read_csv("cleaned.csv")
    return df.to_dict()

@app.get("/unique_locations")
def get_locations():
    df = pd.read_csv("cleaned.csv")
    return df["location"].unique().tolist()

@app.get("/sample_dataframe")
def get_sample_data(n_samples:int=20):
    df = pd.read_csv("cleaned.csv")
    return df.sample(n_samples).to_dict()
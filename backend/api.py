from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import os

app = FastAPI()

#load model pipeline
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pipeline_path = os.path.join(BASE_DIR,"..","backend","RF_Model_pipeline.pkl")
with open(pipeline_path,'rb') as file:
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
    file_path = os.path.join(BASE_DIR,'..','datasets','cleaned_dataset.csv')
    df = pd.read_csv(file_path)
    return df.to_dict()

@app.get("/unique_locations")
def get_locations():
    file_path = os.path.join(BASE_DIR,'..','datasets','cleaned_dataset.csv')
    df = pd.read_csv(file_path)
    return df["location"].unique().tolist()

@app.get("/sample_dataframe")
def get_sample_data(n_samples:int=20):
    file_path = os.path.join(BASE_DIR,'..','datasets','cleaned_dataset.csv')
    df = pd.read_csv(file_path)
    return df.sample(n_samples).to_dict()
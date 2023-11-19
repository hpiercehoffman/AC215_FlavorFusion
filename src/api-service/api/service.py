from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
import asyncio
import pandas as pd
import os
from fastapi import Request
from tempfile import TemporaryDirectory
from pydantic import BaseModel
from api import model_inference
from api import data_download

app = FastAPI(title="API Server", description="API Server", version="v1")

app.add_middleware(
    CORSMiddleware,
    allow_credentials=False,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_reviews(restaurant):
    text = df[df['Name'] == restaurant]['text']
    return list(text)[0]

class RestaurantRequest(BaseModel):
    restaurant: str

@app.get("/")
async def get_index():
    return {"message": "Welcome to the API Service"}


@app.get("/populate")
async def populate():
    global df

    small_file_path = data_download.download_reviews()
    df = pd.read_csv(small_file_path, index_col=0)

    return df['Name'].tolist()


@app.post("/predict")
async def predict(restaurant: RestaurantRequest):

    reviews = get_reviews(restaurant.restaurant)
    
    summary = model_inference.generate_summary(reviews)
    
    prediction_results = {
        "summary": summary
    }
    
    return prediction_results






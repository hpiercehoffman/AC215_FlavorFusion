from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
import asyncio
import pandas as pd
import os
from fastapi import Request
from tempfile import TemporaryDirectory
from pydantic import BaseModel
from api import model_inference

app = FastAPI(title="API Server", description="API Server", version="v1")

df = pd.read_csv('./combined-data-combined_Massachusetts_small.csv', index_col=0)

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

# class Reviews(BaseModel):
#     reviews: str

class RestaurantRequest(BaseModel):
    restaurant: str

@app.get("/")
async def get_index():
    return {"message": "Welcome to the API Service"}

@app.get("/populate")
async def populate():
    return df['Name'].tolist()


@app.post("/predict")
async def predict(restaurant: RestaurantRequest):

    reviews = get_reviews(restaurant.restaurant)
    
    summary = model_inference.generate_summary(reviews)
    
    prediction_results = {
        "summary": summary
    }
    
    return prediction_results

# @app.post("/predict")
# async def predict(reviews: Reviews):
    
#     summary = model_inference.generate_summary(reviews.reviews)
    
#     prediction_results = {
#         "summary": summary
#     }
    
#     return prediction_results





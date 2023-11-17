from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
import asyncio
import pandas as pd
import os
from fastapi import File
from tempfile import TemporaryDirectory

app = FastAPI(title="API Server", description="API Server", version="v1")

app.add_middleware(
    CORSMiddleware,
    allow_credentials=False,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def get_index():
    return {"message": "Welcome to the API Service"}

@app.post("/predict")
async def predict(reviews: str):
    reviews_list = reviews.split("|||||")
    
    prediction_results = {
        "num_reviews": len(reviews_list),
        "placeholder_summary": "This is a placeholder"
    }
    
    return prediction_results





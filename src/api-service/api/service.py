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

app.add_middleware(
    CORSMiddleware,
    allow_credentials=False,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Reviews(BaseModel):
    reviews: str

@app.get("/")
async def get_index():
    return {"message": "Welcome to the API Service"}


@app.post("/predict")
async def predict(reviews: Reviews):
    
    summary = model_inference.generate_summary(reviews.reviews)
    
    prediction_results = {
        "summary": summary
    }
    
    return prediction_results





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

races = ['Western European', 'African', 'South Asian', 'Jewish', 'East Asian']
# Make this dataframe global so we can use it in predict function
#global df

# Download data file from GCS bucket and get its path
small_file_path = data_download.download_reviews()
df = pd.read_csv(small_file_path, index_col=0)

def get_reviews(restaurant):
    """Helper function to get all reviews for a specific restaurant"""
    df_res = df[df['Name'] == restaurant]

    reviews = []
    info_races = []
    for race in races:
        race_df = df_res[df_res['race'] == race]
        if race_df.shape[0] > 0:
            race_df = race_df.groupby(["Name", "address"]).agg({"text": "|||||".join}).reset_index()
            text = race_df['text']
            reviews.append(list(text)[0])
            info_races.append(race)
    return reviews, info_races

class RestaurantRequest(BaseModel):
    """Pydantic object so we can send strings in correct format for API"""
    restaurant: str

@app.get("/")
async def get_index():
    """Default get method"""
    return {"message": "Welcome to the API Service"}


@app.get("/populate")
async def populate():
    """Populate the dropdown menu with list of restaurants"""

    # Return names of all restaurants
    return df['Name'].unique().tolist()


@app.post("/predict")
async def predict(restaurant: RestaurantRequest):
    """Run model inference to generate summary of reviews for chosen restaurant"""

    reviews, races_info = get_reviews(restaurant.restaurant)
    # We can choose whether to use our finetuned model or the original model 
    # trained on multi-news summarization
    summaries = model_inference.generate_summary(reviews, use_finetuned = True)

    race_summary_dict = {race: 'Not enough reviews' for race in races}

    for race_info, summary in zip(races_info, summaries):
        race_summary_dict[race_info] = summary

    prediction_results = {
        "summary": summaries[0]
    }
    
    return prediction_results






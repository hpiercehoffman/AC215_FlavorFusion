import json
import gc
import pandas as pd
import argparse
import shutil
import glob
import os

from tqdm import tqdm
from itertools import islice

from google.cloud import translate_v2 as translate
from google.cloud import storage

GCS_BUCKET_NAME = os.environ["GCS_BUCKET_NAME"]

"""
Load specified lines from an input file (useful since the dataset is large and 
we don't want to store all of it in memory at once).
"""
def load_jsons(filepath, start_line, stop_line):
    print("Loading lines " + str(start_line) + " to " + str(stop_line) + " from " + str(filepath))
    with open(filepath, 'r') as fp:
        json_lines = islice(fp, start_line, stop_line)
        review_jsons = [json.loads(line) for line in json_lines]
    return review_jsons

"""
Helper function to join review tokens into a single string.
"""
def join_tokens(token_list):
    return "".join(token_list)

"""
Call the Google Cloud Translate API to translate a single review-summary pair.
"""
def translate_json(json, translator):
    item_id = json["item_id"]
    summary = join_tokens(json["hq_tokens"])
    reviews = json["lq_tokens_list"]
    reviews = [join_tokens(review) for review in reviews]
    translated_summary = translator.translate(summary, target_language="en")["translatedText"]
    translated_reviews = translator.translate(reviews, target_language="en")
    translated_reviews = [review["translatedText"] for review in translated_reviews]
    return item_id, translated_reviews, translated_summary

"""
Translate a list of review-summary JSONs and create a dataframe with the results.
"""
def translate_reviews(review_jsons):
    ids, reviews, summaries = [], [], []
    translator = translate.Client()
    print("Now translating reviews...")
    for json in tqdm(review_jsons):
        item_id, translated_reviews, translated_summary = translate_json(json, translator)
        ids.append(item_id)
        reviews.append(translated_reviews)
        summaries.append(translated_summary)
    result_df = pd.DataFrame({"id": ids, "reviews": reviews, "summary": summaries})
    result_df["review_str"] = result_df["reviews"].apply(lambda x: "|||||".join(x))
    result_df = result_df[["id", "summary", "review_str"]]
    return result_df

"""
Download raw data from GCS bucket. Use this function if working in a VM without a mounted 
bucket, or if you would like to get a clean copy of the raw data.
"""
def download_data():
    bucket_name = GCS_BUCKET_NAME
    print("Downloading data from " + str(bucket_name))
    
    # Clear existing dataset folders so we get a clean copy
    dataset_folder = "data/chinese_data"
    shutil.rmtree(dataset_folder, ignore_errors=True, onerror=None)
    os.makedirs(dataset_folder, exist_ok=True)

    # Initiate storage client and download data
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix="chinese_data/")
    for blob in blobs:
        print("Data file:", blob.name)
        if not blob.name.endswith("chinese_data/"):
            filename = os.path.basename(blob.name)
            local_file_path = os.path.join(dataset_folder, filename)
            blob.download_to_filename(local_file_path)

def main(args):
    if args.download:
        download_data()
        
    review_jsons = load_jsons(args.reviews_file_path, int(args.start_line), int(args.stop_line))
    result_df = translate_reviews(review_jsons)
    result_df.to_csv(args.output_file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LSARS dataset preprocessing')
    parser.add_argument('--reviews_file_path', type=str, help='Path of the reviews file')
    parser.add_argument('--start_line', type=str, help='File line where processing should start')
    parser.add_argument('--stop_line', type=str, help='File line where processing should stop')
    parser.add_argument('--output_file_path', type=str, help='Filepath to output the results')
    parser.add_argument('-d','--download', action="store_true", help="Download raw data from GCS bucket",)
    args = parser.parse_args()
    
    main(args)


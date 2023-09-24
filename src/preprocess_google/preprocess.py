import pandas as pd
from tqdm import tqdm
import argparse
import json
import numpy as np
import os
import shutil
from utils import parse, RESTS
from google.cloud import storage

GCS_BUCKET_NAME = os.environ["GCS_BUCKET_NAME"]

"""
Download raw data from GCS bucket. Use this function if working in a VM without a mounted 
bucket, or if you would like to get a clean copy of the raw data.
"""
def download_data():
    bucket_name = GCS_BUCKET_NAME
    print("Downloading data from " + str(bucket_name))
    
    # Clear existing dataset folders so we get a clean copy
    dataset_folder = "data/raw-data"
    shutil.rmtree(dataset_folder, ignore_errors=True, onerror=None)
    os.makedirs(dataset_folder, exist_ok=True)

    # Initiate storage client and download data
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix="raw-data/")
    for blob in blobs:
        if not blob.name.endswith("raw-data/"):
            print("Downloading data file:", blob.name)
            filename = os.path.basename(blob.name)
            local_file_path = os.path.join(dataset_folder, filename)
            blob.download_to_filename(local_file_path)
            if 'meta' in local_file_path:
                metadata_file_path = local_file_path
            elif 'review' in local_file_path: 
                reviews_file_path = local_file_path
    return reviews_file_path, metadata_file_path

def upload_data(output_file_path):
    bucket_name = GCS_BUCKET_NAME
    print("Uploading data to " + str(bucket_name))

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    output_file_name = os.path.basename(output_file_path)
    destination_blob_name = os.path.join('combined-data', output_file_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(output_file_path)

def make_metadata_df(fl):
    parser = parse(fl)
    rest_records = []
    print('Processing metadata')
    for record in parser:
        if record['category'] != None:
            if not set(record['category']).isdisjoint(RESTS):
                rest_records.append([record['name'],
                                     record['gmap_id'],
                                     record['address'],
                                     record['avg_rating'],
                                     record['relative_results'],
                                     record['num_of_reviews']])
    
    df = pd.DataFrame(rest_records, columns=['Name', 'gmap_id', 'address', 'avg_rating', 
                                             'relative_results', 'num_of_reviews'])
    return df

def make_reviews_df(fl, min_char=0, max_char=10000):
    parser = parse(fl)
    reviews = []
    print('Processing reviews data')
    for review in parser:
        if review['text'] != None:
            if len(review['text']) >= min_char and len(review['text']) < max_char:
                reviews.append([review['name'],
                                review['rating'],
                                review['text'],
                                review['gmap_id']
                               ])
    df = pd.DataFrame(reviews, columns=['name', 'rating', 'text', 'gmap_id'])
    return df

def main(args):
    if args.download:
        reviews_file_path, metadata_file_path = download_data()
    else:
        reviews_file_path, metadata_file_path = args.reviews_file_path, args.metadata_file_path

    reviews_df = make_reviews_df(reviews_file_path, min_char=args.min_char, max_char=args.max_char)
    meta_df = make_metadata_df(metadata_file_path)
    
    meta_df = meta_df[meta_df['num_of_reviews'] < args.max_num_reviews]
    combined_df = reviews_df.merge(meta_df, on="gmap_id", how="inner")
    
    sub_df = combined_df.loc[:,['text', 'Name', 'address']]
    sub_df = sub_df.groupby(["Name", "address"]).agg({"text": "|||||".join}).reset_index()

    sub_df.to_csv(args.output_file_path)

    if args.upload:
        upload_data(args.output_file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Google Review dataset preprocessing')
    parser.add_argument('--download', action="store_true", help="Download raw data from GCS bucket")
    parser.add_argument('--reviews_file_path', type=str, help='Path of the reviews file')
    parser.add_argument('--metadata_file_path', type=str, help='Path of the business metadata file')
    parser.add_argument('--output_file_path', type=str, help='Path of the reviews file')
    parser.add_argument('--upload', action="store_true", help="Upload output data to GCS bucket")
    parser.add_argument('--min_char', type=int, default=0, help='Minimum number of characters in each review')
    parser.add_argument('--max_char', type=int, default=1000000, help='Minimum number of characters in each review')
    parser.add_argument('--max_num_reviews', type=int, default=1000, help='Maximum number of reviews for each business')
    args = parser.parse_args()
    
    main(args)

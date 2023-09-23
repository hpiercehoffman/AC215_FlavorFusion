"""
Module that contains the command line app.
"""
import argparse
import os
import traceback
import time
from google.cloud import storage
import shutil
import glob
import json


GCS_BUCKET_NAME = os.environ["GCS_BUCKET_NAME"]


def download_data():
    print("download_data")

    bucket_name = GCS_BUCKET_NAME

    # Clear dataset folders
    dataset_folder = "data/chinese_data"
    shutil.rmtree(dataset_folder, ignore_errors=True, onerror=None)
    os.makedirs(dataset_folder, exist_ok=True)

    # Initiate Storage client
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix="chinese_data/")

    # Download data
    for blob in blobs:
        print("Data file:", blob.name)

        if not blob.name.endswith("chinese_data/"):
            filename = os.path.basename(blob.name)
            local_file_path = os.path.join(dataset_folder, filename)
            blob.download_to_filename(local_file_path)


def main(args=None):
    if args.download:
        download_data()


if __name__ == "__main__":
    # Generate the inputs arguments parser
    # if you type into the terminal 'python cli.py --help', it will provide the description
    parser = argparse.ArgumentParser(description="Data Versioning CLI...")

    parser.add_argument(
        "-d",
        "--download",
        action="store_true",
        help="Download raw data from a GCS Bucket",
    )

    args = parser.parse_args()

    main(args)
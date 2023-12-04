import os
import shutil
from google.cloud import storage

GCS_BUCKET_NAME = os.environ["GCS_BUCKET_NAME"]

def download_reviews():
    """
    Download raw data from GCS bucket. We use a hardcoded data file name in this
    prototype implementation.
    """
    bucket_name = GCS_BUCKET_NAME
    print("Downloading data from " + str(bucket_name))
    
    # Initiate storage client and download data to the persistent folder
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    local_file_path = "../persistent/new-Massachusetts_race.csv"
    blob = bucket.blob("combined-data/new-Massachusetts_race.csv")
    blob.download_to_filename(local_file_path)
     
    small_file_path = local_file_path
    return small_file_path

    

import os
import shutil
from google.cloud import storage

GCS_BUCKET_NAME = os.environ["GCS_BUCKET_NAME"]

def download_reviews():
    """
    Download raw data from GCS bucket. Use this function if working in a VM without a mounted 
    bucket, or if you would like to get a clean copy of the raw data.
    """
    bucket_name = GCS_BUCKET_NAME
    print("Downloading data from " + str(bucket_name))
    
    # Initiate storage client and download data
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    local_file_path = "../persistent/combined-data-combined-Massachusetts_small.csv"

    blob = bucket.blob("combined-data/combined-data-combined_Massachusetts_small.csv")
    blob.download_to_filename(local_file_path)
     
    small_file_path = local_file_path
    return small_file_path, small_file_path


if __name__ == "__main__":
    download_reviews()
    

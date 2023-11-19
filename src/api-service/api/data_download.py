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
    
    # Clear existing dataset folders so we get a clean copy
    dataset_folder = "./"
    #shutil.rmtree(dataset_folder, ignore_errors=True, onerror=None)
    os.makedirs(dataset_folder, exist_ok=True)

    # Initiate storage client and download data
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix="combined-data//")
    for blob in blobs:
        print(blob.name)
        if not blob.name.endswith("combined-data//"):
            print("Downloading data file:", blob.name)
            filename = os.path.basename(blob.name)
            local_file_path = os.path.join(dataset_folder, filename)
            blob.download_to_filename(local_file_path)
            if 'small' in local_file_path:
                small_file_path = local_file_path
            else:
                large_file_path = local_file_path
    return small_file_path, large_file_path


if __name__ == "__main__":
    download_reviews()
    

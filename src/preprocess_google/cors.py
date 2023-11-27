import os
import traceback
import time
from google.cloud import storage
from label_studio_sdk import Client

GCS_BUCKET_NAME = os.environ["GCS_BUCKET_NAME"]
LABEL_STUDIO_URL = os.environ["LABEL_STUDIO_URL"]

def set_cors_configuration():
    """Set a bucket's CORS policies configuration."""

    print("set_cors_configuration()")
    bucket_name = GCS_BUCKET_NAME

    # Initiate Storage client
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    bucket.cors = [
        {
            "origin": ["*"],
            "method": ["GET"],
            "responseHeader": ["Content-Type", "Access-Control-Allow-Origin"],
            "maxAgeSeconds": 3600,
        }
    ]
    bucket.patch()

    print(f"Set CORS policies for bucket {bucket.name} is {bucket.cors}")
    return bucket

bucket = set_cors_configuration()
print(bucket)
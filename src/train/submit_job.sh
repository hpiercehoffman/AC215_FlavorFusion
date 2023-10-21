export UUID=$(openssl rand -hex 6)
export GCP_REGION="us-central1" 

gcloud ai custom-jobs create \
--region=$GCP_REGION \
--display-name=$DISPLAY_NAME \
--config=config.yaml
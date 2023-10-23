export UUID=$(openssl rand -hex 6)
export DISPLAY_NAME="primera_training_pruned_$UUID"
export GCP_REGION="us-east1" 

export CMDARGS="--input_dir=./,\
--model_output_path=./,\
--max_source_length=700,\
--max_target_length=100,\
--wandb,--lr=2e-6,\
--download,\
--num_train_epochs=1,\
--k_top_longest=5,\
--test_ratio=0.1,\
--wandb_key=$WANDB_KEY,\
--gcs_bucket_name=lsars-data,\
--num_processes=4,\
--wandb_download_folder=flavorfusion-team/FlavorFusion/model-w10g07vv:v0,\
--quantize"

gcloud ai custom-jobs create \
--region=$GCP_REGION \
--display-name=$DISPLAY_NAME \
--config=config.yaml \
--args=$CMDARGS
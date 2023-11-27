export UUID=$(openssl rand -hex 6)
export DISPLAY_NAME="primera_finetuning_$UUID"
export GCP_REGION="us-central1" 

export CMDARGS="--input_dir=./,\
--model_output_path=./,\
--max_source_length=4096,\
--max_target_length=512,\
--wandb,\
--lr=2e-7,\
--download,\
--num_train_epochs=5,\
--k_top_longest=5,\
--test_ratio=0.05,\
--wandb_key=$WANDB_KEY,\
--gcs_bucket_name=reviews-data,\
--num_processes=4"

gcloud ai custom-jobs create \
--region=$GCP_REGION \
--display-name=$DISPLAY_NAME \
--config=config.yaml \
--args=$CMDARGS
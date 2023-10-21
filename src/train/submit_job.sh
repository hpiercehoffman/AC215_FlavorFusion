export UUID=$(openssl rand -hex 6)
export DISPLAY_NAME="primera_training_$UUID"
# export MACHINE_TYPE="g2-standard-4"
# export REPLICA_COUNT=1
# export EXECUTOR_IMAGE_URI="us-central1-docker.pkg.dev/flavor-fusion-399619/flavor-fusion/cuda-train:latest"
# export ACCELERATOR_TYPE="NVIDIA_L4"
# export ACCELERATOR_COUNT=1
export GCP_REGION="us-central1" 

# export CMDARGS="--input_dir=./ \
# --model_output_path=./ \
# --max_source_length=4096 \
# --max_target_length=1024 \
# --wandb \
# --download \
# --num_train_epochs=20 \
# --k_top_longest=5 \
# --test_ratio=0.1 \
# --wandb_key=$WANDB_KEY
# --gcs_bucket_name=lsars-data \
# --num_processes=4 \
# --model_name=allenai/PRIMERA \
# --prune"

# gcloud ai custom-jobs create \
#   --region=$GCP_REGION \
#   --display-name=$DISPLAY_NAME \
#   --worker-pool-spec=machine-type=$MACHINE_TYPE \
#   --replica-count=$REPLICA_COUNT \
#   --accelerator-type=$ACCELERATOR_TYPE \
#   --accelerator-count=$ACCELERATOR_COUNT \
#   --executor-image-uri=$EXECUTOR_IMAGE_URI \
#   --args=$CMDARGS

gcloud ai custom-jobs create \
--region=$GCP_REGION \
--display-name=$DISPLAY_NAME \
--config=config.yaml
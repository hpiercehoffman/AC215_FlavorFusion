AC215: Milestone 3
==============================

Project Organization
------------
      ├── LICENSE
      ├── README.md
      ├── notebooks
      ├── references
      ├── reports
      │   └── milestone2.md
      ├── requirements.txt
      └── src
            ├── docker-volumes
            │   ├── google-data.dvc
            │   ├── lsars-data.dvc
            │   └── notebooks
            │       ├── process_google.ipynb
            │       ├── process_lsars.ipynb
            │       └── Evaluation_Example.ipynb
            ├── preprocess_google
            │   ├── Dockerfile
            │   ├── Pipfile
            │   ├── Pipfile.lock
            │   ├── preprocess.py
            │   ├── cors.py
            │   ├── utils.py
            │   ├── docker-compose.yml
            │   └── docker-shell.sh	
            ├── preprocess_lsars
            │   ├── Dockerfile
            │   ├── Pipfile
            │   ├── Pipfile.lock
            │   ├── preprocess_lsars.py
            │   ├── docker-compose.yml
            │   └── docker-shell.sh
            └── train
                ├── Dockerfile
                ├── Pipfile
                ├── Pipfile.lock
                ├── train.py
                ├── docker-compose.yml
                ├── docker-shell.sh
                ├── package-trainer.sh
                └── package
                    ├── setup.cfg
                    ├── setup.py
                    ├── PKG-INFO
                    └── trainer
                        ├── __init__.py
                        └── task.py

--------
# Milestone 3 Overview

**Team Members**   
Varun Ullanat, Hannah Pierce-Hoffman

**Group Name**   
FlavorFusion

**Project**   
In this project, we aim to build an app that captures cultural differences in Google resturaunt reviews using abstractive summaries generated by a large language model.

For a full list of external references used in this project, please refer to our [reference document](https://github.com/hpiercehoffman/AC215_FlavorFusion/blob/milestone3/references/references.md).

## Milestone 3 Contents ##

The focus of this milestone is model training. For information on data preprocessing, see our [Milestone 2 report](https://github.com/hpiercehoffman/AC215_FlavorFusion/blob/milestone3/reports/milestone2.md). Data preprocessing takes place in the **preprocess_google** and **preprocess_lsars** Docker containers.

### train ###
This container is responsible for training the [PRIMERA model](https://github.com/allenai/PRIMER) on the [LSARS dataset](https://github.com/ScarletPan/LSARS). 

The inputs to this container are the following:
- 5,000 preprocessed and translated LSARS data points which we generated for Milestone 2, each of which contains:   
    - A unique identifier   
    - 10-40 product reviews   
    - One summary review which captures information from all of the reviews
- Training parameters
- Secrets file (contains service account credentials for Google Cloud)

The output of this container is a file (`pytorch_model.bin`) containing trained model weights. The weight file for each training run is automatically uploaded to Weights and Biases (WandB) at the end of the run.

This container holds the following files:
1. `src/train/train.py` This script trains the PRIMERA model on the LSARS dataset. The script provides options for data augmentation, data streaming, and data downloading from our GCP bucket, as well as standard training parameters such as learning rate and batch size. WandB tracking is integrated into the script, including model upload to WandB at the end of a training run. In our initial runs shown in this milestone, we use pre-trained weights from a version of PRIMERA trained on the [Multi-News dataset](https://huggingface.co/datasets/multi_news). This allows us to leverage transfer learning. In later training runs, we will experiment with using pre-trained weights for only some layers of the model. The PRIMERA model is [hosted](https://huggingface.co/allenai/PRIMERA/tree/main) on Huggingface.
2. `src/train/Dockerfile` This docker sets up a `pipenv` virtual environment for model training. In the file, we use our service account to connect to Google Cloud. We then create a user named `app` and install the required python packages from `src/preprocess_google/Pipfile`. Key packages used in model training include [Pytorch](https://pytorch.org/), [Transformers](https://huggingface.co/docs/transformers/index), and [NLTK](https://www.nltk.org/).
3. `src/train/docker-shell.sh` This script sets up a network so the Docker container can communicate with the outside world via ports. Next, the script builds our Docker container and uses `src/train/docker-compose.yml` to set up the full container environment. In `src/train/docker-compose.yml`, we specify GPU capabilities so the Docker container can take advantage of running on a VM with a GPU.
4. `src/train/package-trainer.sh` This script creates a Python package for serverless training. We ultimately were not able to get a quota request approved for serverless training, but we set up the required infrastructure. See the **Serverless Training** section below for details.

We ran this container inside a GCP VM with a Nvidia L4 GPU. Due to the large model size, an L4 GPU with 24GB of GPU RAM is required for training. We provide detailed instructions for VM setup and model training in the [Setup Notes](https://github.com/hpiercehoffman/AC215_FlavorFusion/edit/milestone3/README.md#setup-notes) section below.

### docker-volumes ###
This directory is mounted to the `train` Docker container in `src/train/docker-compose.yml`. The `docker-volumes/training-data` directory appears inside the Docker container as `/app/data`. Users may specify this directory as a data download and/or model output directory via parameters to the `train.py` script. This permits the Docker container to access the filesystem of the host VM, allowing downloaded data files and model result files to be saved on the VM even after the Docker container is no longe running.

### notebooks ###    
This directory is currently empty, but will be used to store code which doesn't belong to a specific container, such as reports, insights, or visualizations.

### reports ###
This directory contains our reports from past milestones:
- [Milestone 2](https://github.com/hpiercehoffman/AC215_FlavorFusion/blob/milestone3/reports/milestone2.md): Data preprocessing, Label Studio, and DVC.

### references ###  
This directory contains information on models, datasets, and other external references used in this project. References are detailed in [references.md](https://github.com/hpiercehoffman/AC215_FlavorFusion/blob/milestone3/references/references.md).

## Experiment Tracking ##
Our Weights and Biases page contains two completed training runs, each lasting 10-12 hours. The PRIMERA model has a long training time when run on a single GPU, so each of these runs only covers a few epochs. We will use longer training runs to generate a final set of model weights for the LSARS dataset.

Below we show a view of the WandB dashboard for evaluation metrics. Our most relevant evaluation metric is the (ROUGE score)[https://clementbm.github.io/theory/2021/12/23/rouge-bleu-scores.html#rouge], a NLP metric which measures similarity between the input reviews and the generated summary.

![Evaluation metrics](https://github.com/hpiercehoffman/AC215_FlavorFusion/blob/milestone3/images/eval_metrics.png)

We also show the loss curves on the training set. We use WandB's plotting features to apply mild smoothing, making it easier to observe the overall trend of the loss curves.

![Training loss curves](https://github.com/hpiercehoffman/AC215_FlavorFusion/blob/milestone3/images/train_loss_smoothed.png)

## Serverless Training ##
We made a total of 9 GPU quota increase requests for Vertex AI API. These requests were made from two different projects over a five-day period, and included multiple US regions, GPU types, and quota values. We tried using a personal credit card as well as the APCOMP course credits. We also contacted Google support by email to follow up. However, none of our GPU quota increase requests have been approved at the time of this submisssion.

We have created some basic infrastructure for serverless training, as shown in the `src/train/package` directory. This directory contains package metadata as well as a version of the training script which takes our WandB key as a command line argument. We also created a `src/train/package-trainer.sh` to package the training code and upload it to a GCS bucket.

If approved for our GPU quotas, we could use the [Vertex AI console](https://console.cloud.google.com/vertex-ai/training/training-pipelines?project=flavor-fusion-399619) to submit a job based on this package. We tested this method, and the job submits successfully, but then fails because there is no available GPU quota. We could also use a command line script to submit jobs, as shown in the [in-class demo](https://github.com/dlops-io/model-training/tree/main).

The screenshot below shows our failed serverless training job as well as the reason - insufficient quota.

![Serverless training failure](https://github.com/hpiercehoffman/AC215_FlavorFusion/blob/milestone3/images/serverless_no_quota.png)

## Data Pipeline ##
We built our data pipeline in Milestone 2. Our data pipeline uses DVC and Label Studio, as well as preprocessing scripts, to convert data from the **LSARS** and **Google Reviews** datasets into a convenient format for model training. For more information on our data pipeline, including examples showing how to download specific data versions, please see the [Setup Notes](https://github.com/hpiercehoffman/AC215_FlavorFusion/blob/milestone3/reports/milestone2.md#setup-notes) section of our Milestone 2 report.

--------
# Setup Notes #

### GCP VM setup ###

After initial tests in our local environment, we deployed our training script on a GCP VM. We used machine type `g2-standard-4` with the following specs:
- 1 Nvidia L4 GPU with 24GB of GPU RAM
- 4 vCPUs
- 16GB CPU RAM
- 200GB storage space (adjustable)
- Debian OS with CUDA and Docker pre-installed (boot option offered for GPU VMs)

We selected a **spot VM** for affordability. Setting up the VM is fairly straightforward. We followed these steps:
- Create the VM from Google Cloud Console - Compute Engine. Wait for the instance to start.
- Generate SSH keys for Google Cloud on local machine: `gcloud compute config-ssh`
- SSH to the VM from local machine: `gcloud compute ssh --project <our_project_id> --zone us-central1-a <vm_instance_name> -- -L 8080:localhost:8080`
- Install Nvidia drivers when prompted. Run `nvidia-smi` to verify GPU is available.
- Once connected to the VM, clone our [project repo](https://github.com/hpiercehoffman/AC215_FlavorFusion/tree/milestone3) and checkout the `milestone3` branch.
- The OS comes with Docker pre-installed, but not Docker Compose. Therefore, we need to follow [these instructions](https://docs.docker.com/engine/install/debian/#install-using-the-repository) for installing Docker packages on Debian.
- Build the Docker container for model training on the VM.
- To allow model training to run without keeping the SSH connection open, launch a [tmux session](https://tmuxcheatsheet.com/). Tmux is a "terminal multiplexer" which decouples a session from the terminal window, allowing it to run in the background. The command to launch a session is `tmux new -s <session_name>`. 
- You should now be inside your tmux session. From here, launch the training container. Inside the container, run `wandb login`. Enter [wandb API key](https://docs.wandb.ai/quickstart) when prompted.
- You are now ready to run the model training script (see below for arguments). Once the script is running, use `Ctrl b + d` to detach from the tmux session. You can now exit the SSH connection and the model will continue to train as long as the VM is still running.

### Running the training Docker container ###   
The Dockerfile for model training can be run via the `docker-shell.sh` script inside the container. To run this container, do the following:
- Clone the repository and checkout the `milestone3` branch
- `cd src/train`
- `chmod 777 docker-shell.sh`
- `./docker-shell.sh`
The Dockerfile for model training will build and you will be dropped into a shell prompt as `app` user. From there, you can kick off a model training run using the `train.py` script.

### Running train.py script ###
To run this script, you must activate the `train` docker container. Once you are inside the container, you can run the script with the following arguments:
- `--input_dir`: Path to directory containing the preprocessed LSARS data files. If the `--download` flag is specified, files will be downloaded from our GCS bucket to this repository.
- `--model_output_path`: Path to directory where model weights and reports will be saved at the end of the training run. If using WandB, model outputs will also be uploaded to our WandB project.
- `--download`: Flag to indicate that the preprocessed LSARS data should be downloaded from our GCS bucket before model training. This flag is needed if running the training script for the first time in a new GCP VM.
- `--wandb`: Flag to indicate whether WandB should be used for logging. If this flag is active, you will be prompted to enter a WandB API key at the start of the run. Alternately, you can run `wandb login` before running the training script.
- `--streaming`: Whether to stream data during model training. Streaming uses the [streaming functionality](https://huggingface.co/docs/datasets/stream) of Huggingface Transformers. This functionality is similar to TF Data.
- `--test_ratio`: Proportion of the dataset to use for evaluation.
- `--k_top_longest`: Maximum number of reviews to use from each data point. To decrease the demand on GPU RAM, we subsample the reviews from each data point. When subsampling, we preferentially select the longest reviews, since these are more likely to be high-quality reviews which are reflected in the summary review.
- `--max_docs_per_review`: Data augmentation option for splitting each data point into multiple new data points. For example, if `k_top_longest` is set to 20 and `max_docs_per_review` is set to 5, each data point will be subsampled to 20 reviews, then split into 4 new data points, each of which contains 5 reviews and one summary. To run without data augmentation, set `k_top_longest` and `max_docs_per_review` to the same value.
- `--num_processes`: Number of processes used when constructing the Dataset object from the preprocessed data files.
- `--max_source_length`: Maximum number of tokens for each set of reviews being summarized. This number refers to the length of the entire review group, rather than the length of individual reviews in the group. Tokens beyond this length will be truncated.
- `--max_target_length`: Maximum number of tokens for each summary review. Tokens beyond this length will be truncated.
- `--subset_dataset_to`: Number of data points to use if not training on the entire dataset; this is primarily a debugging option when checking if the training script works.
- `--lr`: Learning rate for training. We train with a lower learning rate when using pre-trained weights, since the model doesn't have to start from a random initialization.
- `--batch_size`: Number of data points to be processed in each batch.
- `--num_train_epochs`: Number of epochs to train for.

When training on a single Nvidia L4 GPU with a batch size of 1 and all data points subsampled to the longest 5 reviews with no data augmentation, each epoch takes about 2.25 hours.

# References #

For a full list of external references used in this project, please refer to our [reference document](https://github.com/hpiercehoffman/AC215_FlavorFusion/blob/milestone3/references/references.md).



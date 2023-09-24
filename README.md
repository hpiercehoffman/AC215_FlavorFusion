AC215: Milestone 2
==============================

Project Organization
------------
      ├── LICENSE
      ├── README.md
      ├── notebooks
      ├── references
      ├── requirements.txt
      └── src
            ├── docker-volumes
            │   ├── google-data.dvc
            │   ├── lsars-data.dvc
            │   └── notebooks
            │       ├── process_google.ipynb
            │       └── process_lsars.ipynb
            ├── preprocess_google
            │   ├── Dockerfile
            │   ├── Pipfile
            │   ├── Pipfile.lock
            │   ├── preprocess.py
            │   ├── cors.py
            │   ├── utils.py
            │   ├── docker-compose.yml
            │   └── docker-shell.sh	
            └── preprocess_lsars
                ├── Dockerfile
                ├── Pipfile
                ├── Pipfile.lock
                ├── preprocess_lsars.py
                ├── docker-compose.yml
                ├── docker-shell.sh
                └── requirements.txt

--------
# Milestone 2 Overview

**Team Members**   
Varun Ullanat, Hannah Pierce-Hoffman

**Group Name**   
FlavorFusion

**Project**   
In this project, we aim to build an app that captures cultural differences in Google resturaunt reviews using abstractive summaries generated by a large language model. 

## Milestone 2 Contents ##

We use two datasets in this work:
1. **Google Local dataset**: Reviews and business metadata from the state of Massachusetts, USA, totalling about 2GB of data. This dataset can be found in the reviews-data bucket on our private Google Cloud project. We sourced this data from [here](https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/googlelocal/).
2. **LSARS dataset**: Reviews of products from an e-commerece website with a total space of 250 MB. This dataset can be found in the lsars-data bucket on our private Google Cloud project. We sourced this data from [here](https://github.com/ScarletPan/LSARS).

### preprocess_google container ###
- This container reads 2GB of Google Local data and filters out non-restaurant businesses and low-quality reviews.
- The inputs for this container are the following:
    - Input and output filepaths (must be mounted in `docker-volumes` or downloaded from GCP)
    - Preprocessing parameters
    - Secrets file (contains service account credentials for Google Cloud)
- The output from this container is a CSV file containing merged data for all businesses, where the columns are business name, location, and all reviews seperated by a special token.

1. `src/preprocess_google/preprocess_google.py` Here we do preprocessing on our Google Local dataset. We read the reviews and metadata json files, filter them according to input parameters, and combine them such that all reviews for each restaraunt are concatenated by a special separating token. We also introduce capability to download raw data from GCP and upload outputs to GCP.
1. `src/preprocess_google/utils.py` This file includes utility functions such as reading json files, plus an array of business labels relating to restaurants. 
1. `src/preprocess_google/cors.py` This file enables CORS to connect to Label Studio directly from GCS Bucket. 
1. `src/preprocess_google/Dockerfile` This docker sets up a `pipenv` virtual environment for data processing. In the file, we use our service account to connect to Google Cloud. We then create a user named `app` and install the required python packages from `src/preprocess_google/Pipfile`.
1. `src/preprocess_google/docker-shell.sh` This script sets up a network so the Docker container can communicate with the outside world via ports. Next, the script builds our Docker container and uses `src/preprocess_google/docker-compose.yml` to set up the full container environment.

To run Dockerfile - `Instructions here`

### preprocess_lsars container ### 
- This container is responsible for preprocessing and translating the LSARS dataset, which is in Chinese.
- We use the Google Cloud Translate API to translate the text to English. This API provides free translation for the first 500,000 tokens, but after that, the price is $20 per 1 million tokens. We authenticate to the Cloud Translate API via our project's service account.
- The inputs for this container are the following:
    - Input and output filepaths (must be mounted in `docker-volumes`)
    - Preprocessing parameters
    - Secrets file (contains service account credentials for Google Cloud)
- The output of this container is a CSV file containing translated data. Since the data is large, we translate it in chunks and produce a CSV file for each chunk. We maintain the train-test split from the original dataset.
1. `src/preprocess_lsars/preprocess_lsars.py` In this file, we process a specified set of JSON records from the LSARS dataset. We translate each record using the Google Cloud Translate API, and output a CSV file containing translated records. Each record consists of a unique identifier, a summary review, and a list of summarized reviews concatenated by a separator token.  We also introduce capability to download raw data from GCP and upload outputs to GCP.
1. `src/preprocess_lsars/Dockerfile` This docker sets up a `pipenv` virtual environment for data processing. In the file, we use our service account to connect to Google Cloud. We then create a user named `app` and install the required python packages from `src/preprocess_lsars/Pipfile`.
1. `src/preprocess_lsars/docker-shell.sh` This script sets up a network so the Docker container can communicate with the outside world via ports. Next, the script builds our Docker container and uses `src/preprocess_lsars/docker-compose.yml` to set up the full container environment.

To run Dockerfile - `Instructions here`

### docker-volumes ###
This directory is mounted to both the `preprocess_google` and `preprocess_lsars` Docker containers in the `docker-compose.yml` file for each container. This directory contains data (not tracked on Github) as well as notebooks which may be edited inside the Docker container.

### Notebooks ###    
This directory is currently empty, but will be used to store code which doesn't belong to a specific container, such as reports, insights, or visualizations.

### References ###  
This directory is currently empty, but will be used to store code from the [PRIMERA model](https://github.com/allenai/PRIMER).

--------
# Setup Notes #

### Running Docker containers ###   
Both Docker containers delivered in this milestone can be run via the `docker-shell.sh` script inside the container. To run the container of your choice, do the following:
- Clone the repository and checkout the `milestone2` branch
- `cd src/<desired_directory>`
- `chmod 777 docker-shell.sh`
- `./docker-shell.sh`
The relevant Dockerfile will build and you will be dropped into a shell prompt as `app` user. From there, you can run a data processing script or connect to a Jupyter session.

### Running preprocess_google.py script ###
To run this script, you must activate the `preprocess_google` docker container. Once you are inside the container, you can run the script with the following arguments:
- `--download`: Flag to indicate that the untranslated data should be downloaded from our GCS bucket (will not overwrite any previously translated data)
- `--reviews_file_path`: Path to reviews file (either pre-mounted or downloaded with the `-d` flag)
- `--metadata_file_path`: Path to metadata file (either pre-mounted or downloaded with the `-d` flag)
- `--output_file_path`: Path where translated data will be placed (should be in a mounted volume to avoid losing the data)
- `--upload`: Flag to indicate whether the output file should be uploaded to our GCS bucket 
- `--min_char`: Minimum number of characters in each review
- `--max_char`: Maximum number of characters in each review
- `--stop_line`: Line in the input file where translation should stop
- `--max_num_reviews`: Maximum number of reviews for each business

We only preprocess data from the state of Massachusetts for this milestone. 

### Running preprocess_lsars.py script ###
To run this script, you must activate the `preprocess_lsars` docker container. Once you are inside the container, you can run the script with the following arguments:
- `-d` or `--download`: Flag to indicate that the untranslated data should be downloaded from our GCS bucket (will not overwrite any previously translated data)
- `--reviews_file_path`: Path to untranslated review file (either pre-mounted or downloaded with the `-d` flag)
- `--start_line`: Line in the input file where translation should start (each line is a single JSON record containing a summary and a group of reviews)
- `--stop_line`: Line in the input file where translation should stop
- `--output_file_path`: Path where translated data will be placed (should be in a mounted volume to avoid losing the data)
- `--upload`: Flag to indicate whether the output file should be uploaded to our GCS bucket 
   
Since the LSARS dataset contains large files, we provide `start_line` and `stop_line` arguments so the user can avoid reading an entire file into memory at once. We also provide the option of downloading the raw data, which is useful if the user is working in a GCP VM without a mounted bucket.     
    
Translating 500 reviews takes about 6 minutes. The script will output a progress bar showing how many records have been translated.

### Working with Label Studio

We use Label Studio to manually summarize a set of reviews from the Google Local dataset. This is to run few-shot inference on a trained model in the future. The steps are as follows:

- When you run `./docker-shell.sh` for the preprocess_google container, you will automically see a 'heartexlabs/label-studio:latest' container opened at port `http://localhost:8080`.
- Go to to the port and enter the credentials in `src/preprocess_google/docker-compose.ym`.
- Click on `Create Project` and enter project name and description.
- We can import the processed output file from `preprocess_google.py` directly in the `Data Import` step. Click import as csv/tsv.
- On `Labeling Setup` you have to choose Natural Language Processing > Text Summarization. Select 'text' as the column to be summarized.
- Now, we label about 10 summaries by copy-pasting the reviews into ChatGPT using the following prompt: "Summarise these restaurant reviews that are separated by ||||| into one representative sentence: " followed by the reviews.
- After summarizing them, we export the data by going to `Add Cloud Storage` in `Cloud Storage` under `Settings`.
      - Storage Type: `Google Cloud Storage`
      - Storage Title: `labelled_data`
      - Bucket Name: `reviews-data`
      - Bucket Prefix: `labelled_data`
      - File Filter Regex: .*
      - Enable: Treat every bucket object as a source file
      - Enable: Use pre-signed URLs
      - Ignore: Google Application Credentials (This should populate with your secrets key)
      - Ignore: Google Project ID
- Now, you will see the individual annotated json files in `labelled_data` under the `reviews-data` bucket. 

### Connecting to a docker container with Jupyter ###
We used Jupyter Lab to write and debug preprocessing scripts. Running Jupyter Lab inside either of our docker containers is straightforward. Both containers are configured in `docker-compose.yml` to connect to port 8888 on the host machine. Therefore, once you have a container running on your local machine, you can run the following commands to connect to the container with Jupyter:
- `jupyter lab --ip 0.0.0.0 --no-browser --allow-root` (run inside the container)
- Navigate to `http://localhost:8888` on your local machine
- Enter password "docker" to access Jupyter Lab

### Committing data files to DVC ###
We use DVC as our data versioning pipeline. Our DVC configuration has one remote for each dataset. The remote for LSARS data is called `lsars-data`, and the remote for Google data is called `google-data`. 

To download a specific version of one of our datasets, you can use the [dvc get](https://dvc.org/doc/command-reference/get) command. You need to specify a [dataset tag](https://github.com/hpiercehoffman/AC215_FlavorFusion/tags) as well as a remote to use.

**Example of downloading a specific dataset version**   
`dvc get https://github.com/hpiercehoffman/AC215_FlavorFusion/ src/docker-volumes/lsars-data --force --rev  lsars_train500 --remote lsars-data`     
  
The above command will download the `lsars-data` directory from the `lsars-data` remote. The data version will match the `lsars_train500` tag.

To add new or modified data files to DVC, you can use the [dvc add](https://dvc.org/doc/command-reference/add) command. You can then use [dvc push](https://dvc.org/doc/command-reference/push) to push to the appropriate remote.   
  
**Example of pushing new data to DVC**    
`dvc add src/docker-volumes/lsars-data`  
`dvc push src/docker-volumes/lsars-data -r lsars-data`  
  
The above commands will add any modifications in the `lsars-data` directory to the DVC staging area, then push these modifications to the DVC remote which handles LSARS data.




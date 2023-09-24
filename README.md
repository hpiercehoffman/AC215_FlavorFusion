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
1. Google Local dataset: Reviews and business metadata from the state of Massachusetts, USA, totalling about 2GB of data. This dataset can be found in the reviews-data bucket on our private Google Cloud project. We sourced this data from [here](https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/googlelocal/).
2. LSARS dataset: Reviews of products from an e-commerece website with a total space of 250 MB. This dataset can be found in the lsars-data bucket on our private Google Cloud project. We sourced this data from [here](https://github.com/ScarletPan/LSARS).

### preprocess_google container ###
- This container reads 2GB of Google Local data and filters out non-restaurant businesses and low-quality reviews.
- The inputs for this container are the following:
    - Input and output filepaths (must be mounted in `docker-volumes`)
    - Preprocessing parameters
    - Secrets file (contains service account credentials for Google Cloud)
- The output from this container is 
- Output from this container stored on mounted folder

1. `src/preprocess_google/preprocess.py` Here we do preprocessing on our Google Local dataset. We read the reviews and metadata json files, filter them according to input parameters, and combine them such that all reviews for each restaraunt are concatenated by a special separating token. 
1. `src/preprocess_google/utils.py` This file includes utility functions such as reading json files, plus an array of business labels relating to restaurants. 
1. `src/preprocess_google/cors.py` This file enables CORS to connect to Label Studio directly from GCS Bucket. 
1. `src/preprocess_google/Dockerfile` This docker sets up a `pipenv` virtual environment for data processing. In the file, we use our service account to connect to Google Cloud. We then create a user named `app` and install the required python packages from `src/preprocess_google/Pipfile`.
1. `src/preprocess_google/docker-shell.sh` This script sets up a network so the Docker container can communicate with the outside world via ports. Next, the script builds our Docker container and uses `src/preprocess_google/docker-compose.yml` to set up the full container environment.

To run Dockerfile - `Instructions here`

### preprocess_lsars container ### 
- This container is responsible for preprocessing and translating the LSARS dataset, which is in Chinese.
- We use the Google Cloud Translate API to translate the text to English. This API provides free translation for the first 500,000 tokens, but after that, the price is $20 per 1 million tokens. We authenticate to the Cloud Translate API via our project's service account.
- The inputs for this container are the following:
    - Input and output filepaths (must be mounted in `docker-volumes`
    - Preprocessing parameters
    - Secrets file (contains service account credentials for Google Cloud)
- The output of this container is a CSV file containing translated data. Since the data is large, we translate it in chunks and produce a CSV file for each chunk. We maintain the train-test split from the original dataset.
1. `src/preprocess_lsars/preprocess_lsars.py` In this file, we process a specified set of JSON records from the LSARS dataset. We translate each record using the Google Cloud Translate API, and output a CSV file containing translated records. Each record consists of a unique identifier, a summary review, and a list of summarized reviews concatenated by a separator token.
1. `src/preprocess_lsars/Dockerfile` This docker sets up a `pipenv` virtual environment for data processing. In the file, we use our service account to connect to Google Cloud. We then create a user named `app` and install the required python packages from `src/preprocess_lsars/Pipfile`.
1. `src/preprocess_lsars/docker-shell.sh` This script sets up a network so the Docker container can communicate with the outside world via ports. Next, the script builds our Docker container and uses `src/preprocess_lsars/docker-compose.yml` to set up the full container environment.

To run Dockerfile - `Instructions here`

### Notebooks ###    
This directory is currently empty, but will be used to store code which doesn't belong to a specific container, such as reports, insights, or visualizations.

### References ###  
This directory is currently empty, but will be used to store code from the [PRIMERA model](https://github.com/allenai/PRIMER).

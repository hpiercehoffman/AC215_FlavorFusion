AC215: FlavorFusion
==============================

## Presentation Video ##
- [Add Link]

## Blog Post ##
- [Add Link]

![image](https://github.com/hpiercehoffman/AC215_FlavorFusion/blob/main/src/frontend-simple/logo.png)

Project Organization
------------
      ├── LICENSE
      ├── README.md
      ├── notebooks
      |   └── pruning_evaluation.ipynb
      ├── references
      │   └── references.md
      ├── reports
      │   ├── milestone2.md
      │   ├── milestone3.md
      │   ├── milestone4.md
      │   └── milestone5.md
      ├── images
      │   └── [Screenshots showing project functionality]
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
            ├── inference_cloud_functions
            │   ├── inference_zero_shot.py
            │   ├── inference_trained.py
            │   ├── inference_pruned.py
            │   └── requirements.txt
            ├── frontend-simple
            │   ├── docker-shell.sh
            │   ├── index.html
            │   ├── Dockerfile
            │   └── logo.png
            ├── deployment
            │   ├── deploy-cluster-short.yml
            │   ├── deploy-create-instance.yml
            │   ├── deploy-docker-images.yml
            │   ├── deploy-k8s-cluster.yml
            │   ├── deploy-provision-instance.yml
            │   ├── deploy-setup-containers.yml
            │   ├── deploy-setup-webserver.yml
            │   ├── docker-entrypoint.sh
            │   ├── docker-shell.sh
            │   ├── inventory.yml
            │   └── nginx-conf
            │       └── nginx
            │           └── nginx.conf
            ├── api-service
            │   ├── docker-shell.sh
            │   ├── docker-entrypoint.sh
            │   ├── Dockerfile
            │   ├── Pipfile
            │   ├── Pipfile.lock
            │   └── api
            │       ├── service.py
            │       ├── data_download.py
            │       ├── model_inference.py
            │       └── openapi.json
            └── train
                ├── Dockerfile
                ├── Pipfile
                ├── Pipfile.lock
                ├── train_serverless.py
                ├── config.yml
                ├── requirements.txt
                └── submit_job.sh

--------
# AC215 - Final Project

**Team Members**   
Varun Ullanat, Hannah Pierce-Hoffman

**Group Name**   
FlavorFusion

**Project**   
In this project, we aim to build an app that captures cultural differences in Google resturaunt reviews using abstractive summaries generated by a large language model.

For a full list of external references used in this project, please refer to our [reference document](https://github.com/hpiercehoffman/AC215_FlavorFusion/blob/main/references/references.md).

## Deliverables Breakdown ##
- For information on data preprocessing, see our [Milestone 2 report](https://github.com/hpiercehoffman/AC215_FlavorFusion/blob/main/reports/milestone2.md). Data preprocessing takes place in the **preprocess_google** and **preprocess_lsars** Docker containers.
- For information on model training, see our [Milestone 3 report](https://github.com/hpiercehoffman/AC215_FlavorFusion/blob/main/reports/milestone3.md). Model training takes place in the **train** Docker container.
- For information on model optimization and deployment, see our [Milestone 4 report](https://github.com/hpiercehoffman/AC215_FlavorFusion/blob/main/reports/milestone4.md). The scripts in the **inference_cloud_functions** directory are used for model deployment.
- For information on front-end architecture and API development, see our [Milestone 5 report](https://github.com/hpiercehoffman/AC215_FlavorFusion/blob/main/reports/milestone5.md). The **frontend_simple** container is responsible for running the frontend, while the **api-service** container handles backend API calls.
- This page (Milestone 6) deals with automation, scaling, and CI/CD.

### Completed App ###

### Deployment: Ansible ###


### Scaling: Kubernetes ###


### CI/CD: Github Actions ###




### notebooks ###    
This directory contains code which doesn't belong to a specific container:
- `pruning_evaluation.ipynb`: Code to evaluate and compute benchmark statistics for a base versus pruned model.
- [Add notebook for Ethnicolr classification]

### reports ###
This directory contains our reports from past milestones:
- [Milestone 2](https://github.com/hpiercehoffman/AC215_FlavorFusion/blob/main/reports/milestone2.md): Data preprocessing, Label Studio, and DVC.
- [Milestone 3](https://github.com/hpiercehoffman/AC215_FlavorFusion/blob/main/reports/milestone3.md): Model training, VM setup, and experiment tracking.
- [Milestone 4](https://github.com/hpiercehoffman/AC215_FlavorFusion/blob/main/reports/milestone4.md): Model optimization and model deployment.
- [Milestone 5](https://github.com/hpiercehoffman/AC215_FlavorFusion/blob/main/reports/milestone5.md)
Frontend interface, API development, and solution architecture.

### references ###  
This directory contains information on models, datasets, and other external references used in this project. References are detailed in [references.md](https://github.com/hpiercehoffman/AC215_FlavorFusion/blob/main/references/references.md).

--------
# Setup Notes #

### Additional Ansible and Kubernetes Setup ###


# References #

For a full list of external references used in this project, please refer to our [reference document](https://github.com/hpiercehoffman/AC215_FlavorFusion/blob/milestone5/references/references.md).


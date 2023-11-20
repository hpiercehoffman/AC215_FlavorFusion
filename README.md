AC215: Milestone 5
==============================

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
      │   └── milestone4.md
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
# Milestone 5 Overview

**Team Members**   
Varun Ullanat, Hannah Pierce-Hoffman

**Group Name**   
FlavorFusion

**Project**   
In this project, we aim to build an app that captures cultural differences in Google resturaunt reviews using abstractive summaries generated by a large language model.

For a full list of external references used in this project, please refer to our [reference document](https://github.com/hpiercehoffman/AC215_FlavorFusion/blob/milestone5/references/references.md).

## Milestone 5 Deliverables ##

This milestone deals with front-end architecture and API development.
- For information on data preprocessing, see our [Milestone 2 report](https://github.com/hpiercehoffman/AC215_FlavorFusion/blob/milestone5/reports/milestone2.md). Data preprocessing takes place in the **preprocess_google** and **preprocess_lsars** Docker containers.
- For information on model training, see our [Milestone 3 report](https://github.com/hpiercehoffman/AC215_FlavorFusion/blob/milestone5/reports/milestone3.md). Model training takes place in the **train** Docker container.
- For information on model optimization and deployment, see our [Milestone 4 report](https://github.com/hpiercehoffman/AC215_FlavorFusion/blob/milestone5/reports/milestone4.md). The scripts in the **inference_cloud_functions** directory are used for model deployment.

### Application Design ###

At this point in our development process, we created **design documents** showing the high-level architecture of our app. We created a **solution architecture** which shows the high-level strategy of our entire project, as well as a **technical architecture** which provides implementation details about how the different components of the project work together. We will discuss both design documents below. 

Note that these design documents represent a *final* implementation, meaning that some of the work is not yet completed in this milestone. We note the pending work below each image.

**Solution Architecture**

![image](https://github.com/hpiercehoffman/AC215_FlavorFusion/blob/milestone5/images/solution_architecture.png)

Our solution architecture shows the flow of **processes** (tasks performed by developers and users), **execution** (code running in different parts of the project pipeline), and **state** (stored objects and artifacts). In this view of the project, we abstract away technical details.

*Pending work:* Currently, we have not yet implemented HTTPS communication between the API service and the deployment stage of the ML pipeline. Instead, the API service downloads a trained model from WandB. In our final implementation, the API service will communicate with a deployed model (e.g. a cloud function). 

**Technical Architecture**

![image](https://github.com/hpiercehoffman/AC215_FlavorFusion/blob/milestone5/images/technical_architecture.png)

Our technical architecture provides a detailed view of the project structure, including components responsible for different actions and communication between these components.

*Pending work:* Currently, we have not yet implemented a Nginx container to handle communication between the frontend and the API service. Instead, the frontend communicates directly with the API service via a local port. In our final implementation, we will run a Nginx container on our VM or within a Kubernetes cluster, so that Nginx can act as a reverse proxy to forward requests from the frontend container to the API container.

### Frontend App ###

We implemented a prototype frontend app using HTML and Javascript. The app shows a simple front page where the user can select a restaurant from a dropdown menu. The dropdown menu is populated based on data downloaded from our GCS bucket. The user can click the "Submit" button to generate a summary of 5 random reviews from the selected restaurant. In the final implementation, we will generate multiple summaries for groups of reviews which are stratified by estimated cultural background. 

We also include a Swagger API testing interface in the prototype front-end, so we can easily test our APIs. We'll discuss each API in more detail in the next section.

The screenshot below shows our prototype front-end.

![image](https://github.com/hpiercehoffman/AC215_FlavorFusion/blob/milestone5/images/frontend_with_swagger.png)

### Backend API ###


### notebooks ###    
This directory contains code which doesn't belong to a specific container:
- `pruning_evaluation.ipynb`: Code to evaluate and compute benchmark statistics for a base versus pruned model. 

### reports ###
This directory contains our reports from past milestones:
- [Milestone 2](https://github.com/hpiercehoffman/AC215_FlavorFusion/blob/milestone3/reports/milestone2.md): Data preprocessing, Label Studio, and DVC.
- [Milestone 3](https://github.com/hpiercehoffman/AC215_FlavorFusion/blob/milestone3/reports/milestone3.md): Model training, VM setup, and experiment tracking.
- [Milestone 4](https://github.com/hpiercehoffman/AC215_FlavorFusion/blob/milestone5/reports/milestone4.md): Model optimization and model deployment.

### references ###  
This directory contains information on models, datasets, and other external references used in this project. References are detailed in [references.md](https://github.com/hpiercehoffman/AC215_FlavorFusion/blob/milestone5/references/references.md).

--------
# Setup Notes #

### Running Frontend and API Service ###


# References #

For a full list of external references used in this project, please refer to our [reference document](https://github.com/hpiercehoffman/AC215_FlavorFusion/blob/milestone3/references/references.md).



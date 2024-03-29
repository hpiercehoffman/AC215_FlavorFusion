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

The screenshot below shows our prototype front-end. For full details of how to run the frontend and backend, see the [Setup Notes](https://github.com/hpiercehoffman/AC215_FlavorFusion/tree/milestone5#setup-notes) section.

![image](https://github.com/hpiercehoffman/AC215_FlavorFusion/blob/milestone5/images/frontend_with_swagger.png)

### Backend API ###

We used FastAPI to create backend RESTful APIs which handle communication with the frontend. We implemented the following APIs:
- `/`: Default API call. GET method which returns a welcome message.
- `/populate`: GET method which downloads a subset of our data from a GCS bucket and extracts a list of restaurant names. Restaurant names are then used to populate a dropdown menu in the frontend.
- `/predict`: POST method which runs model inference for a selected restaurant, generating a summary of reviews from the selected restaurant. Future implementations of this API will include options to summarize a specific group of reviews based on estimated cultural background of reviewers.

In addition to testing our APIs by interacting with the frontend, we also used Swagger's [API testing kit](https://swagger.io/solutions/api-testing/) to verify that all APIs are working correctly. The screenshots below show the results of testing each of our three APIs with Swagger. All APIs are working as expected.

**Testing the "/" API**

![image](https://github.com/hpiercehoffman/AC215_FlavorFusion/blob/milestone5/images/swagger_default_api.png)

**Testing the "/populate" API**

![image](https://github.com/hpiercehoffman/AC215_FlavorFusion/blob/milestone5/images/swagger_populate_api.png)

**Testing the "/predict" API**

![image](https://github.com/hpiercehoffman/AC215_FlavorFusion/blob/milestone5/images/swagger_get_api.png)

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

For our prototype implementation in this milestone, we ran the backend API service on a GCP VM and the frontend server on a local machine. In our final implementation, we will use Nginx to handle communication between the frontend and backend. 

Below, we describe the steps to reproduce our development configuration.

1. Create a GCP VM instance with a Nvidia L4 GPU. Install Nvidia drivers and Docker. HTTP traffic should be enabled, although we use SSH in this prototype implementation.
2. Connect to the VM, opening an SSH tunnel from port 9000 of the VM to port 9000 on your local machine:  
   `gcloud compute ssh primera-gpu-200 -- -L 9000:localhost:9000`  
3. Clone our repository on the VM and build the `test-api` Docker image:  
   `git clone https://github.com/hpiercehoffman/AC215_FlavorFusion.git`
4. Set up secrets files at the same directory level as the project repository (not inside the repo). We added a `google_secrets.json` file with the credentials for our service account (necessary to download files from our GCS bucket), and a `wandb_key.txt` file with our WandB API key (necessary to download our trained model from WandB).
5. It may be necessary to change permissions of the directory where the Docker container will download files. You can change permissions as follows, so the Docker container will have permission to download the model and data:  
   `sudo chmod 777 AC215_FlavorFusion/src/api-service/`  
7. Run the dockerfile for our API server:  
   `cd AC215_FlavorFusion/src/api-service/`  
   `sh docker-shell.sh`  
8. This should build and run a Docker container called `test-api`. Once the Docker container is running, start the Uvicorn server:  
   `uvicorn_server_production`  
9. The Uvicorn server is now running on port 9000 of the Docker container, which is connected to port 9000 on the GCP VM. Port 9000 on the VM is also connected to port 9000 on your local machine via SSH connection.
10. Now we can run the frontend on local. Ensure that our repo is cloned on your local machine. Run the dockerfile for the frontend server:  
   `cd AC215_FlavorFusion/src/frontend-simple`  
   `sh docker-shell.sh`  
11. This Docker will run with a connection between port 3000 in the Docker container and port 3000 on your local machine. Therefore, you should run the frontend server on port 3000:  
    `http-server -p 3000`  
12. Once the HTTP server is running, you should be able to visit `localhost:3000` in your browser and see the FlavorFusion homepage. API calls made from this frontend are routed via Axios to port 9000 on your local machine. Since port 9000 on your local machine is connected to the VM and the Docker, you'll be able to send and receive information from the Uvicorn server.

# References #

For a full list of external references used in this project, please refer to our [reference document](https://github.com/hpiercehoffman/AC215_FlavorFusion/blob/milestone5/references/references.md).

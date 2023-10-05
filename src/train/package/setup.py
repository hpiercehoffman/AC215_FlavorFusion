from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ["wandb==0.15.11",
                     "pandas==2.1.1",
                     "numpy==1.26.0",
                     "google-cloud-storage==2.11.0",
                     "google-api-python-client==2.101.0",
                     "torch==2.0.1",
                     "transformers==4.33.3",
                     "datasets==2.14.5",
                     "nltk==3.8.1",
                     "tqdm==4.66.1",
                     "rouge-score==0.1.2"]

setup(
    name="train-primera",
    version="0.0.1",
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    description="PRIMERA mode training",
)
#!/bin/bash

# Define ECR Repository URL
ECR_URL="339712912845.dkr.ecr.ap-southeast-2.amazonaws.com" # infrastructure account

# Define Image Name
IMAGE_NAME="exfil-xgboost"
AWS_PROFILE="aws-infrastructure"

# Get the current Git commit hash to use as the image tag
IMAGE_TAG=$(git rev-parse HEAD)

echo "Building and running Docker container for commit: $IMAGE_TAG"

# Open Docker application
open -a Docker

pip install --upgrade pip
pip install "cython<3.0.0"
pip install -r requirements.txt --no-build-isolation

# Build the Docker image with the commit hash as a tag
docker buildx build --platform linux/amd64 -t xgboost-container-base:1.7-1-cpu-py3 -f docker/1.7-1/base/Dockerfile.cpu .
python3 setup.py bdist_wheel --universal

docker buildx build --platform linux/amd64 -t ${IMAGE_NAME}:${IMAGE_TAG} -f docker/1.7-1/final/Ddocockerfile.cpu .
docker tag "${IMAGE_NAME}:${IMAGE_TAG}" "${ECR_URL}/${IMAGE_NAME}:${IMAGE_TAG}"

# Login to AWS ECR and push the image
aws ecr get-login-password --region ap-southeast-2 --profile $AWS_PROFILE | docker login --username AWS --password-stdin $ECR_URL
docker push "${ECR_URL}/${IMAGE_NAME}:${IMAGE_TAG}"

echo "Pushed ${IMAGE_NAME}:${IMAGE_TAG} to AWS ECR"

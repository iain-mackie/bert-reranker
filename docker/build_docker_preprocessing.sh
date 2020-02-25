# build docker
DOCKER_PATH="./Dockerfile_preprocessing"

echo "Trying to build docker image from:" $DOCKER_PATH;

if [ -f "$DOCKER_PATH" ]; then
  echo "file exits"
  sudo docker build -t iainmackie/bert-reranker-preprocessing:v7 -f $DOCKER_PATH .

else
  echo "Error - path to file not found:" $DOCKER_PATH;
  exit 0;

fi

#!/bin/bash
MACHINENAME=${1:-microsoftaicompd12}
PYTHON_SCRIPT_PATH=${2:-pig_chase_jumper}

DOCKER_PATH=../../docker

eval $(docker-machine env $MACHINENAME)


docker build $DOCKER_PATH/malmo -t malmo:latest
docker build $DOCKER_PATH/malmopy-cntk-cpu-py27 -t malmopy-cntk-cpu-py27:latest
docker build ../danish_puppeteers -t danishpuppet:latest

# Open tensorboard results
#xdg-open http://$(docker-machine ip $MACHINENAME):6006 &

# Set variable PYTHON_SCRIPT_NAME to experiment file in .env
# and refer to ${PYTHON_SCRIPT_NAME} for the command in the
# docker compose file.
# https://docs.docker.com/compose/environment-variables/#the-env-file

echo "PYTHON_SCRIPT_PATH=$PYTHON_SCRIPT_PATH" > .env

docker-compose up

#!/bin/bash

machinename=microsoftaicompd12
docker_path=../../docker

#eval $(docker-machine env $machinename)

docker build $docker_path/malmo -t malmo:latest
docker build $docker_path/malmopy-cntk-cpu-py27 -t malmopy-cntk-cpu-py27:latest
docker build ../danish_puppeteers -t danishpuppet:latest


xdg-open http://$(docker-machine ip $machinename):6006 &

docker-compose up 

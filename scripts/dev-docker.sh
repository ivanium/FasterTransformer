#!/bin/bash
set -x

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$( cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd )

IMAGE_NAME=manifold_base
# IMAGE_NAME=nvcr.io/nvidia/pytorch:22.09-py3
CONTAINER_NAME=manifold-dev-$USER

start_docker() {
    docker_running=$(docker ps --format '{{.Names}}' | grep ${CONTAINER_NAME})
    if [[ ! $docker_running ]]
    then
        docker run --rm --runtime=nvidia --gpus all \
            -d -it --name ${CONTAINER_NAME} \
            -v ${HOME}/.cache/huggingface:/root/.cache/huggingface \
            -v ${ROOT_DIR}/../manifold:/manifold \
            -v ${ROOT_DIR}:/FasterTransformer \
            ${IMAGE_NAME} /bin/bash
    fi
}

into_docker() {
    docker exec -it ${CONTAINER_NAME} /bin/bash
}

op=$1
if [[ ! $op ]] || [[ $op == start ]]
then
    start_docker
    into_docker
elif [[ $op == stop ]]
then
    docker stop ${CONTAINER_NAME}
    # docker rm ${CONTAINER_NAME}
fi

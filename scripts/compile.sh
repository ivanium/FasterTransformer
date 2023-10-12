#!/bin/bash

SCRIPT_NAME=$(basename "$0")
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$( cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd )

CONTAINER_NAME=manifold-dev-${USER}

DSM_NUMBER=75 # For T4, it's 75; for A100, it's 80

# if grep -q docker /proc/1/cgroup;
if [ -f /.dockerenv ];
then
  echo "Inside docker"
  if [[ ! -d ${ROOT_DIR}/build ]];
  then
    mkdir -p  ${ROOT_DIR}/build
  fi

  pushd ${ROOT_DIR}/build
  if [[ ! -f Makefile ]];  # cmake first
  then
    cmake -DSM=${DSM_NUMBER} -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON -DBUILD_MULTI_GPU=ON ..
  fi
  make -j12
  popd
else
  echo "On host machine"
  docker exec -it ${CONTAINER_NAME} bash -c "cd ${ROOT_DIR} && ./scripts/${SCRIPT_NAME}"
fi

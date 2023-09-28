#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$( cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd )

CONTAINER_NAME=manifold-dev-${USER}

if grep -q docker /proc/1/cgroup;
then
  echo "Inside docker"
  pushd /FasterTransformer/build
  make -j
  popd
else
  echo "On host machine"
  docker exec -it ${CONTAINER_NAME} bash -c "cd /FasterTransformer/build && make -j"
fi
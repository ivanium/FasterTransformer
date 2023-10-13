#!/bin/bash

SCRIPT_NAME=$(basename "$0")
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$( cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd )

CONTAINER_NAME=manifold-dev-${USER}

#if grep -q docker /proc/1/cgroup;
if [ -f /.dockerenv ];
then
  echo "Inside docker"
  pushd /FasterTransformer/build
  # ./bin/mnfd_llama /FasterTransformer/models/llama2/1-gpu/config.ini
  if [ "$#" -eq 1 ]; then
    echo "Running mnfd_llama with "$1""
    ./bin/mnfd_llama "$1"
  else
    echo "Running mnfd_llama with default config.ini"
    ./bin/mnfd_llama /FasterTransformer/models/llama2/1-gpu/config.ini
  fi
  popd
  # decode output tokens
  python3 ${ROOT_DIR}/examples/utils/hf_detokenize.py ${ROOT_DIR}/build/out
else
  echo "On host machine"
  docker exec -it ${CONTAINER_NAME} bash -c "cd /FasterTransformer && ./scripts/${SCRIPT_NAME}"
fi

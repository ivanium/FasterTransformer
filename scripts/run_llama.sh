#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$( cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd )

CONTAINER_NAME=manifold-dev

if grep -q docker /proc/1/cgroup;
then
  echo "Inside docker"
  pushd /FasterTransformer/build
  ./bin/llama_example /FasterTransformer/models/llama2/1-gpu/config.ini
  popd
else
  echo "On host machine"
  docker exec -it ${CONTAINER_NAME} bash -c "cd /FasterTransformer/build && ./bin/llama_example /FasterTransformer/models/llama2/1-gpu/config.ini"
fi

python3 examples/utils/hf_detokenize.py ${ROOT_DIR}/build/out
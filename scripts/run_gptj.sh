#!/bin/bash

SCRIPT_NAME=$(basename "$0")
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$( cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd )

CONTAINER_NAME=manifold-dev-${USER}

if grep -q docker /proc/1/cgroup;
then
  echo "Inside docker"
  pushd /FasterTransformer/build
  ./bin/gptj_example /FasterTransformer/models/gptj-6b/1-gpu/config.ini
  popd
  # decode output tokens
  python3 ${ROOT_DIR}/examples/utils/hf_detokenize.py ${ROOT_DIR}/build/out EleutherAI/gpt-j-6B
else
  echo "On host machine"
  docker exec -it ${CONTAINER_NAME} bash -c "cd /FasterTransformer && ./scripts/${SCRIPT_NAME}"
fi

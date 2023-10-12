#! /usr/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$( cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd )


batch_initial=1
batch_max=128
output_initial=1
output_max=1024

for ((batch=batch_initial; batch<=batch_max; batch=batch*2))
do
    for ((output=output_initial; output<=output_max; output=output*2))
    do
        config_file="${ROOT_DIR}/tmp/manifold_llama-"$batch"-"$output".ini"
        output_file="${ROOT_DIR}/tmp/manifold_llama-"$batch"-"$output".log"
        sed -r 's#^request_batch_size=[0-9]+#request_batch_size='"$batch"'#; s#^request_output_len=[0-9]+#request_output_len='"$output"'#' ${ROOT_DIR}/models/llama2/1-gpu/config.ini > $config_file
        ${ROOT_DIR}/scripts/run_llama.sh "$config_file"  > "$output_file"
        echo "done $batch $output"
    done
done

echo "done all the executions, now plotting"

python3 plot_llama.py batch_initial batch_max output_initial output_max ${ROOT_DIR}/tmp
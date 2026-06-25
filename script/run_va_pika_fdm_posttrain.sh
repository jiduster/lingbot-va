#!/usr/bin/bash

set -e

export CONFIG_NAME=${CONFIG_NAME:-"pika_fdm_train"}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-"/tmp/hf_datasets_cache"}

bash script/run_va_posttrain.sh "$@"

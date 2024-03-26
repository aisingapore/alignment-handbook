#!/bin/bash

set -euo pipefail

num_gpus_pernode=${1}
num_gpus=${2}
num_node=${3}
master_addr=${4}
master_port=${5}
node_rank=${6}

export SHARED_FS_DIR=/fsx
export HF_HOME="${SHARED_FS_DIR}/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

source "${SHARED_FS_DIR}/miniconda3/etc/profile.d/conda.sh"
conda activate "${SHARED_FS_DIR}/envs/alignment_handbook"
export PATH=$(echo $PATH | sed -e "s|/home/$(whoami)/.local/bin:||")

script_dir=$(cd -- "$(dirname -- ${BASH_SOURCE[0]})" &> /dev/null && pwd)
project_dir=$(readlink -f "${script_dir}/../..")

source "${script_dir}/hf_token"

export TZ='Asia/Singapore'
export TMPDIR=/fsx/tmp
export SHARED_FS_DIR=/fsx
export SHARED_OPT_DIR=${SHARED_FS_DIR}/opt
export AWS_OFI_DIR=${SHARED_OPT_DIR}/aws-ofi-nccl
export CUDA_DIR=/usr/local/cuda-12.2
export EFA_DIR=/opt/amazon/efa
export NCCL_DIR=${SHARED_OPT_DIR}/nccl
export OPENMPI_DIR=/opt/amazon/openmpi

export PATH=${CUDA_DIR}/bin:${OPENMPI_DIR}/bin:${PATH:+:${PATH}}

export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
export LD_LIBRARY_PATH=${NCCL_DIR}/build/lib:${CUDA_DIR}/lib64:${CUDA_DIR}:${EFA_DIR}/lib:${OPENMPI_DIR}/lib:${AWS_OFI_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export FI_PROVIDER='efa'
export FI_EFA_USE_DEVICE_RDMA=1
export NCCL_DEBUG=INFO
export NCCL_NVLS_ENABLE=1
export CUDA_LAUNCH_BLOCKING=1

if [[ $(( num_node * num_gpus_pernode )) != ${num_gpus} ]]; then
    echo "Invalid topology: ${num_node} * ${num_gpus_pernode} != ${num_gpus}"
    exit 1
fi

ACCELERATE_LOG_LEVEL=info \
accelerate launch \
    --config_file "${project_dir}/recipes/accelerate_configs/multi_node.yaml" \
    --num_processes ${num_gpus} \
    --num_machines ${num_node} \
    --machine_rank ${node_rank} \
    --main_process_ip ${master_addr} \
    --main_process_port ${master_port} \
    --rdzv_conf rdzv_endpoint="${master_addr}:${master_port}" \
    "${project_dir}/scripts/run_cpt.py" \
    "${project_dir}/recipes/zephyr-7b-gemma/cpt/config_mod.yaml"

#!/bin/bash
set -e

DATASETS=(reddit flickr yelp ogbn-products ogbn-proteins)
MODELS=(sage gcn gin)
K_VALUES=(4 8 16 32 64 128 256)

run_experiment() {
    local dataset=$1 model=$2 k=$3 use_maxk=$4 gpu=${5:-0}
    local exp_id="${dataset}_${model}_k${k}_maxk${use_maxk}"
    
    mkdir -p logs
    
    local cmd="python maxk_gnn_integrated.py --dataset $dataset --model $model --maxk $k --gpu $gpu"
    [[ "$use_maxk" == "true" ]] && cmd="$cmd --use_maxk_kernels"
    
    export CUDA_VISIBLE_DEVICES=$gpu
    timeout 7200 $cmd > "logs/${exp_id}.log" 2>&1 || echo "FAILED: $exp_id"
}

run_all() {
    local kernels_available=$(python -c "try: import maxk_cuda_kernels; print('true')" 2>/dev/null || echo "false")
    
    for dataset in "${DATASETS[@]}"; do
        for model in "${MODELS[@]}"; do
            for k in "${K_VALUES[@]}"; do
                # Baseline
                run_experiment "$dataset" "$model" "$k" "false" 0
                
                # MaxK if available
                [[ "$kernels_available" == "true" ]] && run_experiment "$dataset" "$model" "$k" "true" 0
            done
        done
    done
}

run_subset() {
    local datasets="$1" models="$2" k_vals="$3"
    local kernels_available=$(python -c "try: import maxk_cuda_kernels; print('true')" 2>/dev/null || echo "false")
    
    IFS=',' read -ra DS <<< "$datasets"
    IFS=',' read -ra MS <<< "$models"
    IFS=',' read -ra KS <<< "$k_vals"
    
    for d in "${DS[@]}"; do
        for m in "${MS[@]}"; do
            for k in "${KS[@]}"; do
                run_experiment "$d" "$m" "$k" "false" 0
                [[ "$kernels_available" == "true" ]] && run_experiment "$d" "$m" "$k" "true" 0
            done
        done
    done
}

case "${1:-help}" in
    all) run_all ;;
    subset) run_subset "$2" "$3" "$4" ;;
    *) echo "Usage: $0 {all|subset DATASETS MODELS K_VALUES}" ;;
esac

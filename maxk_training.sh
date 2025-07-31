#!/bin/bash

DATASETS=(reddit flickr yelp ogbn-products ogbn-proteins)
MODELS=(sage gcn gin)
K_VALUES=(4 8 16 32 64 128 256)

run_maxk_experiment() {
    local dataset=$1 model=$2 k=$3 gpu=${4:-0}
    local exp_id="${dataset}_${model}_k${k}_maxk_true"
    
    mkdir -p logs
    
    local cmd="python maxk_gnn_integrated.py --dataset $dataset --model $model --maxk $k --gpu $gpu --use_maxk_kernels"
    
    export CUDA_VISIBLE_DEVICES=$gpu
    nohup timeout 7200 $cmd > "logs/${exp_id}.log" 2>&1 &
    echo "Started MaxK: $exp_id (PID: $!)"
}

run_all_maxk() {
    for dataset in "${DATASETS[@]}"; do
        for model in "${MODELS[@]}"; do
            for k in "${K_VALUES[@]}"; do
                run_maxk_experiment "$dataset" "$model" "$k" 0
                sleep 2
            done
        done
    done
}

run_subset_maxk() {
    local datasets="$1" models="$2" k_vals="$3"
    
    IFS=',' read -ra DS <<< "$datasets"
    IFS=',' read -ra MS <<< "$models"
    IFS=',' read -ra KS <<< "$k_vals"
    
    for d in "${DS[@]}"; do
        for m in "${MS[@]}"; do
            for k in "${KS[@]}"; do
                run_maxk_experiment "$d" "$m" "$k" 0
                sleep 2
            done
        done
    done
}

case "$1" in
    all) run_all_maxk ;;
    subset) run_subset_maxk "$2" "$3" "$4" ;;
    kill) pkill -f "python maxk_gnn_integrated.py" ;;
    *) echo "Usage: $0 {all|subset|kill}" ;;
esac

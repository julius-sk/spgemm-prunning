#!/bin/bash

# Comprehensive MaxK-GNN Training Script
# Trains SAGE/GCN/GIN on multiple datasets with various k values
# Tests both with and without MaxK kernels for performance comparison

set -e  # Exit on any error

# Configuration
DATASETS=("reddit" "flickr" "yelp" "ogbn-arxiv" "ogbn-products" "ogbn-proteins")
MODELS=("sage" "gcn" "gin")
K_VALUES=(4 8 16 32 64 128 256)
SEEDS=(42 97 123)  # Multiple seeds for statistical significance
GPUS=(0)  # Add more GPU IDs if you have multiple GPUs
NUM_EPOCHS=200
PATIENCE=50

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]# Signal handlers for graceful shutdown
setup_signal_handlers() {
    trap 'handle_interrupt' INT TERM
}

handle_interrupt() {
    echo ""
    warning "Interrupt signal received. Cleaning up..."
    
    show_job_status
    
    read -p "Kill all running jobs? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        kill_all_jobs
    fi
    
    log "Cleanup completed. Exiting..."
    exit 130
}

# Help function
show_help() {
    cat << EOF
Comprehensive MaxK-GNN Training Script
=====================================

This script trains SAGE/GCN/GIN models on multiple datasets with various k values,
comparing performance with and without MaxK kernels in background mode.

USAGE:
    $0 [COMMAND] [OPTIONS]

COMMANDS:
    full                 Run complete experiment suite (all datasets, models, k values)
    subset DATASETS MODELS K_VALUES [MAX_CONCURRENT]
                        Run subset of experiments
    status              Show current job status
    kill                Kill all running jobs  
    monitor             Monitor jobs in real-time
    summary             Generate results summary
    clean               Clean temporary files
    help                Show this help

DATASETS: reddit,flickr,yelp,ogbn-arxiv,ogbn-products,ogbn-proteins
MODELS: sage,gcn,gin  
K_VALUES: 4,8,16,32,64,128,256

EXAMPLES:
    # Run all experiments with 4 concurrent jobs
    $0 full 4
    
    # Run subset: Reddit + Flickr, SAGE + GCN, k=16,32,64
    $0 subset "reddit,flickr" "sage,gcn" "16,32,64" 2
    
    # Monitor running jobs
    $0 monitor
    
    # Show current status
    $0 status
    
    # Generate summary after completion
    $0 summary

CONFIGURATION:
    Results: $RESULT_DIR
    Logs: $LOG_DIR
    Seeds: ${SEEDS[*]}
    GPUs: ${GPUS[*]}
    Epochs: $NUM_EPOCHS
    Patience: $PATIENCE

EOF
}

# Command line interface
main() {
    setup_signal_handlers
    
    # Check if Python environment is ready
    if ! python -c "import torch, dgl" >/dev/null 2>&1; then
        error "Python environment not ready. Please activate maxkgnn conda environment."
        exit 1
    fi
    
    # Check for required files
    if [[ ! -f "maxk_gnn_integrated.py" ]]; then
        error "maxk_gnn_integrated.py not found. Please ensure you're in the correct directory."
        exit 1
    fi
    
    local command="${1:-help}"
    
    case "$command" in
        "full")
            local max_concurrent="${2:-4}"
            log "Starting full experiment suite with $max_concurrent concurrent jobs"
            run_full_experiment_suite "$max_concurrent"
            ;;
            
        "subset")
            if [[ $# -lt 4 ]]; then
                error "Usage: $0 subset DATASETS MODELS K_VALUES [MAX_CONCURRENT]"
                exit 1
            fi
            local datasets="$2"
            local models="$3" 
            local k_values="$4"
            local max_concurrent="${5:-2}"
            run_subset_experiments "$datasets" "$models" "$k_values" "$max_concurrent"
            ;;
            
        "status")
            show_job_status
            ;;
            
        "monitor")
            log "Monitoring jobs (Ctrl+C to exit)"
            while true; do
                clear
                show_job_status
                sleep 30
            done
            ;;
            
        "kill")
            kill_all_jobs
            ;;
            
        "summary")
            generate_experiment_summary
            ;;
            
        "clean")
            warning "Cleaning temporary files..."
            rm -rf "$TEMP_DIR"/*
            log "Temporary files cleaned"
            ;;
            
        "help"|"-h"|"--help")
            show_help
            ;;
            
        *)
            error "Unknown command: $command"
            echo "Use '$0 help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/experiment_logs"
RESULT_DIR="${SCRIPT_DIR}/results"
TEMP_DIR="${SCRIPT_DIR}/temp"

# Create directories
mkdir -p "$LOG_DIR" "$RESULT_DIR" "$TEMP_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Function to check if MaxK kernels are available
check_maxk_kernels() {
    python -c "
try:
    import maxk_cuda_kernels
    print('AVAILABLE')
except ImportError:
    print('NOT_AVAILABLE')
" 2>/dev/null
}

# Function to estimate memory requirements
estimate_memory() {
    local dataset="$1"
    local model="$2"
    local k="$3"
    
    case "$dataset" in
        "reddit") echo "16" ;; # GB
        "flickr") echo "8" ;;
        "yelp") echo "12" ;;
        "ogbn-arxiv") echo "6" ;;
        "ogbn-products") echo "24" ;;
        "ogbn-proteins") echo "20" ;;
        *) echo "16" ;;
    esac
}

# Function to get dataset-specific parameters
get_dataset_params() {
    local dataset="$1"
    
    case "$dataset" in
        "reddit")
            echo "--hidden_dim 256 --num_layers 3 --lr 0.01 --dropout 0.5"
            ;;
        "flickr")
            echo "--hidden_dim 256 --num_layers 2 --lr 0.01 --dropout 0.5"
            ;;
        "yelp")
            echo "--hidden_dim 256 --num_layers 3 --lr 0.01 --dropout 0.5"
            ;;
        "ogbn-arxiv")
            echo "--hidden_dim 256 --num_layers 3 --lr 0.01 --dropout 0.5"
            ;;
        "ogbn-products")
            echo "--hidden_dim 256 --num_layers 3 --lr 0.003 --dropout 0.5"
            ;;
        "ogbn-proteins")
            echo "--hidden_dim 256 --num_layers 3 --lr 0.01 --dropout 0.1"
            ;;
        *)
            echo "--hidden_dim 256 --num_layers 3 --lr 0.01 --dropout 0.5"
            ;;
    esac
}

# Function to run a single training experiment
run_experiment() {
    local dataset="$1"
    local model="$2"
    local k="$3"
    local use_maxk="$4"  # "true" or "false"
    local seed="$5"
    local gpu="$6"
    
    local experiment_id="${dataset}_${model}_k${k}_maxk${use_maxk}_seed${seed}"
    local log_file="${LOG_DIR}/${experiment_id}.log"
    local result_file="${RESULT_DIR}/${experiment_id}.json"
    
    # Skip if already completed
    if [[ -f "$result_file" ]]; then
        info "Skipping $experiment_id (already completed)"
        return 0
    fi
    
    log "Starting experiment: $experiment_id"
    
    # Get dataset-specific parameters
    local dataset_params
    dataset_params=$(get_dataset_params "$dataset")
    
    # Estimate memory requirement
    local mem_req
    mem_req=$(estimate_memory "$dataset" "$model" "$k")
    info "Estimated memory requirement: ${mem_req}GB"
    
    # Build command
    local cmd="python maxk_gnn_integrated.py"
    cmd="$cmd --dataset $dataset"
    cmd="$cmd --model $model"
    cmd="$cmd --maxk $k"
    cmd="$cmd --seed $seed"
    cmd="$cmd --gpu $gpu"
    cmd="$cmd --epochs $NUM_EPOCHS"
    cmd="$cmd --patience $PATIENCE"
    cmd="$cmd $dataset_params"
    cmd="$cmd --save_results"
    cmd="$cmd --result_file $result_file"
    
    if [[ "$use_maxk" == "true" ]]; then
        cmd="$cmd --use_maxk_kernels"
        cmd="$cmd --validate_kernels"
    fi
    
    # Add profiling for performance analysis
    cmd="$cmd --profile_kernels"
    cmd="$cmd --log_timing"
    
    # Set environment variables
    export CUDA_VISIBLE_DEVICES="$gpu"
    export PYTHONUNBUFFERED=1
    
    # Create job script for background execution
    local job_script="${TEMP_DIR}/${experiment_id}.sh"
    cat > "$job_script" << EOF
#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONUNBUFFERED=1

echo "\$(date): Starting $experiment_id" >> "$log_file"
start_time=\$(date +%s)

if timeout 7200 $cmd >> "$log_file" 2>&1; then
    end_time=\$(date +%s)
    duration=\$((end_time - start_time))
    echo "\$(date): ✅ Completed $experiment_id in \${duration}s" >> "$log_file"
    
    # Extract metrics and create result file
    best_val_acc=\$(grep "Best Val Acc" "$log_file" | tail -n 1 | awk '{print \$NF}' 2>/dev/null || echo "0.0")
    best_test_acc=\$(grep "Best Test Acc" "$log_file" | tail -n 1 | awk '{print \$NF}' 2>/dev/null || echo "0.0")
    final_train_acc=\$(grep "Train Acc" "$log_file" | tail -n 1 | awk '{print \$NF}' 2>/dev/null || echo "0.0")
    avg_epoch_time=\$(grep "Avg epoch time" "$log_file" | tail -n 1 | awk '{print \$NF}' 2>/dev/null || echo "0.0")
    
    # Create comprehensive result JSON
    cat > "$result_file" << RESULT_EOF
{
    "experiment_id": "$experiment_id",
    "dataset": "$dataset",
    "model": "$model",
    "k": $k,
    "use_maxk": $use_maxk,
    "seed": $seed,
    "gpu": $gpu,
    "status": "completed",
    "duration_seconds": \$duration,
    "best_val_acc": \$best_val_acc,
    "best_test_acc": \$best_test_acc,
    "final_train_acc": \$final_train_acc,
    "avg_epoch_time": \$avg_epoch_time,
    "log_file": "$log_file",
    "completed_at": "\$(date -Iseconds)"
}
RESULT_EOF
    
    echo "SUCCESS" > "${TEMP_DIR}/${experiment_id}.status"
else
    exit_code=\$?
    echo "\$(date): ❌ Failed $experiment_id (exit code: \$exit_code)" >> "$log_file"
    
    cat > "$result_file" << RESULT_EOF
{
    "experiment_id": "$experiment_id",
    "dataset": "$dataset",
    "model": "$model", 
    "k": $k,
    "use_maxk": $use_maxk,
    "seed": $seed,
    "gpu": $gpu,
    "status": "failed",
    "exit_code": \$exit_code,
    "failed_at": "\$(date -Iseconds)",
    "log_file": "$log_file"
}
RESULT_EOF
    
    echo "FAILED" > "${TEMP_DIR}/${experiment_id}.status"
    exit \$exit_code
fi
EOF
    
    chmod +x "$job_script"
    
    # Submit job in background
    nohup bash "$job_script" > "${TEMP_DIR}/${experiment_id}_job.log" 2>&1 &
    local job_pid=$!
    
    # Store job information
    echo "$job_pid" > "${TEMP_DIR}/${experiment_id}.pid"
    
    info "🚀 Submitted $experiment_id as background job (PID: $job_pid)"
}

# Function to check job status
check_job_status() {
    local experiment_id="$1"
    local pid_file="${TEMP_DIR}/${experiment_id}.pid"
    local status_file="${TEMP_DIR}/${experiment_id}.status"
    
    if [[ ! -f "$pid_file" ]]; then
        echo "NOT_SUBMITTED"
        return
    fi
    
    local pid=$(cat "$pid_file")
    
    if [[ -f "$status_file" ]]; then
        cat "$status_file"
    elif kill -0 "$pid" 2>/dev/null; then
        echo "RUNNING"
    else
        echo "UNKNOWN"
    fi
}

# Function to wait for jobs with progress monitoring
wait_for_jobs() {
    local max_concurrent="$1"
    local check_interval="${2:-30}"  # seconds
    
    info "Monitoring background jobs (max concurrent: $max_concurrent)"
    
    while true; do
        local running_count=0
        local completed_count=0
        local failed_count=0
        
        # Count job statuses
        for experiment_file in "${TEMP_DIR}"/*.pid; do
            [[ -f "$experiment_file" ]] || continue
            
            local experiment_id=$(basename "$experiment_file" .pid)
            local status=$(check_job_status "$experiment_id")
            
            case "$status" in
                "RUNNING") ((running_count++)) ;;
                "SUCCESS") ((completed_count++)) ;;
                "FAILED") ((failed_count++)) ;;
            esac
        done
        
        # Display status
        printf "\r${BLUE}Jobs - Running: %d, Completed: %d, Failed: %d${NC}" \
               "$running_count" "$completed_count" "$failed_count"
        
        # Check if we can start more jobs
        if [[ $running_count -lt $max_concurrent ]]; then
            return 0  # Space available
        fi
        
        # Check if all jobs are done
        if [[ $running_count -eq 0 ]]; then
            echo ""
            log "All background jobs completed"
            return 1  # All done
        fi
        
        sleep "$check_interval"
    done
}

# Function to get comprehensive job status
show_job_status() {
    echo ""
    log "Current Job Status:"
    echo "===================="
    
    local total=0
    local running=0
    local completed=0
    local failed=0
    local not_started=0
    
    printf "%-40s %-10s %-15s %-10s\n" "Experiment" "Status" "Progress" "Runtime"
    printf "%-40s %-10s %-15s %-10s\n" "----------" "------" "--------" "-------"
    
    for experiment_file in "${TEMP_DIR}"/*.pid; do
        [[ -f "$experiment_file" ]] || continue
        
        local experiment_id=$(basename "$experiment_file" .pid)
        local status=$(check_job_status "$experiment_id")
        local log_file="${LOG_DIR}/${experiment_id}.log"
        
        # Get progress info
        local progress=""
        local runtime=""
        
        if [[ -f "$log_file" ]]; then
            local current_epoch=$(grep -o "Epoch [0-9]*" "$log_file" | tail -n 1 | awk '{print $2}' || echo "0")
            progress="${current_epoch}/${NUM_EPOCHS}"
            
            local start_line=$(grep "Starting $experiment_id" "$log_file" | head -n 1)
            if [[ -n "$start_line" ]]; then
                local start_time=$(echo "$start_line" | awk '{print $1, $2}')
                local start_epoch=$(date -d "$start_time" +%s 2>/dev/null || echo "0")
                local current_epoch_time=$(date +%s)
                local elapsed=$((current_epoch_time - start_epoch))
                runtime="${elapsed}s"
            fi
        fi
        
        printf "%-40s %-10s %-15s %-10s\n" "$experiment_id" "$status" "$progress" "$runtime"
        
        ((total++))
        case "$status" in
            "RUNNING") ((running++)) ;;
            "SUCCESS") ((completed++)) ;;
            "FAILED") ((failed++)) ;;
            *) ((not_started++)) ;;
        esac
    done
    
    echo "===================="
    echo "Total: $total | Running: $running | Completed: $completed | Failed: $failed | Not Started: $not_started"
}

# Function to kill all running jobs
kill_all_jobs() {
    warning "Killing all running jobs..."
    
    for pid_file in "${TEMP_DIR}"/*.pid; do
        [[ -f "$pid_file" ]] || continue
        
        local experiment_id=$(basename "$pid_file" .pid)
        local status=$(check_job_status "$experiment_id")
        
        if [[ "$status" == "RUNNING" ]]; then
            local pid=$(cat "$pid_file")
            if kill -TERM "$pid" 2>/dev/null; then
                info "Killed job: $experiment_id (PID: $pid)"
            fi
        fi
    done
    
    sleep 2
    
    # Force kill if still running
    for pid_file in "${TEMP_DIR}"/*.pid; do
        [[ -f "$pid_file" ]] || continue
        
        local experiment_id=$(basename "$pid_file" .pid)
        local pid=$(cat "$pid_file")
        
        if kill -0 "$pid" 2>/dev/null; then
            kill -KILL "$pid" 2>/dev/null
            warning "Force killed: $experiment_id (PID: $pid)"
        fi
    done
}

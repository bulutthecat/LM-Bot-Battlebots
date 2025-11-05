
set -e 



MAX_JOBS=5




SEED=42


LOG_DIR_ROOT="./logs_hyper_tuning"


PYTHON_EXEC="./venv/bin/python"


if [ ! -f "botv2.py" ]; then
    echo "Error: botv2.py not found in the current directory."
    echo "Please ensure this script is in the same folder as botv2.py."
    exit 1
fi

if ! grep -q "TRAIN_MODEL = True" "botv2.py"; then
    echo "Error: TRAIN_MODEL is not set to True in botv2.py."
    echo "Please edit botv2.py and set 'TRAIN_MODEL = True' to enable training."
    exit 1
fi

if [ ! -f "$PYTHON_EXEC" ]; then
    echo "Error: Python executable not found at $PYTHON_EXEC"
    echo "Please update the PYTHON_EXEC variable in this script."
    exit 1
fi



BASE_LR=0.0001
BASE_GAMMA=0.99
BASE_GAE_LAMBDA=0.95
BASE_CLIP_RANGE=0.2
BASE_VF_COEF=0.5
BASE_ENT_COEF=0.0
BASE_MAX_GRAD_NORM=0.5





LR_VALUES=(0.00005 0.00007 0.00008 0.00009 0.000095 0.000105 0.00011 0.00013 0.00015 0.0002)

GAMMA_VALUES=(0.97 0.975 0.98 0.985 0.988 0.991 0.992 0.993 0.995 0.997)

GAE_LAMBDA_VALUES=(0.92 0.93 0.935 0.94 0.945 0.955 0.96 0.965 0.97 0.98)

CLIP_RANGE_VALUES=(0.1 0.12 0.14 0.16 0.18 0.22 0.24 0.26 0.28 0.3)

VF_COEF_VALUES=(0.3 0.35 0.4 0.45 0.48 0.52 0.55 0.6 0.65 0.7)

ENT_COEF_VALUES=(0.00001 0.00005 0.0001 0.0005 0.001 0.002 0.005 0.01 0.015 0.02)

MAX_GRAD_NORM_VALUES=(0.2 0.3 0.35 0.4 0.45 0.55 0.6 0.65 0.7 0.8)




cleanup() {
    echo "--- Interrupted. Killing all running training jobs... ---"
    
    pkill -P $$
    echo "Cleanup complete. Exiting."
    exit 1
}


trap cleanup SIGINT SIGTERM


wait_for_jobs() {
    
    local job_count=$(pgrep -P $$ python | wc -l)
    while [[ $job_count -ge $MAX_JOBS ]]; do
        echo "--- Max jobs ($MAX_JOBS) running. Waiting for a slot... (Next check in 15s) ---"
        
        wait -n
        sleep 10 
        job_count=$(pgrep -P $$ python | wc -l)
    done
}




launch_run() {
    local param_name=$1  
    local param_val=$2   

    
    local run_name="tune_${param_name}_${param_val}"
    local run_log_dir="${LOG_DIR_ROOT}/${run_name}"
    local model_file="${run_log_dir}/model.zip"

    mkdir -p "$run_log_dir"

    
    local args_map=(
        "lr" "$BASE_LR"
        "gamma" "$BASE_GAMMA"
        "gae_lambda" "$BASE_GAE_LAMBDA"
        "clip_range" "$BASE_CLIP_RANGE"
        "ent_coef" "$BASE_ENT_COEF"
        "vf_coef" "$BASE_VF_COEF"
        "max_grad_norm" "$BASE_MAX_GRAD_NORM"
    )

    
    local final_args=()
    for i in $(seq 0 2 $((${
        local name="${args_map[$i]}"
        local val="${args_map[$i+1]}"
        
        final_args+=("--${name}")
        if [[ "$name" == "$param_name" ]]; then
            final_args+=("$param_val")
        else
            final_args+=("$val")
        fi
    done

    
    final_args+=("--model_name" "$model_file")
    
    final_args+=("--log_dir" "$run_log_dir")

    
    if [ -n "$SEED" ]; then
        final_args+=("--seed" "$SEED")
    fi

    echo "--- LAUNCHING: $run_name (Logging to $run_log_dir/train.log) ---"

    
    "$PYTHON_EXEC" botv2.py "${final_args[@]}" > "${run_log_dir}/train.log" 2>&1 &
}


main() {
    echo "Starting hyperparameter tuning..."
    echo "Max parallel jobs: $MAX_JOBS"
    echo "Log root: $LOG_DIR_ROOT"
    echo "Python Executable: $PYTHON_EXEC"

    
    if [ -n "$SEED" ]; then
        echo "--- Using consistent seed for all runs: $SEED ---"
    else
        echo "--- Running non-deterministically (no seed set) ---"
    fi
    
    echo "WARNING: This will launch 71 training runs in total."
    echo "You can stop this script at any time with Ctrl+C."
    sleep 3

    mkdir -p "$LOG_DIR_ROOT"

    
    echo "--- TUNING: baseline ---"
    wait_for_jobs
    launch_run "baseline" "default" 

    
    echo "--- TUNING: lr ---"
    for val in "${LR_VALUES[@]}"; do
        wait_for_jobs
        launch_run "lr" "$val"
    done

    
    echo "--- TUNING: gamma ---"
    for val in "${GAMMA_VALUES[@]}"; do
        wait_for_jobs
        launch_run "gamma" "$val"
    done

    
    echo "--- TUNING: gae_lambda ---"
    for val in "${GAE_LAMBDA_VALUES[@]}"; do
        wait_for_jobs
        launch_run "gae_lambda" "$val"
    done

    
    echo "--- TUNING: clip_range ---"
    for val in "${CLIP_RANGE_VALUES[@]}"; do
        wait_for_jobs
        launch_run "clip_range" "$val"
    done

    
    echo "--- TUNING: vf_coef ---"
    for val in "${VF_COEF_VALUES[@]}"; do
        wait_for_jobs
        launch_run "vf_coef" "$val"
    done

    
    echo "--- TUNING: ent_coef ---"
    for val in "${ENT_COEF_VALUES[@]}"; do
        wait_for_jobs
        launch_run "ent_coef" "$val"
    done

    
    echo "--- TUNING: max_grad_norm ---"
    for val in "${MAX_GRAD_NORM_VALUES[@]}"; do
        wait_for_jobs
        launch_run "max_grad_norm" "$val"
    done

    echo "--- All 71 runs have been launched. ---"
    echo "--- Waiting for the final batch of jobs to complete... ---"
    wait
    echo "--- Hyperparameter tuning complete. ---"
    echo "All logs are saved in: $LOG_DIR_ROOT"
    echo ""
    echo "To view results, run:"
    echo "tensorboard --logdir $LOG_DIR_ROOT"
}


main
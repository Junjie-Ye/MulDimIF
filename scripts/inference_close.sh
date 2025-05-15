#!/bin/bash

export PYTHONPATH="Code:$PYTHONPATH"

echo "---Inference(closesource) Start---"

model="gpt-4o-2024-08-06"
data_path="Data/test.json"
result_save_dir="Code/inference/result"
max_tokens=1024
save_per_num=10
temperature=0.0
base_url=""
api_key="your_api_key_here"
log_dir="Code/inference/log"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            model="$2"
            shift 2
            ;;
        --data_path)
            data_path="$2"
            shift 2
            ;;
        --result_save_dir)
            result_save_dir="$2"
            shift 2
            ;;
        --max_tokens)
            max_tokens="$2"
            shift 2
            ;;
        --save_per_num)
            save_per_num="$2"
            shift 2
            ;;
        --temperature)
            temperature="$2"
            shift 2
            ;;
        --base_url)
            base_url="$2"
            shift 2
            ;;
        --api_key)
            api_key="$2"
            shift 2
            ;;
        --log_dir)
            log_dir="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            ;;
    esac
done

model_name=$(basename $model)
test_name=$(basename $data_path)

result_save_path="${result_save_dir}/${model_name}_${test_name}.jsonl"
log_path="${log_dir}/${model_name}_${test_name}.log"
mkdir -p $(dirname $result_save_path)
> $result_save_path
mkdir -p $(dirname $log_path)
> $log_path

log() {
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] $1" >> "$log_path" 2>&1
}

{
    log "==========================================="
    log "time: $(date +"%Y-%m-%d %H:%M:%S")"
    log "model name: $model_name"
    log "model: $model"
    log "test set name: $test_name"
    log "data path: $data_path"
    log "result save path: $result_save_path"
    log "max_tokens: $max_tokens"
    log "save_per_num: $save_per_num"
    log "temperature: $temperature"
    log "base_url: $base_url"
    log "log path: $log_path"
    log "==========================================="
} >> $log_path 2>&1

python Code/inference/inference_closesource.py \
    --model ${model} \
    --data_path ${data_path} \
    --result_save_path ${result_save_path} \
    --max_tokens ${max_tokens} \
    --save_per_num ${save_per_num} \
    --temperature ${temperature} \
    --base_url ${base_url} \
    --api_key ${api_key} \
    >> $log_path 2>&1

echo "---Inference(closesource) Finish---"

echo "---Evaluation Start---"

evaluation_save_path="${result_save_dir}/${model_name}_${test_name}.json"
python Code/evaluation/evaluation.py \
    --file_path ${result_save_path} \
    --save_path ${evaluation_save_path} \
    >> $log_path 2>&1

echo "---Evaluation Finish---"
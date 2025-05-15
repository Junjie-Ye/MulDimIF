      
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="Code:$PYTHONPATH"

echo "---Inference(vllm) Start---"

model_path="Qwen2.5-7B-Instruct"

model_type="auto"
data_path="Data/test.json"
result_save_dir="Code/inference/result"
batch_size=128
max_new_tokens=8192
save_per_num=32
temperature=0.0
sampling_times=1
tensor_parallel_size=4
gpu_memory_utilization=0.95
log_dir="Code/inference/log"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_path)
            model_path="$2"
            shift 2
            ;;
        --model_type)
            model_type="$2"
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
        --batch_size)
            batch_size="$2"
            shift 2
            ;;
        --max_new_tokens)
            max_new_tokens="$2"
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
        --sampling_times)
            sampling_times="$2"
            shift 2
            ;;
        --tensor_parallel_size)
            tensor_parallel_size="$2"
            shift 2
            ;;            
        --gpu_memory_utilization)
            gpu_memory_utilization="$2"
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

model_name=$(basename $model_path)
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
    log "model path: $model_path"
    log "model type: $model_type"
    log "test set name: $test_name"
    log "data path: $data_path"
    log "result save path: $result_save_path"
    log "batch_size: $batch_size"
    log "max_new_tokens: $max_new_tokens"
    log "save_per_num: $save_per_num"
    log "temperature: $temperature"
    log "sampling_times: $sampling_times"
    log "tensor_parallel_size: $tensor_parallel_size"
    log "gpu_memory_utilization: $gpu_memory_utilization"
    log "log path: $log_path"
    log "==========================================="
} >> $log_path 2>&1


python Code/inference/inference_vllm.py \
    --model_path ${model_path} \
    --model_type ${model_type} \
    --data_path ${data_path} \
    --result_save_path ${result_save_path} \
    --batch_size ${batch_size} \
    --max_new_tokens ${max_new_tokens} \
    --save_per_num ${save_per_num} \
    --temperature ${temperature} \
    --sampling_times ${sampling_times} \
    --tensor_parallel_size ${tensor_parallel_size} \
    --gpu_memory_utilization ${gpu_memory_utilization} \
    >> $log_path 2>&1

echo "---Inference(vllm) Finish---"

echo "---Evaluation Start---"

evaluation_save_path="${result_save_dir}/${model_name}_${test_name}.json"
python Code/evaluation/evaluation.py \
    --file_path ${result_save_path} \
    --save_path ${evaluation_save_path} \
    >> $log_path 2>&1

echo "---Evaluation Finish---"
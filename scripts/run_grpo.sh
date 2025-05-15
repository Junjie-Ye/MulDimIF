set -x
ulimit -u 16385
ulimit -n 4096
export HYDRA_FULL_ERROR=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTHONPATH="Code/evaluation:Code:Code/rl:$PYTHONPATH"

MODEL_PATH="Qwen2.5-7B-Instruct"
TRAIN_FILE="Data/train.parquet"
VAL_FILE="Data/test.parquet"
LOG_PATH="scripts/logs"
LOG_FILE="grpo_qwen2_5_7b_seq_balance.log"
MODEL_TYPE="qwen"

mkdir -p $LOG_PATH

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_FILE \
    data.val_files=$VAL_FILE \
    data.train_batch_size=32 \
    data.max_prompt_length=4096 \
    data.max_response_length=8192 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=32 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.max_num_batched_tokens=12288 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    reward_model.reward_manager=muldimif \
    reward_model.model_type=$MODEL_TYPE \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='grpo' \
    trainer.experiment_name='grpo_qwen2_5_7b_seq_balance' \
    +trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=35 \
    trainer.test_freq=35 \
    trainer.total_epochs=1 $@ 2>&1 | tee $LOG_PATH/$LOG_FILE

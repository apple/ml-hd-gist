export model_path=hd_gist_8gpu_deepspeed-llama-7b-hf-batch1_accum4-maxlen1500-sgd_d3st_prompt+date_jsonInstruct/hd_gist_8gpu_deepspeed-llama-7b-hf-batch1_accum4-maxlen1500-sgd-run-42/
export port=$(shuf -i25000-30000 -n1); \
torchrun --master_port=$port --nproc_per_node=8 train.py \
    training.do_train=false \
    training.do_eval=true \
    wandb.tag=eval_hd_gist \
    model.model_name_or_path=./exp/${model_path} \
    data.dataset_name=sgd \
    data.config_name='d3st_prompt+date_jsonInstruct' \
    model.max_length=1500 \
    +training.max_new_tokens=500 \
    training.per_device_eval_batch_size=1 \
    training.gist.add_gist_token=false \
    training.gist.add_slot_gist_token=true \
    training.gist.add_ctg_val_gist_token=true \
    training.gist.inbatch_reconstruct_ratio=0 \
    training.seed=42
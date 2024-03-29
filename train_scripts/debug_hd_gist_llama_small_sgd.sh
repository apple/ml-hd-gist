python -m train \
  +model=llama-debug \
  data.dataset_name=sgd \
  data.config_name='d3st_prompt+date_jsonInstruct' \
  training.gist.condition=gist \
  wandb.project=gist_debug \
  wandb.tag=baseline_1gpu \
  model.max_source_length=512 \
  model.max_target_length=64 \
  model.max_length=1500 \
  training.gist.add_gist_token=false \
  training.gist.num_gist_tokens=0 \
  training.per_device_train_batch_size=1 \
  training.per_device_eval_batch_size=1 \
  training.gradient_accumulation_steps=4 \
  training.eval_steps=1000 \
  training.save_steps=1000 \
  +training.max_new_tokens=80 \
  training.gist.add_slot_gist_token=true \
  training.gist.add_ctg_val_gist_token=true \
  training.seed=10 \
  training.gist.inbatch_reconstruct_ratio=0.1
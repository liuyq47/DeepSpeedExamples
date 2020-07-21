#!/bin/bash

deepspeed --num_gpus 8 --num_nodes 8 --hostfile /shared/ssh/nccl_hostname_8 \
 deepseed_train_new.py --cf bert_large_lamb.json  --max_seq_length 128 --output_dir ./128output/ \
  --print_steps 1 --deepspeed --deepspeed_transformer_kernel \
  --deepspeed_config deepspeed_bsz64k_lamb_config_seq128.json \
   --lr_schedule "EP" --lr_offset 0.0 \
   --input_dir /shared/bert/pt/phase1/ --max_steps 7038  --max_pred 20


deepspeed --num_gpus 8 --num_nodes 8 --hostfile /shared/ssh/nccl_hostname_8 \
 deepseed_train_new.py --cf bert_large_lamb.json  --max_seq_length 512 --output_dir ./512output2/ \
  --print_steps 1 --deepspeed --deepspeed_transformer_kernel \
  --deepspeed_config deepspeed_bsz32k_lamb_config_seq512.json \
  --rewarmup --lr_schedule "EP" --attention_dropout_checkpoint --lr_offset 0.0 \
  --load_training_checkpoint ./saved_models/bing_bert_large_lamb_seq/ --load_checkpoint_id epoch1_step7038 \
   --input_dir /shared/bert/pt/test --max_steps 1563 --max_pred 80
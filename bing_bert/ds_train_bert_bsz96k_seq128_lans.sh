#!/bin/bash

base_dir=`pwd`

# Where should we save checkpoints and tensorboard events?
JOB_NAME=lans_96k_seq128
OUTPUT_DIR=${base_dir}/bert_model_outputs
ENV_NAME=pytorch_bert
INPUT_DATA_DIR="/shared/bert/pt/phase1"
mkdir -p $OUTPUT_DIR

herringrun -c /shared/${ENV_NAME} python ${base_dir}/deepspeed_train_herring_ddp.py \
--cf ${base_dir}/bert_large_lans.json \
--max_seq_length 128 \
--output_dir $OUTPUT_DIR \
--deepspeed \
--deepspeed_transformer_kernel \
--print_steps 100 \
--lr_schedule "LANS" \
--job_name $JOB_NAME \
--deepspeed_config ${base_dir}/deepspeed_bsz64k_lans_config_seq128.json \
--input_dir=${INPUT_DATA_DIR} \
--max_pred=20 \
--max_steps=3519 \
--normalize_invertible \
--attention_dropout_checkpoint
&> ${JOB_NAME}.log

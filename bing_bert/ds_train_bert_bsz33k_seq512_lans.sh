#!/bin/bash

base_dir=`pwd`

# Where should we save checkpoints and tensorboard events?
JOB_NAME=lans_32k_seq512
OUTPUT_DIR=${base_dir}/bert_model_outputs
ENV_NAME=pytorch_bert
INPUT_DATA_DIR="/shared/bert/pt/phase2"

# Assumes job name in previous seq128 run, will resume training from epoch 150
CHECKPOINT_BASE_PATH=${OUTPUT_DIR}/saved_models/lans_96k_seq128
CHECKPOINT_NAME=`basename ${CHECKPOINT_BASE_PATH}/epoch1_*`
echo "checkpoint id: $CHECKPOINT_NAME"

mkdir -p $OUTPUT_DIR

herringrun -c /shared/${ENV_NAME} python ${base_dir}/deepspeed_train_herring_ddp.py \
--cf ${base_dir}/bert_large_lans.json \
--max_seq_length 512 \
--output_dir $OUTPUT_DIR \
--deepspeed \
--deepspeed_transformer_kernel \
--print_steps 100 \
--lr_schedule "LANS" \
--job_name $JOB_NAME \
--deepspeed_config ${base_dir}/deepspeed_bsz33k_lans_config_seq512.json \
--input_dir=${INPUT_DATA_DIR} \
--rewarmup \
--attention_dropout_checkpoint \
--load_training_checkpoint ${CHECKPOINT_BASE_PATH} \
--load_checkpoint_id ${CHECKPOINT_NAME} \
--max_pred=80 \
--max_steps=784
&> ${JOB_NAME}.log

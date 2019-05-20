#!/usr/bin/env bash
now=$(date +"%Y%m%d_%H%M%S")

export PATH=/mnt/lustre/lijingyu/Data_t1/anaconda2/envs/py27pt02/bin:$PATH
export TORCH_MODEL_ZOO=/mnt/lustre/DATAshare2/sunhongbin/pytorch_pretrained_models

split=0
jobname=prid-$split-scan-128-test

num_gpus=1
log_dir=logs/prid-split${split}-scan-128

if [ ! -d $log_dir ]; then
  echo create log $log_dir
  mkdir -p $log_dir
fi

srun -p P100 --job-name=$jobname --gres=gpu:$num_gpus  \
python  -u train_val.py \
        -d prid2011sequence \
        -b 32 \
        --seq_len 10 \
        --seq_srd 5 \
        --split $split \
        --features 128 \
        --a1 resnet50 \
        --lr1 1e-3 \
        --lr2 1e-3 \
        --lr3 1 \
        --train_mode cnn_rnn \
        --lr1step 20 \
        --lr2step 10 \
        --lr3step 30 \
        --test 1 \
        --logs-dir $log_dir \
        2>&1 | tee ${log_dir}/record-test-${now}.txt &\

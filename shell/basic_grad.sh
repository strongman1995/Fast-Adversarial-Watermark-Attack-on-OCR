#!/usr/bin/env bash

model_path='4.ckpt.json'
batch_size=100
clip_min=0.0
clip_max=1.0

CUDA_N=1
eps=0.2
eps_iter=5
pert_type='2'
nb_iter=1000
font_name=('Courier' 'Georgia' 'Helvetica' 'Times' 'Arial')
case=('easy' 'random' 'hard' 'insert' 'delete' 'replace-full-word')

CUDA_VISIBLE_DEVICES=$CUDA_N python basic_grad.py --model_path=$model_path \
                                                  --font_name=${font_name[4]} \
                                                  --case=${case[0]} \
                                                  --pert_type=$pert_type \
                                                  --eps=$eps \
                                                  --eps_iter=$eps_iter \
                                                  --nb_iter=$nb_iter \
                                                  --batch_size=$batch_size \
                                                  --clip_min=$clip_min \
                                                  --clip_max=$clip_max

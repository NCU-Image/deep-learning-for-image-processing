#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 17:52:45 2019
@author: akshita
"""
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--finetuning", action='store_true', default=False)
parser.add_argument("--inductive", action='store_true', default=False)
parser.add_argument("--transductive", action='store_true', default=False)
opt = parser.parse_args()

#  Inductive gzsl and zsl#final model
if opt.inductive:
    os.system('''CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=8 python train_tfvaegan_inductive.py --gammaD 1 --gammaG 1 --gzsl \
    --manualSeed 4115 --encoded_noise --preprocessing --cuda --image_embedding res101 --class_embedding att --nepoch 400 --ngh 4096 \
    --ndh 4096 --lambda1 10 --critic_iter 5 --dataset SUN --batch_size 64 --nz 102 --latent_size 102 --attSize 102 --resSize 2048 --lr 0.001 \
    --classifier_lr 0.0005 --syn_num 400 --nclass_all 717 --dataroot data \
    --recons_weight 0.01 --a1 0.1 --a2 0.01 --feedback_loop 2 --feed_lr 0.0001''')

# transductive gzsl and zsl #final model
if opt.transductive:
    os.system('''CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=8 python train_tfvaegan_transductive.py --gammaD 1 --gammaG 1 --gammaD2 1 --gammaG_D2 1 --gzsl \
    --manualSeed 4115 --encoded_noise --preprocessing --cuda --image_embedding res101 --class_embedding att --nepoch 800 --ngh 4096 \
    --ndh 4096 --lambda1 10 --critic_iter 5 --dataset SUN --batch_size 64 --nz 102 --latent_size 102 --attSize 102 --resSize 2048 --lr 0.001 \
    --classifier_lr 0.0005 --syn_num 400 --nclass_all 717 --dataroot data \
    --recons_weight 0.01 --a1 0.1 --a2 0.01 --feedback_loop 2 --feed_lr 0.0001 --dec_lr 0.0001''')
    
if opt.finetuning:
    # finetuning #Inductive
    if opt.inductive:
        os.system('''CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=8 python train_tfvaegan_inductive.py --gammaD 1 --gammaG 1 \
        --manualSeed 4115 --val_every 1 --preprocessing --cuda --fine_tune_image_embedding SUN_56_448_lr_1e_3bs_16 --class_embedding att --nepoch 400 --ngh 4096 \
        --ndh 4096 --lambda1 10 --critic_iter 5 --dataset SUN --batch_size 64 --nz 102 --latent_size 102 --attSize 102 --resSize 2048 \
        --classifier_lr 0.0005 --syn_num 400 --nclass_all 717 --dataroot data \
        --recons_weight 1.0 --dec_lr 0.00001 --a1 0.1 --a2 0.01 --feedback_loop 2 --lr 0.0001 --feed_lr 0.00001''')
    # finetuning#Transductive
    if opt.transductive:
        os.system('''CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=8 python train_tfvaegan_transductive.py --gammaD 1 --gammaG 1 --gammaD2 1 --gammaG_D2 1 \
        --manualSeed 4115 --val_every 1 --preprocessing --cuda --fine_tune_image_embedding SUN_56_448_lr_1e_3bs_16 --class_embedding att --nepoch 400 --ngh 4096 \
        --ndh 4096 --lambda1 10 --critic_iter 5 --dataset SUN --batch_size 64 --nz 102 --latent_size 102 --attSize 102 --resSize 2048 \
        --classifier_lr 0.0005 --syn_num 400 --nclass_all 717 --dataroot data \
        --recons_weight 1.0 --dec_lr 0.00001 --a1 0.1 --a2 0.01 --feedback_loop 2 --lr 0.0001 --feed_lr 0.00001''')
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

# indcutive
# final model
if opt.inductive:
    os.system('''CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=8 python train_tfvaegan_inductive.py --gammaD 10 \
    --gammaG 10 --gzsl --encoded_noise --manualSeed 9182 --preprocessing --cuda --image_embedding res101 \
    --class_embedding att --nepoch 120 --syn_num 1800 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 \
    --nclass_all 50 --dataroot data --dataset AWA2 \
    --batch_size 64 --nz 85 --latent_size 85 --attSize 85 --resSize 2048 \
    --lr 0.00001 --classifier_lr 0.001 --recons_weight 0.1 --freeze_dec \
    --feed_lr 0.0001 --dec_lr 0.0001 --feedback_loop 2 --a1 0.01 --a2 0.01''')

# transductive
# final model
if opt.transductive:
    os.system('''CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=8 python train_tfvaegan_transductive.py --gammaD 10 \
    --gammaG 10 --gammaG_D2 10 --gammaD2 10 --gzsl --encoded_noise --manualSeed 9182 --preprocessing --cuda --image_embedding res101 \
    --class_embedding att --nepoch 800 --syn_num 1800 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --nclass_all 50 --dataroot data --dataset AWA2 \
    --batch_size 64 --nz 85 --latent_size 85 --attSize 85 --resSize 2048 \
    --lr 0.00001 --classifier_lr 0.001 --recons_weight 0.1 --freeze_dec \
    --feed_lr 0.0001 --dec_lr 0.0001 --feedback_loop 2 --a1 0.01 --a2 0.01''')

if opt.finetuning:
    # indcutive finetuning
    if opt.inductive:
        os.system('''CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=8 python train_tfvaegan_inductive.py --gammaD 1 \
        --gammaG 1 --encoded_noise --gzsl --manualSeed 9182 --preprocessing --cuda --image_embedding AWA_1_448_lr_1e_3bs_16 \
        --class_embedding att --nepoch 800 --syn_num 1800 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --nclass_all 50 --dataroot data --dataset AWA2 \
        --batch_size 64 --nz 85 --latent_size 85 --attSize 85 --resSize 2048 \
        --lr 0.0001 --classifier_lr 0.001 --recons_weight 1 --freeze_dec \
        --feed_lr 0.00001 --dec_lr 0.00001 --feedback_loop 2 --a1 0.1 --a2 0.1''')
    # transductive finetuning
    if opt.transductive:
        os.system('''CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=8 python train_tfvaegan_transductive.py --gammaD 1 \
        --gammaG 1 --gammaG_D2 1 --gammaD2 1 --encoded_noise --gzsl --manualSeed 9182 --preprocessing --cuda --image_embedding AWA_1_448_lr_1e_3bs_16 \
        --class_embedding att --nepoch 800 --syn_num 1800 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --nclass_all 50 --dataroot data --dataset AWA2 \
        --batch_size 64 --nz 85 --latent_size 85 --attSize 85 --resSize 2048 \
        --lr 0.0001 --classifier_lr 0.001 --recons_weight 1 --freeze_dec \
        --feed_lr 0.00001 --dec_lr 0.00001 --feedback_loop 2 --a1 0.1 --a2 0.1''')
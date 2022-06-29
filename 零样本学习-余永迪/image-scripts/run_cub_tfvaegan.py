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

#Indcutive
#final model
if opt.inductive:
    os.system('''CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=8  python train_tfvaegan_inductive.py --gammaD 10 --gammaG 10 \
    --gzsl --manualSeed 3483 --encoded_noise --preprocessing --cuda --image_embedding res101 --class_embedding att \
    --nepoch 300 --ngh 4096 --ndh 4096 --lr 0.0001 --classifier_lr 0.001 --lambda1 10 --critic_iter 5 --dataroot data --dataset CUB \
    --nclass_all 200 --batch_size 64 --nz 312 --latent_size 312 --attSize 312 --resSize 2048 --syn_num 300 \
    --recons_weight 0.01 --a1 1 --a2 1 --feed_lr 0.00001 --dec_lr 0.0001 --feedback_loop 2''')

#transductive
#final model
if opt.transductive:
    os.system('''CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=8  python train_tfvaegan_transductive.py --gammaD 10 --gammaG 10 --gammaD2 10 --gammaG_D2 10 \
    --gzsl --manualSeed 3483 --encoded_noise --preprocessing --cuda --image_embedding res101 --class_embedding att \
    --nepoch 500 --ngh 4096 --ndh 4096 --lr 0.0001 --classifier_lr 0.001 --lambda1 10 --critic_iter 5 --dataroot data --dataset CUB \
    --nclass_all 200 --batch_size 64 --nz 312 --latent_size 312 --attSize 312 --resSize 2048 --syn_num 300 \
    --recons_weight 0.01 --a1 1 --a2 1 --feed_lr 0.00001 --dec_lr 0.0001 --feedback_loop 2''')

if opt.finetuning:
    #Indcutive FINETUNING
    if opt.inductive:
        os.system('''CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=8  python train_tfvaegan_inductive.py --gzsl --gammaD 10 --gammaG 10 \
        --manualSeed 3483 --preprocessing --encoded_noise --cuda --image_embedding res101_150_cub_448_lr_1e_3bs_16 --class_embedding att \
        --nepoch 300 --ngh 4096 --ndh 4096 --lr 0.0001 --classifier_lr 0.001 --lambda1 10 --critic_iter 5 --dataroot data --dataset CUB \
        --nclass_all 200 --batch_size 64 --nz 312 --latent_size 312 --attSize 312 --resSize 2048 --syn_num 300 \
        --feed_lr 0.0001 --recons_weight 0.1 --feedback_loop 2 --a1 1 --a2 1 --dec_lr 0.0001''')
    
    #transductive FINETUNING
    elif opt.transductive:
        os.system('''CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=8  python train_tfvaegan_transductive.py --gzsl --gammaD 10 --gammaG 10 --gammaD2 10 --gammaG_D2 10\
        --manualSeed 3483 --preprocessing --encoded_noise --cuda --image_embedding res101_150_cub_448_lr_1e_3bs_16 --class_embedding att \
        --nepoch 300 --ngh 4096 --ndh 4096 --lr 0.0001 --classifier_lr 0.001 --lambda1 10 --critic_iter 5 --dataroot data --dataset CUB \
        --nclass_all 200 --batch_size 64 --nz 312 --latent_size 312 --attSize 312 --resSize 2048 --syn_num 300 \
        --feed_lr 0.0001 --recons_weight 0.1 --feedback_loop 2 --a1 1 --a2 1 --dec_lr 0.0001''')

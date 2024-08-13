#!/usr/bin/env bash

python main.py  --dataset twitch --device 1 --train_epochs 100 --test_epochs 3 --entropy_weight 5 --replay_weight 10 --vae_weight 1 --edge_weight 1 --lr_model 0.0001 --wd_model 0.0005 --lr_mem 0.001 --wd_mem 0.0005 --warmup_epochs 2 --inner_loop 10 --mt 0.2 --syn_ratio 0.15 --seed 0

python main.py  --dataset fb100 --device 1 --train_epochs 150 --test_epochs 10 --entropy_weight 5 --replay_weight 1 --vae_weight 1 --edge_weight 5 --lr_model 0.0001 --wd_model 0.0005 --lr_mem 0.001 --wd_mem 0.0005 --warmup_epochs 9 --inner_loop 1 --mt 0.25 --syn_ratio 0.05 --seed 0

python main.py  --dataset ogbn-arxiv --device 1 --model sage --train_epochs 200 --test_epochs 3 --entropy_weight 5 --replay_weight 10 --vae_weight 1 --edge_weight 1 --lr_model 0.0001 --wd_model 0.0005 --lr_mem 0.001 --wd_mem 0.0005 --warmup_epochs 2 --inner_loop 10 --mt 0.2 --syn_ratio 0.15 --seed 0

python main.py  --dataset elliptic --device 1 --train_epochs 200 --test_epochs 5 --entropy_weight 1 --replay_weight 2 --vae_weight 2 --edge_weight 1 --lr_model 0.0001 --wd_model 0.0005 --lr_mem 0.001 --wd_mem 0.0005 --warmup_epochs 3 --inner_loop 50 --mt 0.3 --syn_ratio 0.15 --seed 0




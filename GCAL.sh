#!/usr/bin/env bash

python main.py --device=0 --dataset=twitch --model=GCN  --seed=0 
python main.py --device=0 --dataset=ogbn-arxiv --model=GCN  --seed=0
python main.py --device=0 --dataset=elliptic --model=GCN  --seed=0 
python main.py --device=0 --dataset=fb100 --model=GCN  --seed=0 

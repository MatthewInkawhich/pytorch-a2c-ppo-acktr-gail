#!/bin/bash

python main.py --run-name "ppo_spaceinvaders_base" --env-name "SpaceInvadersNoFrameskip-v4" --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --log-interval 10 --use-linear-lr-decay --entropy-coef 0.01 #--noisy-train

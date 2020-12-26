#!/usr/bin/env bash

python train.py --model global_sum_embedding_mlp --learning_rate 5e-3 --alpha 0.05 --l2_attention 1e-5 --l2_lstm 1e-8 --alias twitter_lr5e-3_l2a1e-5_l2l0 --batch_size 256 --max_epoch 50
python train.py --model global_sum_embedding_mlp --learning_rate 2e-2 --alpha 0.05 --l2_attention 1e-5 --l2_lstm 1e-8 --alias twitter_lr2e-2_l2a1e-5_l2l1e-8 --batch_size 256 --max_epoch 50
python train.py --model global_sum_embedding_mlp --learning_rate 5e-3 --alpha 0.05 --l2_attention 1e-5 --l2_lstm 1e-8 --alias twitter_lr5e-3_l2a1e-5_l2l1e-8 --batch_size 256 --max_epoch 50
python train.py --model global_sum_embedding_mlp --learning_rate 1e-2 --alpha 0.05 --l2_attention 1e-5 --l2_lstm 1e-8 --alias twitter_lr1e-2_l2a1e-5_l2l1e-8 --batch_size 256 --max_epoch 50

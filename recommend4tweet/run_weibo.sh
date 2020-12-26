python train2.py --model global_sum_embedding_mlp --learning_rate 5e-3 --alpha 0.3 --l2_attention 1e-5 --l2_lstm 1e-8 --alias weibo_lr5e-3_l2a1e-5_l2l1e-8 --batch_size 32 --max_epoch 100
python train2.py --model global_sum_embedding_mlp --learning_rate 0.02 --alpha 0.3 --l2_attention 1e-5 --l2_lstm 1e-8 --alias weibo_lr2e-2_l2a1e-5_l2l1e-8 --batch_size 32 --max_epoch 100
#python train2.py --model global_sum_embedding_mlp --learning_rate 0.01 --alpha 0.3 --l2_attention 1e-5 --l2_lstm 1e-8 --alias weibo_lr1e-2_l2a1e-5_l2l1e-8 --batch_size 256 --max_epoch 100


python train2.py --model global_sum_embedding --learning_rate 5e-3 --alpha 0.3 --l2_attention 1e-5 --l2_embedding 0 --alias weibo_lr5e-3_l2a1e-5_l2l0 --batch_size 64 --max_epoch 50
python train2.py --model global_sum_embedding --learning_rate 5e-3 --alpha 0.3 --l2_attention 0 --l2_embedding 0 --alias weibo_lr5e-3_l2a0_l2l1e-8 --batch_size 64 --max_epoch 50


#python train2.py --model global_sum_embedding --learning_rate 0.01 --alpha 0.3 --l2_attention 1e-5 --l2_embedding 1e-8 --alias weibo_lr1e-2_l2a1e-5_l2l1e-8 --batch_size 64 --max_epoch 50
#python train2.py --model global_sum_embedding --learning_rate 1e-3 --alpha 0.3 --l2_attention 1e-5 --l2_embedding 1e-8 --alias weibo_lr1e-3_l2a1e-5_l2l1e-8 --batch_size 64 --max_epoch 50
#python train2.py --model global_sum_embedding --learning_rate 0.02 --alpha 0.3 --l2_attention 1e-5 --l2_embedding 1e-8 --alias weibo_lr2e-2_l2a1e-5_l2l1e-8 --batch_size 64 --max_epoch 50
#python train2.py --model global_sum_embedding --learning_rate 5e-3 --alpha 0.3 --l2_attention 1e-5 --l2_embedding 0 --alias weibo_lr5e-3_l2a1e-5_l2l0 --batch_size 64 --max_epoch 50
#python train2.py --model global_sum_embedding --learning_rate 5e-3 --alpha 0.3 --l2_attention 1e-8 --l2_embedding 0 --alias weibo_lr5e-3_l2a1e-8_l2l0 --batch_size 64 --max_epoch 50
#python train2.py --model global_sum_embedding --learning_rate 0.01 --alpha 0.3 --l2_attention 1e-4 --l2_embedding 1e-8 --alias weibo_lr1e-2_l2a1e-4_l2l1e-8 --batch_size 64 --max_epoch 50
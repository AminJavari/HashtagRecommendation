# Weakly Supervised Attention for Hashtag Recommendation using Graph Data

### For graph-based gmf model
- Related Folder:
\recommend4group
- How to use: 
    ```shell
    python train.py --model gmf --learning_rate 0.005 --alpha 0.03 --alias supervision_user_32_0.03 --batch_size 32
    ```

### Fusion model for microblog-hashtag task:
- Related Folder:
\recommend4tweet
- How to use:
    ```shell
    python train.py --model global_sum_embedding --learning_rate 5e-3 --alpha 0 --l2_attention 1e-5 --l2_lstm 1e-8 --alias twitter_com_alpha0_lr5e-3_l2a1e-5_l2l0 --batch_size 256 --max_epoch 50
    ```
    
### Fusion model for user-hashtag task:
- Related Folder:
\recommend4user
- How to use:
    ```shell
    python train.py --model global_sum_embedding --learning_rate 5e-3 --alpha 0 --l2_attention 1e-5 --l2_lstm 1e-8 --alias twitter_com_alpha0_lr5e-3_l2a1e-5_l2l0 --batch_size 256 --max_epoch 50
    ```

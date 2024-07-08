python -m train.train \
       --wandb_pj_name         fog-challenge \
       --entity                chrenx \
       --save_best_model       \
       --seed                  11 \
       \
       --exp_name              transformer_v3 \
       --cuda_id               3 \
       \
       --description           "transformer ,training=False, win 512, kaggle" \
       --optimizer             "adam" \
       --train_datasets        kaggle_pd_data \
       --feats                 LowerBack_Acc_X LowerBack_Acc_Y LowerBack_Acc_Z \
       --train_num_steps       50000 \
       --batch_size            128 \
       --random_aug            \
       --max_grad_norm         1 \
       --weight_decay          1e-6 \
       --grad_accum_step       1 \
       --window                2976 \
       --learning_rate         26e-5 \
       --lr_scheduler          ReduceLROnPlateau \
       --preload_gpu \
       --disable_wandb \
    #    --penalty_cost          2.5 \
    #    --disable_scheduler
    #    --window                -1 \
    #    --disable_wandb
    #    --preload_gpu \
    #    --disable_wandb         #!!!
    #    --preload_gpu           \      
    #    --weight_decay                   1e-6 \


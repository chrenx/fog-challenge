python -m train.train_transformer_v2 \
       --version               2 \
       --device                2 \
       --exp_name              transformer_v2 \
       --wandb_pj_name         fog-challenge \
       --entity                chrenx \
       --save_best_model       \
       --description           "transformer ,training=False, win 2976, kaggle" \
       --optimizer             "adam" \
       --seed                  11 \
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
    #    --disable_scheduler
    #    --window                -1 \
    #    --disable_wandb
    #    --preload_gpu \
    #    --disable_wandb         #!!!
    #    --preload_gpu           \      
    #    --weight_decay                   1e-6 \
    #    --learning_rate                  0.0006 \


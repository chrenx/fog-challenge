python -m train.train_unet_v1 \
       --version               1 \
       --device                3 \
       --exp_name              unet_v1 \
       --wandb_pj_name         fog-challenge \
       --entity                chrenx \
       --save_best_model       \
       --description           "unet lab only, window 6976, has scheduler, kaggle" \
       --optimizer             "adam" \
       --seed                  11 \
       --train_datasets        kaggle_pd_data \
       --feats                 LowerBack_Acc_X LowerBack_Acc_Y LowerBack_Acc_Z \
       --train_num_steps       100000 \
       --batch_size            128 \
       --random_aug            \
       --max_grad_norm         1 \
       --weight_decay          1e-6 \
       --grad_accum_step       1 \
       --window                6976 \
       --preload_gpu
    #    --window                -1 \
    #    --disable_wandb
    #    --preload_gpu \
    #    --disable_wandb         #!!!
    #    --preload_gpu           \      
    #    --weight_decay                   1e-6 \
    #    --learning_rate                  0.0006 \


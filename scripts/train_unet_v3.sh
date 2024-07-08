python -m train.train_unet_v3 \
       --version               3 \
       --device                2 \
       --exp_name              unet_v3 \
       --wandb_pj_name         fog-challenge \
       --entity                chrenx \
       --save_best_model       \
       --description           "unet lab only, win 6304, change tvt ratio, kaggle" \
       --optimizer             "adam" \
       --seed                  11 \
       --train_datasets        kaggle_pd_data \
       --feats                 LowerBack_Acc_X LowerBack_Acc_Y LowerBack_Acc_Z \
       --train_num_steps       10000 \
       --batch_size            128 \
       --random_aug            \
       --max_grad_norm         1 \
       --weight_decay          1e-6 \
       --grad_accum_step       1 \
       --window                6304 \
       --learning_rate         26e-4 \
       --lr_scheduler          ReduceLROnPlateau \
       --preload_gpu           \
    #    --disable_scheduler
    #    --window                -1 \
    #    --disable_wandb
    #    --preload_gpu \
    #    --disable_wandb         #!!!
    #    --preload_gpu           \      
    #    --weight_decay                   1e-6 \
    #    --learning_rate                  0.0006 \


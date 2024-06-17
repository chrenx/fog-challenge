python -m train.train_transformer_bilstm_v2 \
       --version               2 \
       --device                0 \
       --exp_name              finetune2 \
       --wandb_pj_name         fog-challenge \
       --entity                chrenx \
       --save_best_model       \
       --description           "just kaggle data, finetune" \
       --optimizer             "adamw" \
       --seed                  11 \
       --fog_model_dim         320 \
       --penalty_cost          2 \
       --train_datasets        kaggle_pd_data \
       --num_feats             3 \
       --random_aug            \
       --train_num_steps       100000 \
       --fog_model_num_encoder_layers   4 \
       --weight_decay                   1e-6 \
       --learning_rate                  0.0006 \
       --fog_model_first_dropout        0.2 \
       --fog_model_encoder_dropout      0.2 \
       --fog_model_mha_dropout          0.2 \
    #    --disable_wandb         #!!!
    #    --data_name             train1_dataset_fog_release_blks15552_ps18.p val1_dataset_fog_release_blks15552_ps18.p \             


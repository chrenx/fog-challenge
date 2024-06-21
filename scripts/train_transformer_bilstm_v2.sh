python -m train.train_transformer_bilstm_v2 \
       --version               2 \
       --device                3 \
       --exp_name              transformer_bilstm_v2 \
       --wandb_pj_name         fog-challenge \
       --entity                chrenx \
       --save_best_model       \
       --description           "different way to validate, BS 128 adam, kaggle" \
       --optimizer             "adam" \
       --seed                  11 \
       --fog_model_dim         320 \
       --penalty_cost          2 \
       --train_datasets        kaggle_pd_data \
       --num_feats             3 \
       --train_num_steps       100000 \
       --batch_size            128 \
       --random_aug            \
    #    --disable_wandb         #!!!            
    #    --fog_model_num_encoder_layers   4 \
    #    --weight_decay                   1e-6 \
    #    --learning_rate                  0.0006 \
    #    --fog_model_first_dropout        0.2 \
    #    --fog_model_encoder_dropout      0.2 \
    #    --fog_model_mha_dropout          0.2 \
    #    --data_name             train1_dataset_fog_release_blks15552_ps18.p val1_dataset_fog_release_blks15552_ps18.p \             


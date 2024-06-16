python -m train.train_transformer_bilstm_v2 \
       --version               2 \
       --device                1 \
       --exp_name              transfomer_bilstm_v2 \
       --wandb_pj_name         fog-challenge \
       --entity                chrenx \
       --save_best_model       \
       --description           "just kaggle data" \
       --optimizer             "adamw" \
       --seed                  11 \
       --fog_model_dim         320 \
       --penalty_cost          2 \
       --random_aug            \
       --train_datasets        kaggle_pd_data \
       --disable_wandb         #!!!  
    #    --data_name             train1_dataset_fog_release_blks15552_ps18.p val1_dataset_fog_release_blks15552_ps18.p \             


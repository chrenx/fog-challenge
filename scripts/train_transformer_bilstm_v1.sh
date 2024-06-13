python -m train.train_transformer_bilstm_v1 \
       --device                7 \
       --exp_name              transfomer_bilstm \
       --wandb_pj_name         fog-challenge \
       --entity                chrenx \
       --save_best_model       \
       --data_name             train1_dataset_fog_release_blks15552_ps18.p val1_dataset_fog_release_blks15552_ps18.p \
       --description           "penalty 2, model dim 320" \
       --optimizer             "adamw" \
       --seed                  11 \
       --fog_model_dim         320 \
       --penalty_cost          2 \
    #    --disable_wandb         #!!!               


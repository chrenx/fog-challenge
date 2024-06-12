python -m train.train_transformer_bilstm_v1 \
       --device                7 \
       --exp_name              transfomer_bilstm \
       --wandb_pj_name         fog-challenge \
       --entity                chrenx \
       --save_best_model        \
       --save_every_n_epoch    100 \
       --data_name             train1_dataset_fog_release_blks15552_ps18.p val1_dataset_fog_release_blks15552_ps18.p \
       --description           "using AdamW" \
       --optimizer             "adamw"
    #    --disable_wandb         #!!!               


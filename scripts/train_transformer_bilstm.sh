python -m train.train_transformer_bilstm \
       --device                6 \
       --exp_name              transfomer_bilstm \
       --wandb_pj_name         fog-challenge \
       --entity                chrenx \
       --save_best_model        \
       --data_name             train1_dataset_fog_release_blks15552_ps18.p val1_dataset_fog_release_blks15552_ps18.p \
       --disable_wandb         #!!!               


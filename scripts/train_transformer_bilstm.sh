python -m train.train_transformer_bilstm \
       --device                cpu \
       --exp_name              transfomer_bilstm \
       --wandb_pj_name         fog-challenge \
       --entity                chrenx \
       --save_and_sample_every 10 \
       --train_num_steps       10 \
       --save_best_model        \
       --disable_wandb          #!!!!


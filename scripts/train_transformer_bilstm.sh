python -m train.train_transformer_bilstm \
       --device=7 \
       --exp_name="transfomer_bilstm" \
       --wandb_pj_name="fog-challenge" \
       --entity="chrenx" \
       --save_best_model \
       --save_and_sample_every 10 \
       --train_num_steps 10 \
       --disable_wandb #!!!!
       # --batch_size=32
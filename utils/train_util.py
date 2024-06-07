import os, yaml

def cycle_dataloader(dl):
    while True:
        for data in dl:
            yield data

def save_group_args(opt):
    # Save running settings
    # grouped_args = {
    #     'project_info': {
    #         'project': opt.project,
    #         'exp_name': opt.exp_name,
    #         'save_dir': opt.save_dir,
    #         'weights_dir': opt.weights_dir
    #     },
    #     'wandb_setup': {
    #         'disable_wandb': opt.disable_wandb,
    #         'wandb_pj_name': opt.wandb_pj_name,
    #         'entity': opt.entity,
    #     },
    #     'data_info': {
    #         'root_dpath': opt.root_dpath,
    #     },
    #     'gpu_info': {
    #         'device': opt.device,
    #         'device_info': opt.device_info,
    #     },
    #     'training_monitor': {
    #         'save_and_sample_every': opt.save_and_sample_every,
    #         'save_best_model': opt.save_best_model,
    #     },
    #     'hyperparameters': {
    #         'seed': opt.seed,
    #         'batch_size': opt.batch_size,
    #         'learning_rate': opt.learning_rate,
    #         'train_num_steps': opt.train_num_steps
    #     },
    # }
    # with open(os.path.join(opt.save_dir, 'opt.yaml'), 'w') as f:
    #     yaml.safe_dump(grouped_args, f, sort_keys=False)
    with open(os.path.join(opt.save_dir, 'opt.yaml'), 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=False)
        

import argparse, logging, os, random, shutil, yaml
from datetime import datetime

import torch
import numpy as np

MYLOGGER = logging.getLogger()


def run_train(opt):
    pass

def parse_opt():
    parser = argparse.ArgumentParser()
    
    # project information: names
    project_info = parser.add_argument_group('project_info')
    project_info.add_argument('--project', default='runs/train', help='project/name')
    project_info.add_argument('--exp_name', default='transfomer_bilstm', 
                                            help='save to project/name')

    # wandb setup
    wandb_setup = parser.add_argument_group('wandb_setup')
    wandb_setup.add_argument('--disable_wandb', action='store_true')
    wandb_setup.add_argument('--wandb_pj_name', type=str, default='', help='wandb project name')
    wandb_setup.add_argument('--entity', default='', help='W&B entity')

    # data path
    data_info = parser.add_argument_group('data_info')
    data_info.add_argument('--root_dpath', default='data/rectified_data', 
                                           help='root data directory')
    
    # GPU
    gpu_info = parser.add_argument_group('gpu_info')
    gpu_info.add_argument('--device', default='0', help='assign gpu')

    # hyperparameters
    hyperparam = parser.add_argument_group('hyperparameters')
    hyperparam.add_argument('--seed', type=int, default=42, 
                                      help='set up seed for torch, numpy, random')
    hyperparam.add_argument('--batch_size', type=int, default=64, help='batch size')
    hyperparam.add_argument('--learning_rate', type=float, default=2e-4, 
                                               help='generator_learning_rate')
    hyperparam.add_argument('--train_num_steps', type=int, default=8000000, 
                                                 help='number of training steps')
    
    # training utils
    train_utils = parser.add_argument_group('train_utils')
    train_utils.add_argument('--save_and_sample_every', type=int, default=20000, 
                                                        help='save and sample')
    train_utils.add_argument('--save_best_model', action='store_true', 
                                                  help='save best model during training')

    opt = parser.parse_args()
    
    assert torch.cuda.is_available(), "**** No available GPUs."
    device_info = torch.cuda.get_device_name(int(opt.device))  
    gpu_info = parser.add_argument_group('gpu_info')  
    gpu_info.add_argument('--device_info', default=device_info, help='assign gpu')
    
    opt = parser.parse_args()
    
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    cur_time = datetime.now()
    cur_time = '{:%Y_%m_%d_%H:%M:%S}.{:02.0f}'.format(cur_time, cur_time.microsecond / 10000.0)
    opt.save_dir = os.path.join(opt.project, opt.exp_name, cur_time)
    opt.device = f"cuda:{opt.device}"

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    # Create dir
    os.makedirs(opt.save_dir, exist_ok=True)

    # Save some important code
    source_files = ['train/train_transformer_bilstm.py']
    codes_dest = os.path.join(opt.save_dir, 'codes')
    os.makedirs(codes_dest)
    for file_dir in source_files:
        shutil.copy2(file_dir, codes_dest)
        
    # Create weight dir to store model weights
    opt.weights_dir = os.path.join(opt.save_dir, 'weights')
    os.makedirs(opt.weights_dir, exist_ok=True)
    
    # Save running settings
    with open(os.path.join(opt.save_dir, 'opt.yaml'), 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=False)
        
    logging.basicConfig(filename=os.path.join(opt.save_dir,"training_info.log"),
                        filemode='a',
                        format='%(asctime)s.%(msecs)02d %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d-%H:%M:%S',
                        level=os.environ.get("LOGLEVEL", "INFO"))
    MYLOGGER.setLevel(logging.INFO)
    
    MYLOGGER.info(f"Running at {cur_time}")
    MYLOGGER.info(f"Using device: {opt.device}")

    run_train(opt)


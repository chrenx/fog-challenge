import argparse, logging, os, random, shutil
from datetime import datetime

import torch, wandb
import numpy as np

from data.fog_dataset import FoGDataset
from models.transformer_bilstm import TransformerBiLSTM
from utils.config import FEATURES_LIST
from utils.train_util import cycle_dataloader, save_group_args


MYLOGGER = logging.getLogger()

def cycle_dataloader(dl):
    while True:
        for data in dl:
            yield data

class Trainer(object):
    def __init__(self, model, opt):
        super().__init__()
        if not opt.disable_wandb:
            MYLOGGER.info("Initialize W&B")
            wandb.init(config=opt, project=opt.wandb_pj_name, entity=opt.entity, 
                       name=opt.exp_name, dir=opt.save_dir)
        self.model = model
        self.opt = opt
        self.prepare_dataloader()
        
    def prepare_dataloader(self):
        MYLOGGER.info("Loading training data ...")
        full_dataset = FoGDataset(self.opt)
        
        #TODO: split dataset 80% for train and 20% for validation
    

def run_train(opt):
    model = TransformerBiLSTM(opt)
    trainer = Trainer(model, opt)
    pass

def parse_opt():
    parser = argparse.ArgumentParser()
    
    # project information: names ===============================================
    parser.add_argument('--project', default='runs/train', help='project/name')
    parser.add_argument('--exp_name', default='transfomer_bilstm', 
                                            help='save to project/name')

    # wandb setup ==============================================================
    parser.add_argument('--disable_wandb', action='store_true')
    parser.add_argument('--wandb_pj_name', type=str, default='fog-challenge', 
                                           help='wandb project name')
    parser.add_argument('--entity', default='', help='W&B entity or username')

    # data path
    parser.add_argument('--root_dpath', default='data/rectified_data', 
                                        help='directory that contains different processed datasets')
    
    # GPU ======================================================================
    parser.add_argument('--device', default='0', help='assign gpu')
    parser.add_argument('--device_info', type=str, default='')
    
    # training monitor =========================================================
    parser.add_argument('--save_and_sample_every', type=int, default=20000, 
                                                        help='save and sample')
    parser.add_argument('--save_best_model', action='store_true', 
                                                  help='save best model during training')

    # hyperparameters ==========================================================
    parser.add_argument('--seed', type=int, default=42, 
                                      help='set up seed for torch, numpy, random')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-4, 
                                               help='generator_learning_rate')
    parser.add_argument('--train_num_steps', type=int, default=8000000, 
                                                 help='number of training steps')
    
    parser.add_argument('--block_size', type=int, default=15552)
    parser.add_argument('--block_stride', type=int, default=15552 // 16) # 972
    parser.add_argument('--patch_size', type=int, default=18)

    parser.add_argument('--fog_model_input_dim', type=int, default=18*len(FEATURES_LIST))
    parser.add_argument('--fog_model_dim', type=int, default=320)
    parser.add_argument('--fog_model_num_heads', type=int, default=8)
    parser.add_argument('--fog_model_num_encoder_layers', type=int, default=5)
    parser.add_argument('--fog_model_num_lstm_layers', type=int, default=2)
    parser.add_argument('--fog_model_first_dropout', type=float, default=0.1)
    parser.add_argument('--fog_model_encoder_dropout', type=float, default=0.1)
    parser.add_argument('--fog_model_mha_dropout', type=float, default=0.0)
    parser.add_argument('--feats_list', type=str, nargs='+', default=FEATURES_LIST)
    
    # file tracker =============================================================
    parser.add_argument('--save_dir', type=str, default='')
    parser.add_argument('--weights_dir', type=str, default='')

    opt = parser.parse_args()
        
    return opt

if __name__ == "__main__":
    
    assert torch.cuda.is_available(), "**** No available GPUs."
    
    opt = parse_opt()
    cur_time = datetime.now()
    cur_time = '{:%Y_%m_%d_%H:%M:%S}.{:02.0f}'.format(cur_time, cur_time.microsecond / 10000.0)
    opt.save_dir = os.path.join(opt.project, opt.exp_name, cur_time)
    opt.device_info = torch.cuda.get_device_name(int(opt.device)) 
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
    
    logging.basicConfig(filename=os.path.join(opt.save_dir,"training_info.log"),
                        filemode='a',
                        format='%(asctime)s.%(msecs)02d %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d-%H:%M:%S',
                        level=os.environ.get("LOGLEVEL", "INFO"))
    MYLOGGER.setLevel(logging.INFO)
    MYLOGGER.info(f"Running at {cur_time}")
    MYLOGGER.info(f"Using device: {opt.device}")
    
    save_group_args(opt)

    run_train(opt)


import importlib, logging, math, os, random, shutil, sys, yaml
from datetime import datetime

import torch
import numpy as np


#* Data Augmentation ===============================================================================
def norm_axis(a,b,c):
    newa=a/(math.sqrt(float(a*a+b*b+c*c)))
    newb=b/(math.sqrt(float(a*a+b*b+c*c)))
    newc=c/(math.sqrt(float(a*a+b*b+c*c)))
    return ([newa,newb,newc])

def rotation_matrix(axis, theta):
    axis = np.asarray(axis)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)], 
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)], 
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

def rotateC(image,theta,a,b,c): ## theta: angle, a, b, c, eular vector
    axis=norm_axis(a,b,c)
    imagenew=np.dot(image, rotation_matrix(axis,theta))
    return imagenew
#* ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

def is_valid_one_hot(matrix):
    # Check if the matrix has shape (N, 3)
    if matrix.size(1) != 3:
        return False

    # Check if each row has exactly one '1' and the rest '0'
    for row in matrix:
        if torch.sum(row) != 1 or not torch.all((row == 0) | (row == 1)):
            return False

    return True

def count_model_parameters(model): 
    return sum(p.numel() for p in model.parameters() if p.requires_grad) 

def create_training_folders(opt):
    os.makedirs(opt.save_dir, exist_ok=True)
    os.makedirs(opt.weights_dir, exist_ok=True)
    os.makedirs(opt.codes_dir, exist_ok=True)
    
def cycle_dataloader(dl):
    while True:
        for data in dl:
            # print(data.shape)
            yield data

def get_cur_time():
    cur_time = datetime.now()
    cur_time = '{:%Y_%m_%d_%H_%M_%S}_{:02.0f}'.format(cur_time, 
                                                      cur_time.microsecond / 10000.0)
    return cur_time

def print_initial_info(opt, redirect_file=None, model=None):
    def print_initial_info_helper(opt):
        if model is not None:
            print()
            print("--------------------- Model Arch ---------------------")
            print(model)
        print()
        print("--------------------- Training Setup ---------------------")
        print(f"Current Time:    {opt.cur_time}")
        print(f"GPU:             {opt.device}, {opt.device_info}")
        print(f"Description:     {opt.description}")
        print(f"Save dir:        {opt.save_dir}")
        print(f"Use WandB:       {not opt.disable_wandb}")
        print(f"Preload2GPU:     {opt.preload_gpu}")
        print(f"Train dname:     {opt.train_dname}")
        print(f"Val dname:       {opt.val_dname}")
        print(f"Test dname:      {opt.test_dname}")
        print(f"Train n trials:  {opt.train_n_trials}")
        print(f"Train n batch:   {opt.train_n_batch}")
        print(f"Val n trials:    {opt.val_n_trials}")
        print(f"Val n batch:     {opt.val_n_batch}")
        print(f"Test n trials:   {opt.test_n_trials}")
        print(f"Test n batch:    {opt.test_n_batch}")
        print(f"Window:          {opt.window}")
        print()
        print(f"--------------------- {opt.exp_name} ---------------------")
        print(f"# of params:  {opt.model_num_params}")
        print()
        
    if redirect_file is not None:
        file = open(os.path.join(opt.save_dir, redirect_file), 'w')
        sys.stdout = file
    
    print_initial_info_helper(opt)
    
    sys.stdout = sys.__stdout__
        
def save_codes(opt):
    # Save some important code
    source_files = [f'scripts/train.sh',
                    f'train/train.py', 
                    f'utils/train_util.py', 'utils/config.py',
                    f'data/fog_dataset.py',
                    f'models/{opt.exp_name}.py']
    for file_dir in source_files:
        shutil.copy2(file_dir, opt.codes_dir)

def save_group_args(opt):
    def is_serializable(value):
        try:
            yaml.safe_dump(value)
            return True
        except yaml.YAMLError:
            return False
    serializable_dict = {k: v for k, v in vars(opt).items() if is_serializable(v)}
    
    with open(os.path.join(opt.save_dir, 'opt.yaml'), 'w') as f:
        yaml.safe_dump(serializable_dict, f, sort_keys=False)

def set_redirect_printing(opt):
    training_info_log_path = os.path.join(opt.save_dir,"training_info.log")
    
    if not opt.disable_wandb:
        sys.stdout = open(training_info_log_path, "w")
        
        logging.basicConfig(filename=os.path.join(training_info_log_path),
                    filemode='a',
                    format='%(asctime)s.%(msecs)02d %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d-%H:%M:%S',
                    level=os.environ.get("LOGLEVEL", "INFO"))
    else:
        logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

def set_seed(opt):
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)


class ModelLoader:
    def __init__(self, exp_name, opt=None):
        self.model = None
        self.load_model(exp_name, opt)
        
    def load_model(self, exp_name, opt):
        tmp = exp_name.split('_')
        model_name = ''.join(tmp[:-1]) # except the version
        match model_name:
            case 'transformer':
                class_name = 'Transformer'
            case 'transformerbilstm':
                class_name = 'TransformerBiLSTM'
            case 'unet':
                class_name = 'UNet'
            case _:
                raise ValueError(f"Unknown model type in experiment name: {exp_name}")

        # Dynamically import the module and class
        module = importlib.import_module(f'models.{exp_name}')
        model_class = getattr(module, class_name)
        
        # Instantiate the model class
        match model_name:
            case 'transformer':
                if exp_name == 'transformer_v3':
                        
                    self.model = model_class(
                                    input_dim = len(opt.feats) * 3,
                                    feat_dim  = opt.fog_model_feat_dim,
                                    nheads    = opt.fog_model_nheads,
                                    nlayers   = opt.fog_model_nlayers,
                                    dropout   = opt.fog_model_encoder_dropout,
                                    clip_dim  = opt.clip_dim,
                                    feats     = opt.feats,
                                    txt_cond  = opt.txt_cond,
                                    clip_version = opt.clip_version,
                                    activation = opt.activation
                                )
                else:
                    self.model = model_class(
                                    input_dim = opt.fog_model_input_dim,
                                    feat_dim  = opt.fog_model_feat_dim,
                                    nheads    = opt.fog_model_nheads,
                                    nlayers   = opt.fog_model_nlayers,
                                    dropout   = opt.fog_model_encoder_dropout,
                                )
            case 'transformerbilstm':
                self.model = model_class(opt)
            case 'unet':
                if exp_name == "unet_v4":
                    self.model = model_class(
                                    channel  = len(opt.feats) * 3,
                                    feats    = opt.feats
                                )
                else:
                    self.model = model_class()
        
        assert self.model is not None, "Error when loading model"
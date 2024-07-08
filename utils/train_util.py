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
        print(f"Val n trials:    {opt.val_n_trials}")
        print(f"Test n trials:   {opt.test_n_trials}")
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
    with open(os.path.join(opt.save_dir, 'opt.yaml'), 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=False)   

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
        if 'transformer' in tmp:
            class_name = 'Transformer'
        elif 'unet' in tmp:
            class_name = 'UNet'
        else:
            raise ValueError(f"Unknown model type in experiment name: {exp_name}")

        # Dynamically import the module and class
        module = importlib.import_module(f'models.{exp_name}')
        model_class = getattr(module, class_name)
        
        # Instantiate the model class
        if 'transformer' in tmp:
            self.model = model_class(opt)
        elif 'unet' in tmp:
            self.model = model_class()
        
        assert self.model is not None, "Error when loading model"
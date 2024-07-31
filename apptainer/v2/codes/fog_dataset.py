import argparse, joblib, math, os, random, sys

import torch
import numpy as np
from torch.utils.data import Dataset


DATASETS_FEATS_MODEL = {
    'kaggle': ['lowerback_acc'],
    'turn': ['l_latshank_acc', 'r_latshank_acc', 'l_latshank_gyr', 'r_latshank_gyr'],
    'daphnet': ['lowerback_acc', 'l_midlatthigh_acc', 'l_ankle_acc']
}



class FoGDataset(Dataset):
    '''
    Freezing of Gaits dataset for parkinson disease
    '''
    
    def __init__(self, opt, model_name):
        """Initialize FoGDataset.

        Args:
            opt.root_dpath: 'data/rectified_data'
        """

        self.window = opt.window
        self.device = opt.device
        self.preload_gpu = opt.preload_gpu
        self.opt = opt
        self.model_name = model_name
        
        self.window_data_dict = {}
        
        # self.idx_feats = {} # 0: lowerback_acc
        # self.feats_data = {} # lowerback_acc: (N,3)
        # for idx in range(len(opt.feats)):
        #     self.idx_feats[idx] = opt.feats[idx]
        #     self.feats_data[opt.feats[idx]] = None
            
          
        dname = f"all_test_data_{model_name}_window{self.window}.p"  
        
        mode_dpath = os.path.join(opt.root_dpath, dname)
        
        self._load_train_val_test_data(mode_dpath) 
    
                    
    def _load_train_val_test_data(self, dpath):
        try:
            d_dict = joblib.load(dpath)
        except Exception as e:
            print(f"\n**** {dpath} cannot be loaded.")
            print(e)
            return
        
        count_key = len(self.window_data_dict.keys())
        for _, value in d_dict.items():
            
            self.window_data_dict[count_key] = value
                        
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! watch out cuda memory
            for body_name in DATASETS_FEATS_MODEL[self.model_name]:
                model_input = value[body_name]
                if self.preload_gpu:
                    self.window_data_dict[count_key][body_name] = model_input.to(self.device)
                else:
                    self.window_data_dict[count_key][body_name] = model_input
                             
            count_key += 1  
        
        print(f"Finishing loading {dpath}")

    def __len__(self):
        return len(self.window_data_dict)

    def __getitem__(self, idx):
        return self.window_data_dict[idx]
    
    


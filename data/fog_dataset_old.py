import argparse, joblib, math, os, random, sys

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.config import ALL_DATASETS, VALIDATION_SET, FEATURES_LIST
from utils.train_util import rotateC

class FoGDataset(Dataset):
    '''
    Freezing of Gaits dataset for parkinson disease
    '''
    
    def __init__(self, opt, mode='train'):
        """Initialize FoGDataset.

        Args:
            opt.root_dpath: 'data/rectified_data'
        """

        print(f"Initialize FoG dataset: {mode} mode --------")
        
        self.mode = mode
        self.feats = opt.feats
        self.window = opt.window
        self.random_aug = opt.random_aug
        self.device = opt.device
        self.preload_gpu = opt.preload_gpu
        self.lab_home = opt.lab_home
        self.opt = opt
        
        self.window_data_dict = {}
        
        for data_folder in tqdm(os.listdir(opt.root_dpath), total=len(os.listdir(opt.root_dpath)), 
                                desc=f"Load {self.mode} data", file=sys.stdout):
            if data_folder not in ALL_DATASETS or data_folder not in opt.train_datasets:
                continue
            
            full_or_window = f"window{self.window}" if self.window != -1 else "full"
    
            if self.random_aug:
                dname = f"{self.mode}_{data_folder}_{full_or_window}_{self.lab_home}_randomaug.p"  
            else:
                dname = f"{self.mode}_{data_folder}_{full_or_window}_{self.lab_home}_allaug.p"                
            
            mode_dpath = os.path.join(opt.root_dpath, data_folder, dname)
            
            if not os.path.exists(mode_dpath):
                if self.window != -1:
                    self._generate_train_val_data_window_lab_home(opt.root_dpath, data_folder)
                else:
                    self._generate_train_val_data_full_lab_home(opt.root_dpath, data_folder)

            self._load_train_val_data(mode_dpath) 
            
            if not opt.disable_wandb:
                opt.wandb.config.update({f'{self.mode}_data': dname})

            setattr(self.opt, f"{self.mode}_dname", dname)
                        
        print(f"{self.mode}: total number of window {len(self.window_data_dict)}\n")
                    
    def _load_train_val_data(self, dpath):
        try:
            d_dict = joblib.load(dpath)
        except Exception:
            print(f"\n**** {dpath} cannot be loaded.")
            return
        
        count_key = len(self.window_data_dict.keys())
        for _, value in d_dict.items():
            self.window_data_dict[count_key] = value
            
            model_input = value['model_input'] # (window, num_feats)
            
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! watch out cuda memory
            if self.preload_gpu:
                self.window_data_dict[count_key]['model_input'] = model_input.to(self.device)
            else:
                self.window_data_dict[count_key]['model_input'] = model_input
                                                                    
            count_key += 1  
        
        mode_set = set()
        for key, value in self.window_data_dict.items():
            mode_set.add(value['ori_series_name'])
        
        setattr(self.opt, f'{self.mode}_n_trials', len(mode_set))

        print(f"Finishing loading {dpath}")

    def _generate_train_val_data_window_lab_home(self, root_dpath, dataset_name):
        """Generate and store train or val data for one specific dataset.
        
        Args:
            root_dpath: e.g. 'data/rectified_data'
            dataset_name: e.g. 'dataset_fog_release' or 'kaggle_pd_data'
        """
        dpath = os.path.join(root_dpath, dataset_name, f"all_{dataset_name}.p")
        try:
            single_data_dict = joblib.load(dpath)
        except Exception:
            print(f"\n**** {dpath} cannot be loaded.")
            return
        
        print("_generate_train_val_data_window_lab_home")
        
        # find max seq_len
        total_seq_len = 0
        total_num_series = 0
        for key, value in single_data_dict.items():
            if self.lab_home == "lab":
                lab_or_home = value['ori_filename'].split("_")[-1]
                if lab_or_home != "tdcsfog":
                    continue
            total_num_series += 1
            total_seq_len += value['gt'].shape[0]

        count = 0
        all_data_dict = {}
        validation_set = set() # series_name: total 970, pick 100
        validation_trials = set()
        test_set = set()
        test_trials = set()
        for series_name, series_info in tqdm(single_data_dict.items(), 
                                             total=len(single_data_dict),
                                             desc=f"Gen train/val data: {dataset_name}", 
                                             file=sys.stdout):
            
            if self.lab_home == "lab":
                lab_or_home = series_info['ori_filename'].split("_")[-1]
                if lab_or_home != "tdcsfog":
                    continue
            
            series_len = len(series_info['gt'])

            # (series_len, num_feats)  e.g. (16641, 9)
            concate_feat = []
            
            for feat in self.feats:
                if feat == 'Annotation':
                    continue
                concate_feat.append(series_info[feat][:, None])
            
            concate_feat = torch.cat(concate_feat, dim=1) # (series_len, num_feats)
            
            # series_info['gt'] = series_info['gt'].to(self.device)
            
            padding_len = math.ceil(series_len / self.window) * self.window - series_len
            pad_feat = torch.zeros(padding_len, concate_feat.shape[1], device=concate_feat.device)
            pad_gt = torch.ones(padding_len, dtype=torch.int8, 
                                    device=series_info['gt'].device) * 2

            # (T', num_feats)  e.g. (31104, 9)
            concate_feat = torch.cat([concate_feat, pad_feat], dim=0)
            
            # (T',)
            concate_gt = torch.cat([series_info['gt'], pad_gt], dim=0)
            
            this_is_valid = False
            this_is_test = False
            
            # Consider whether this series is included in validation
            if random.random() < 0.6 and len(validation_set) <= 0.1 * total_num_series:
                # check if this window includes fog event
                has_one_in_second_column = (series_info['gt'][:] == 1).any().item()
                if has_one_in_second_column:
                    this_is_valid = True
                    validation_trials.add(series_info['ori_filename'])
                    
            # Consider whether this series is included in test set
            if random.random() < 0.6 and len(test_set) <= 0.1 * total_num_series:
                # check if this window includes fog event
                has_one_in_second_column = (series_info['gt'][:] == 1).any().item()
                if has_one_in_second_column and not this_is_valid:
                    this_is_test = True
                    test_trials.add(series_info['ori_filename'])
                    
            if this_is_valid or this_is_test:
                assert this_is_valid != this_is_test, "both valid and test"
        
            # split by window ==================================================
            for i in range(0, concate_feat.shape[0], self.window):
                start_t_idx = int(i)
                end_t_idx = int(i + self.window)
                
                assert end_t_idx <= concate_feat.shape[0], "length unmatched"
            
                # (window, num_feats)
                model_input = concate_feat[start_t_idx:end_t_idx].detach().clone()
                model_input = model_input.to(dtype=torch.float32)

                # (window,)
                gt = concate_gt[start_t_idx:end_t_idx].detach().clone() 
                
                gt = torch.nn.functional.one_hot(gt.to(torch.int64), num_classes=3)
                
                # check if it is valid one hot
                valid_one_hot = ((gt.sum(dim=1) == 1) & ((gt == 0) | (gt == 1)).all(dim=1)) \
                                                                               .all().item()
                assert valid_one_hot, "not valid one hot encoding"
                
                # (window,3)
                gt = gt.float()
                # (window,3)
                inference_gt = gt.detach().clone()
                
                added_window = {
                    'series_name': series_name,  #                              str
                    'ori_series_name': series_info['ori_filename'], #           str
                    'start_t_idx': start_t_idx, # 0,     15552, ...               int
                    'end_t_idx': end_t_idx,     # 15552, 15552+15552, ...         int
                    'model_input': model_input.cpu(), # (window, num_feats)  torch
                    'gt': gt.cpu(), # (window, 3) one-hot                      torch
                    'inference_gt': inference_gt.cpu(),                      # (window, 3) one-hot
                }
                
                if this_is_valid:
                    validation_set.add(count)
                if this_is_test:
                    test_set.add(count)
                
                all_data_dict[count] = added_window
                count += 1
                
        # Initialize counters and sequence lengths                              
        train_count, val_count, test_count = 0, 0, 0                            
        train_seq_len, val_seq_len, test_seq_len = 0, 0, 0                      
        # Initialize dictionaries for training and validation data              
        train_data_dict = {}                                                    
        val_data_dict = {}                                                      
        test_data_dict = {}                                                     
                                                                                
        for key, value in all_data_dict.items():                      
                                                                                
            if key in validation_set:
                if value['ori_series_name'] not in validation_trials:
                    raise "unmatch set and trials"                                       
                val_data_dict[val_count] = value                                
                val_seq_len += value['model_input'].shape[0]                    
                val_count += 1                                                  
            elif key in test_set:  
                if value['ori_series_name'] not in test_trials:
                    raise "unmatch set and trials"                                              
                test_data_dict[test_count] = value                              
                test_seq_len += value['model_input'].shape[0]                   
                test_count += 1                                                 
            else:  # training set                                               
                #* data augmentation ***********************************************
                model_input = value['model_input'] # (window, num_feats) (15552, 3)
                ori_dtype = model_input.dtype                                   
                if self.random_aug:                                             
                    if random.random() <= 0.5:                                  
                        # Augment data                                          
                        model_input = self._augment_data(model_input.cpu().detach().numpy())
                else: # augment all data                                        
                    model_input = self._augment_data(model_input.cpu().detach().numpy())
                                                                                
                # (window, num_feats)                                           
                model_input = model_input.to(ori_dtype)                         
                                                                                
                assert model_input.shape == (self.window, len(self.feats)), "incorrect shape"
                                                                                
                value['model_input'] = model_input                              
                #*******************************************************************
                                                                                
                train_data_dict[train_count] = value                            
                train_seq_len += value['model_input'].shape[0]                  
                train_count += 1 

        print(f"\n+++++++++++++++++: ")
        print("total number of trials: ", total_num_series)       
        print()
                                            
        print(f'test_set # trials {len(test_trials)}')      
        print(f'test_set # window {len(test_data_dict.keys())}')  
        print(test_trials)
        print()                    
                                               
        print(f'validation_set # trials {len(validation_trials)}')    
        print(f'validation_set # window {len(val_data_dict.keys())}') 
        print(validation_trials)
        print()           

        print(f'train_set # trials {total_num_series-len(test_trials)-len(validation_trials)}')
        print(f'train_set # window {len(train_data_dict.keys())}')
        print()                                                                 
                                                                                
        print(f"----- train seq len: {train_seq_len} -----")                    
        print(f"----- val seq len: {val_seq_len} -----")                        
        print(f"----- test seq len: {test_seq_len} -----")                      
        print()                                                                 
                                                                                
        train_dname = f"train_{dataset_name}_window{self.window}_{self.lab_home}_allaug.p"     
        val_dname = f"val_{dataset_name}_window{self.window}_{self.lab_home}_allaug.p"         
        test_dname = f"test_{dataset_name}_window{self.window}_{self.lab_home}_allaug.p"       
                                                                                
        if self.random_aug:                                                     
            train_dname = f"train_{dataset_name}_window{self.window}_{self.lab_home}_randomaug.p"
            val_dname = f"val_{dataset_name}_window{self.window}_{self.lab_home}_randomaug.p"  
            test_dname = f"test_{dataset_name}_window{self.window}_{self.lab_home}_randomaug.p"
                                                                                
        train_dpath = os.path.join(root_dpath, dataset_name, train_dname)       
        val_dpath = os.path.join(root_dpath, dataset_name, val_dname)           
        test_dpath = os.path.join(root_dpath, dataset_name, test_dname)         
                                                                                
        joblib.dump(train_data_dict, open(train_dpath, 'wb'))                   
        joblib.dump(val_data_dict, open(val_dpath, 'wb'))                       
        joblib.dump(test_data_dict, open(test_dpath, 'wb'))                     
                                                                                
        print(f"Finishing spliting train, val, test data for {dataset_name}\n")                

    def _generate_train_val_data_full_lab_home(self, root_dpath, dataset_name):
        """
        Args:
            root_dpath: e.g. 'data/rectified_data'
            dataset_name: e.g. 'dataset_fog_release' or 'kaggle_pd_data'
        """
        dpath = os.path.join(root_dpath, dataset_name, f"all_{dataset_name}.p")
        try:
            single_data_dict = joblib.load(dpath)
        except Exception:
            print(f"\n**** {dpath} cannot be loaded.")
            return
        
        print(f"_generate_train_val_data_full_lab_home")
        
        # find max seq_len
        total_seq_len = 0
        max_seq_len = 0
        total_num_series = 0
        for key, value in single_data_dict.items():
            if self.lab_home == "lab":
                lab_or_home = value['ori_filename'].split("_")[-1]
                if lab_or_home != "tdcsfog":
                    continue
            total_num_series += 1
            total_seq_len += value['gt'].shape[0]
            if value['gt'].shape[0] > max_seq_len:
                max_seq_len = value['gt'].shape[0]
        ori_max_seq_len = max_seq_len
        while max_seq_len % 32 != 0: # cuz maxpool in unet
            max_seq_len += 1

        print("------------------")
        print(f"orig max seq len: {ori_max_seq_len}, revised max seq len: {max_seq_len}")
        
        count = 0
        all_data_dict = {}
        validation_set = set() # series_name: total 970, pick 100
        test_set = set()
        for series_name, series_info in tqdm(single_data_dict.items(), 
                                             total=len(single_data_dict),
                                             desc=f"Gen train/val data: {dataset_name}", 
                                             file=sys.stdout):
            
            if self.lab_home == "lab":
                lab_or_home = series_info['ori_filename'].split("_")[-1]
                if lab_or_home != "tdcsfog":
                    continue
            
            series_len = len(series_info['gt'])

            # (series_len, num_feats)  e.g. (16641, 9)
            concate_feat = []
            
            for feat in self.feats:
                if feat == 'Annotation':
                    continue
                concate_feat.append(series_info[feat][:, None])
            
            concate_feat = torch.cat(concate_feat, dim=1) # (series_len, num_feats)
            
            # series_info['gt'] = series_info['gt'].to(self.device)
            
            padding_len = max_seq_len - series_len
            pad_feat = torch.zeros(padding_len, concate_feat.shape[1], device=concate_feat.device)
            pad_gt = torch.zeros(padding_len, dtype=torch.int8, 
                                    device=series_info['gt'].device) * 2

            # (T', num_feats)  e.g. (31104, 9)
            concate_feat = torch.cat([concate_feat, pad_feat], dim=0)
            
            # (T',)
            concate_gt = torch.cat([series_info['gt'], pad_gt], dim=0)
            
            
            # (window, num_feats)
            model_input = concate_feat.detach().clone()
            model_input = model_input.to(dtype=torch.float32)

            # (window,)
            gt = concate_gt.detach().clone() 
            
            gt = torch.nn.functional.one_hot(gt.to(torch.int64), num_classes=3)
                
            # check if it is valid one hot
            valid_one_hot = ((gt.sum(dim=1) == 1) & ((gt == 0) | (gt == 1)).all(dim=1)) \
                                                                            .all().item()
            assert valid_one_hot, "not valid one hot encoding"
                
            # Consider whether this series is included in validation
            if random.random() < 0.3 and len(validation_set) < 0.1 * total_num_series:
                # check if this window includes fog event
                has_one_in_second_column = (gt[:, 1] == 1).any().item()
                if has_one_in_second_column:
                    validation_set.add(count)
                    
             # Consider whether this series is included in test set
            if random.random() < 0.3 and len(test_set) < 0.03 * total_num_series:
                # check if this window includes fog event
                has_one_in_second_column = (gt[:, 1] == 1).any().item()
                if has_one_in_second_column and count not in validation_set:
                    test_set.add(count)

            # (window,3)
            gt = gt.float()
            # (window,3)
            inference_gt = gt.detach().clone()
            
            added_window = {
                'series_name': series_name,  #                              str
                'ori_series_name': series_info['ori_filename'], #           str
                'model_input': model_input.cpu(), # (window, num_feats)  torch
                'gt': gt.cpu(), # (window, 3) one-hot                      torch
                'inference_gt': inference_gt.cpu(),                      # (window, 3) one-hot
            }
            all_data_dict[count] = added_window
            count += 1
        
        # Initialize counters and sequence lengths
        train_count, val_count, test_count = 0, 0, 0
        train_seq_len, val_seq_len, test_seq_len = 0, 0, 0
        # Initialize dictionaries for training and validation data
        train_data_dict = {}
        val_data_dict = {}
        test_data_dict = {}

        for key, value in tqdm(all_data_dict.items(),
                               total=len(all_data_dict.keys()), 
                               desc="split train val"):
            
            if key in validation_set:
                val_data_dict[val_count] = value
                val_seq_len += value['model_input'].shape[0]
                val_count += 1
            elif key in test_set:
                test_data_dict[test_count] = value
                test_seq_len += value['model_input'].shape[0]
                test_count += 1
            else:  # training set
                #* data augmentation ***********************************************
                model_input = value['model_input'] # (window, num_feats) (15552, 3)
                ori_dtype = model_input.dtype
                if self.random_aug:
                    if random.random() <= 0.5:
                        # Augment data
                        model_input = self._augment_data(model_input.cpu().detach().numpy())
                else: # augment all data
                    model_input = self._augment_data(model_input.cpu().detach().numpy())
                    
                # (window, num_feats)
                model_input = model_input.to(ori_dtype) 
                
                assert model_input.shape == (max_seq_len, len(self.feats)), "incorrect shape"

                value['model_input'] = model_input
                #*******************************************************************
                
                train_data_dict[train_count] = value
                train_seq_len += value['model_input'].shape[0]
                train_count += 1
                
        print(f"+++++++++++++++++: ")
        print(f'test_set # example {len(test_set)}:')
        # print(test_set)
        print(f'validation_set # example {len(validation_set)}:')
        # print(validation_set)
        print(f'train set # example {len(all_data_dict.keys())-len(validation_set)-len(test_set)}')
        print()
        
        print(f"----- train seq len: {train_seq_len} -----")
        print(f"----- val seq len: {val_seq_len} -----")
        print(f"----- test seq len: {test_seq_len} -----")
        print()
        
        train_dname = f"train_{dataset_name}_full_{self.lab_home}_allaug.p"
        val_dname = f"val_{dataset_name}_full_{self.lab_home}_allaug.p"
        test_dname = f"test_{dataset_name}_full_{self.lab_home}_allaug.p"
        
        if self.random_aug:
            train_dname = f"train_{dataset_name}_full_{self.lab_home}_randomaug.p"
            val_dname = f"val_{dataset_name}_full_{self.lab_home}_randomaug.p"
            test_dname = f"test_{dataset_name}_full_{self.lab_home}_randomaug.p"
     
        train_dpath = os.path.join(root_dpath, dataset_name, train_dname)
        val_dpath = os.path.join(root_dpath, dataset_name, val_dname)
        test_dpath = os.path.join(root_dpath, dataset_name, test_dname)
        
        joblib.dump(train_data_dict, open(train_dpath, 'wb'))
        joblib.dump(val_data_dict, open(val_dpath, 'wb'))
        joblib.dump(test_data_dict, open(test_dpath, 'wb'))
        
        print(f"Finishing spliting train, val, test data for {dataset_name}")
   
    def _generate_train_val_data_window(self, root_dpath, dataset_name):
        """Generate and store train or val data for one specific dataset.
        
        Args:
            root_dpath: e.g. 'data/rectified_data'
            dataset_name: e.g. 'dataset_fog_release' or 'kaggle_pd_data'
        """
        dpath = os.path.join(root_dpath, dataset_name, f"all_{dataset_name}.p")
        try:
            single_data_dict = joblib.load(dpath)
        except Exception:
            print(f"\n**** {dpath} cannot be loaded.")
            return
        
        print("_generate_train_val_data_window")
        
        # find max seq_len
        total_seq_len = 0
        for key, value in single_data_dict.items():
            total_seq_len += value['gt'].shape[0]
        
        
        count = 0
        all_data_dict = {}
        validation_set = set() # series_name: total 970, pick 100
        for series_name, series_info in tqdm(single_data_dict.items(), 
                                             total=len(single_data_dict),
                                             desc=f"Gen train/val data: {dataset_name}", 
                                             file=sys.stdout):
            series_len = len(series_info['gt'])

            # (series_len, num_feats)  e.g. (16641, 9)
            concate_feat = []
            
            for feat in self.feats:
                if feat == 'Annotation':
                    continue
                concate_feat.append(series_info[feat][:, None])
            
            concate_feat = torch.cat(concate_feat, dim=1) # (series_len, num_feats)
            
            # series_info['gt'] = series_info['gt'].to(self.device)
            
            padding_len = math.ceil(series_len / self.window) * self.window - series_len
            pad_feat = torch.zeros(padding_len, concate_feat.shape[1], device=concate_feat.device)
            pad_gt = torch.zeros(padding_len, dtype=torch.int8, 
                                    device=series_info['gt'].device) * 2

            # (T', num_feats)  e.g. (31104, 9)
            concate_feat = torch.cat([concate_feat, pad_feat], dim=0)
            
            # (T',)
            concate_gt = torch.cat([series_info['gt'], pad_gt], dim=0)
            
        
            for i in range(0, concate_feat.shape[0], self.window):
                start_t_idx = int(i)
                end_t_idx = int(i + self.window)
                
                assert end_t_idx <= concate_feat.shape[0], "length unmatched"
            
                # (window, num_feats)
                model_input = concate_feat[start_t_idx:end_t_idx].detach().clone()
                model_input = model_input.to(dtype=torch.float32)

                # (window,)
                gt = concate_gt[start_t_idx:end_t_idx].detach().clone() 
                
                gt = torch.nn.functional.one_hot(gt.to(torch.int64), num_classes=3)
                
                # check if it is valid one hot
                valid_one_hot = ((gt.sum(dim=1) == 1) & ((gt == 0) | (gt == 1)).all(dim=1)) \
                                                                               .all().item()
                assert valid_one_hot, "not valid one hot encoding"
                
                # Consider whether this series is included in validation
                if random.random() < 0.3 and \
                   len(validation_set) < 0.11*(total_seq_len//self.window):
                    # check if this window includes fog event
                    has_one_in_second_column = (gt[:, 1] == 1).any().item()
                    if has_one_in_second_column:
                        validation_set.add(count)

                # (window,3)
                gt = gt.float()
                # (window,3)
                inference_gt = gt.detach().clone()
                
                added_window = {
                    'series_name': series_name,  #                              str
                    'ori_series_name': series_info['ori_filename'], #           str
                    'start_t_idx': start_t_idx, # 0,     15552, ...               int
                    'end_t_idx': end_t_idx,     # 15552, 15552+15552, ...         int
                    'model_input': model_input.cpu(), # (window, num_feats)  torch
                    'gt': gt.cpu(), # (window, 3) one-hot                      torch
                    'inference_gt': inference_gt.cpu(),                      # (window, 3) one-hot
                }
                
                all_data_dict[count] = added_window
                count += 1
        
        # Initialize counters and sequence lengths
        train_count, val_count = 0, 0
        train_seq_len, val_seq_len = 0, 0
        # Initialize dictionaries for training and validation data
        train_data_dict = {}
        val_data_dict = {}

        for key, value in tqdm(all_data_dict.items(),
                               total=len(all_data_dict.keys()), 
                               desc="split train val"):
            
            if key in validation_set:
                val_data_dict[val_count] = value
                val_seq_len += value['model_input'].shape[0]
                val_count += 1
            else:  # training set
                #* data augmentation ***********************************************
                model_input = value['model_input'] # (window, num_feats) (15552, 3)
                ori_dtype = model_input.dtype
                if self.random_aug:
                    if random.random() <= 0.5:
                        # Augment data
                        model_input = self._augment_data(model_input.cpu().detach().numpy())
                else: # augment all data
                    model_input = self._augment_data(model_input.cpu().detach().numpy())
                    
                # (window, num_feats)
                model_input = model_input.to(ori_dtype) 
                
                assert model_input.shape == (int(self.window), len(self.feats)), "incorrect shape"

                value['model_input'] = model_input
                #*******************************************************************
                
                train_data_dict[train_count] = value
                train_seq_len += value['model_input'].shape[0]
                train_count += 1
                
        print(f"+++++++++++++++++: ")
        print(f'validation_set num example {len(validation_set)}:')
        print(validation_set)
        print(f'train set num example {len(all_data_dict.keys()) - len(validation_set)}')
        print()
        
        print(f"----- train seq len: {train_seq_len} -----")
        print(f"----- val seq len: {val_seq_len} -----")
        print()
        
        train_dname = f"train_{dataset_name}_window{self.window}_allaug.p"
        val_dname = f"val_{dataset_name}_window{self.window}_allaug.p"
        
        if self.random_aug:
            train_dname = f"train_{dataset_name}_window{self.window}_randomaug.p"
            val_dname = f"val_{dataset_name}_window{self.window}_randomaug.p"
     
        train_dpath = os.path.join(root_dpath, dataset_name, train_dname)
        val_dpath = os.path.join(root_dpath, dataset_name, val_dname)
        
        joblib.dump(train_data_dict, open(train_dpath, 'wb'))
        joblib.dump(val_data_dict, open(val_dpath, 'wb'))
        
        print(f"Finishing spliting train and val data for {dataset_name}")

    def _generate_train_val_data_full(self, root_dpath, dataset_name):
        """Generate and store train or val data for one specific dataset.
        
        Args:
            root_dpath: e.g. 'data/rectified_data'
            dataset_name: e.g. 'dataset_fog_release' or 'kaggle_pd_data'
        """
        dpath = os.path.join(root_dpath, dataset_name, f"all_{dataset_name}.p")
        try:
            single_data_dict = joblib.load(dpath)
        except Exception:
            print(f"\n**** {dpath} cannot be loaded.")
            return
        
        print("_generate_train_val_data_full")
        
        # find max seq_len
        total_seq_len = 0
        max_seq_len = 0
        for key, value in single_data_dict.items():
            total_seq_len += value['gt'].shape[0]
            if value['gt'].shape[0] > max_seq_len:
                max_seq_len = value['gt'].shape[0]
        ori_max_seq_len = max_seq_len
        while max_seq_len % 32 != 0: # cuz maxpool in unet
            max_seq_len += 1
        print("------------------")
        print(f"orig max seq len: {ori_max_seq_len}, revised max seq len: {max_seq_len}")
        
        count = 0
        all_data_dict = {}
        validation_set = set() # series_name: total 970, pick 100
        test_set = set()
        for series_name, series_info in tqdm(single_data_dict.items(), 
                                             total=len(single_data_dict),
                                             desc=f"Gen train/val data: {dataset_name}", 
                                             file=sys.stdout):
            series_len = len(series_info['gt'])

            # (series_len, num_feats)  e.g. (16641, 9)
            concate_feat = []
            
            for feat in self.feats:
                if feat == 'Annotation':
                    continue
                concate_feat.append(series_info[feat][:, None])
            
            concate_feat = torch.cat(concate_feat, dim=1) # (series_len, num_feats)
            
            # series_info['gt'] = series_info['gt'].to(self.device)
            
            padding_len = max_seq_len - series_len
            pad_feat = torch.zeros(padding_len, concate_feat.shape[1], device=concate_feat.device)
            pad_gt = torch.zeros(padding_len, dtype=torch.int8, 
                                    device=series_info['gt'].device) * 2

            # (T', num_feats)  e.g. (31104, 9)
            concate_feat = torch.cat([concate_feat, pad_feat], dim=0)
            
            # (T',)
            concate_gt = torch.cat([series_info['gt'], pad_gt], dim=0)
            
            
            # (window, num_feats)
            model_input = concate_feat.detach().clone()
            model_input = model_input.to(dtype=torch.float32)

            # (window,)
            gt = concate_gt.detach().clone() 
            
            gt = torch.nn.functional.one_hot(gt.to(torch.int64), num_classes=3)
                
            # check if it is valid one hot
            valid_one_hot = ((gt.sum(dim=1) == 1) & ((gt == 0) | (gt == 1)).all(dim=1)) \
                                                                            .all().item()
            assert valid_one_hot, "not valid one hot encoding"
                
            # Consider whether this series is included in validation
            if random.random() < 0.3 and \
                len(validation_set) < 0.1*len(single_data_dict.keys()):
                # check if this window includes fog event
                has_one_in_second_column = (gt[:, 1] == 1).any().item()
                if has_one_in_second_column:
                    validation_set.add(count)
                    
             # Consider whether this series is included in test set
            if random.random() < 0.3 and \
                len(test_set) < 0.03*len(single_data_dict.keys()):
                # check if this window includes fog event
                has_one_in_second_column = (gt[:, 1] == 1).any().item()
                if has_one_in_second_column and count not in validation_set:
                    test_set.add(count)

            # (window,3)
            gt = gt.float()
            # (window,3)
            inference_gt = gt.detach().clone()
            
            added_window = {
                'series_name': series_name,  #                              str
                'ori_series_name': series_info['ori_filename'], #           str
                'model_input': model_input.cpu(), # (window, num_feats)  torch
                'gt': gt.cpu(), # (window, 3) one-hot                      torch
                'inference_gt': inference_gt.cpu(),                      # (window, 3) one-hot
            }
            all_data_dict[count] = added_window
            count += 1
        
        # Initialize counters and sequence lengths
        train_count, val_count, test_count = 0, 0, 0
        train_seq_len, val_seq_len, test_seq_len = 0, 0, 0
        # Initialize dictionaries for training and validation data
        train_data_dict = {}
        val_data_dict = {}
        test_data_dict = {}

        for key, value in tqdm(all_data_dict.items(),
                               total=len(all_data_dict.keys()), 
                               desc="split train val"):
            
            if key in validation_set:
                val_data_dict[val_count] = value
                val_seq_len += value['model_input'].shape[0]
                val_count += 1
            elif key in test_set:
                test_data_dict[test_count] = value
                test_seq_len += value['model_input'].shape[0]
                test_count += 1
            else:  # training set
                #* data augmentation ***********************************************
                model_input = value['model_input'] # (window, num_feats) (15552, 3)
                ori_dtype = model_input.dtype
                if self.random_aug:
                    if random.random() <= 0.5:
                        # Augment data
                        model_input = self._augment_data(model_input.cpu().detach().numpy())
                else: # augment all data
                    model_input = self._augment_data(model_input.cpu().detach().numpy())
                    
                # (window, num_feats)
                model_input = model_input.to(ori_dtype) 
                
                assert model_input.shape == (max_seq_len, len(self.feats)), "incorrect shape"

                value['model_input'] = model_input
                #*******************************************************************
                
                train_data_dict[train_count] = value
                train_seq_len += value['model_input'].shape[0]
                train_count += 1
                
        print(f"+++++++++++++++++: ")
        print(f'test_set # example {len(test_set)}:')
        print(test_set)
        print(f'validation_set # example {len(validation_set)}:')
        print(validation_set)
        print(f'train set # example {len(all_data_dict.keys())-len(validation_set)-len(test_set)}')
        print()
        
        print(f"----- train seq len: {train_seq_len} -----")
        print(f"----- val seq len: {val_seq_len} -----")
        print(f"----- test seq len: {test_seq_len} -----")
        print()
        
        train_dname = f"train_{dataset_name}_full_allaug.p"
        val_dname = f"val_{dataset_name}_full_allaug.p"
        test_dname = f"test_{dataset_name}_full_allaug.p"
        
        if self.random_aug:
            train_dname = f"train_{dataset_name}_full_randomaug.p"
            val_dname = f"val_{dataset_name}_full_randomaug.p"
            test_dname = f"test_{dataset_name}_full_randomaug.p"
     
        train_dpath = os.path.join(root_dpath, dataset_name, train_dname)
        val_dpath = os.path.join(root_dpath, dataset_name, val_dname)
        test_dpath = os.path.join(root_dpath, dataset_name, test_dname)
        
        joblib.dump(train_data_dict, open(train_dpath, 'wb'))
        joblib.dump(val_data_dict, open(val_dpath, 'wb'))
        joblib.dump(test_data_dict, open(test_dpath, 'wb'))
        
        print(f"Finishing spliting train, val, test data for {dataset_name}")

    def _augment_data(self, model_input):
        # model_input = model_input.reshape(self.block_size//self.patch_size, 
        #                                             self.patch_size, 
        #                                             self.num_feats)
        # model_input = model_input.reshape(self.block_size//self.patch_size, 
        #                                 self.patch_size,
        #                                 3,3) # (864,18,3,3)
        
        # model_input: (max_seq_len, num_feats)
        theta = random.random()*math.pi*2
        theta = random.random()*360
        a=random.random()
        b=random.random()
        c=random.random()
        model_input = rotateC(model_input, theta, a, b, c)
        return torch.tensor(model_input).clone().detach()

    def __len__(self):
        return len(self.window_data_dict)

    def __getitem__(self, idx):
        return self.window_data_dict[idx]
        # return self.window_data_dict[idx]['model_input']
    
    


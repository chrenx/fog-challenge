import argparse, joblib, math, os, random, sys

import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.config import ALL_DATASETS, VALIDATION_SET
from utils.train_util import rotateC, is_valid_one_hot

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
        self.window = opt.window
        self.random_aug = opt.random_aug
        self.device = opt.device
        self.preload_gpu = opt.preload_gpu
        self.lab_home = opt.lab_home
        self.opt = opt
        
        self.window_data_dict = {}
        
        self.idx_feats = {} # 0: lowerback_acc
        self.feats_data = {} # lowerback_acc: (N,3)
        for idx in range(len(opt.feats)):
            self.idx_feats[idx] = opt.feats[idx]
            self.feats_data[opt.feats[idx]] = None
            
        # self.all_dname = opt.train_datasets[0]
        # for i in range(1,len(opt.train_datasets)):
        #     if opt.train_datasets[i] not in ALL_DATASETS:
        #         raise Exception(f"{opt.train_datasets[i]} is not recognized.")
        #     self.all_dname += f"_{opt.train_datasets[i]}"

        self.all_dname = opt.train_datasets
        if opt.train_datasets not in ALL_DATASETS:
            raise Exception(f"{opt.train_datasets} is not recognized.")
          
        full_or_window = f"window{self.window}" if self.window != -1 else "full"

        if self.random_aug:
            dname = f"{self.mode}_{self.all_dname}_{full_or_window}_randomaug.p"  
        else:
            dname = f"{self.mode}_{self.all_dname}_{full_or_window}_allaug.p"                
        
        mode_dpath = os.path.join(opt.root_dpath, 'all_data', dname)
        
        if not os.path.exists(mode_dpath):
            self._generate_train_val_test(opt.root_dpath, [opt.train_datasets])
  
        self._load_train_val_test_data(mode_dpath) 
        
        if not opt.disable_wandb:
            opt.wandb.config.update({f'{self.mode}_data': dname})

        setattr(self.opt, f"{self.mode}_dname", dname)
                        
        print(f"{self.mode}: total number of window {len(self.window_data_dict)}\n")
                    
    def _load_train_val_test_data(self, dpath):
        try:
            d_dict = joblib.load(dpath)
        except Exception:
            print(f"\n**** {dpath} cannot be loaded.")
            return
        
        count_key = len(self.window_data_dict.keys())
        for _, value in d_dict.items():
            self.window_data_dict[count_key] = value
            
            events = self.window_data_dict[count_key]['event']
            events = set(events)
            if events == {'unlabeled'}:
                events = 'unlabeled'
            else:
                events.discard('unlabeled')
                events = ", ".join(events)
            self.window_data_dict[count_key]['event'] = events
            
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! watch out cuda memory
            for idx, body_name in value['idx_feats'].items():
                model_input = value[body_name]
                if self.preload_gpu:
                    self.window_data_dict[count_key][body_name] = model_input.to(self.device)
                else:
                    self.window_data_dict[count_key][body_name] = model_input
                             
            count_key += 1  
        
        mode_set = set()
        for key, value in self.window_data_dict.items():
            mode_set.add(value['ori_filename'])
        
        setattr(self.opt, f'{self.mode}_n_trials', len(mode_set))

        print(f"Finishing loading {dpath}")

    def _generate_train_val_test(self, root_dpath, train_datasets):
        """Generate and store train or val data for one specific dataset.
        
        Args:
            root_dpath: e.g. 'data/rectified_data'
            train_datasets: e.g. ['daphnet', 'kaggle_pd_data']
        """
        train_data_dict, val_data_dict, test_data_dict = {}, {}, {}
        train_count, val_count, test_count = 0, 0, 0
        train_seq_len, val_seq_len, test_seq_len = 0, 0, 0
        train_n_trials, val_n_trials, test_n_trials = 0, 0, 0
        
        for dataset in tqdm(train_datasets, total=len(train_datasets), 
                            desc='Generate training datasets'):
            dpath = os.path.join(root_dpath, 'all_data', f"all_{dataset}.p")
            try:
                single_data_dict = joblib.load(dpath)
            except Exception:
                print(f"\n**** {dpath} cannot be loaded.")
                return
        
            # count num of unique trials in current dataset
            total_num_trials = set()
            for key, value in single_data_dict.items():
                total_num_trials.add(value['ori_filename'])
            total_num_trials = len(total_num_trials)
            
            # print(f"\n{dataset} total_num_trials {total_num_trials} \n")
            
            # clear self.feats_data
            for key in self.feats_data.keys():
                self.feats_data[key] = None
            torch.cuda.empty_cache()
            
            cur_train_trials_name, cur_val_trials_name, cur_test_trials_name = set(), set(), set()
            cur_train_seq_len, cur_val_seq_len, cur_test_seq_len = 0, 0, 0
            cur_train_n_win, cur_val_n_win, cur_test_n_win = 0, 0, 0

            for series_name, series_info in tqdm(single_data_dict.items(), 
                                                total=len(single_data_dict),
                                                desc=dataset):
            
                series_len = len(series_info['gt'])

                # (series_len, n_orient)  e.g. (16641, 3)
                self._align_data(series_info, dataset)
            
                padding_len_ceil = math.ceil(series_len / self.window) * self.window - series_len
                padding_len_floor = math.floor(series_len / self.window) * self.window
                
                if padding_len_ceil < 0.3 * self.window:            
                    pad_gt = torch.ones(padding_len_ceil, dtype=torch.int8, 
                                            device=series_info['gt'].device) * 2
                    concate_gt = torch.cat([series_info['gt'], pad_gt], dim=0) # (T',)
                    
                    pad_event = np.array(['unlabeled'] * padding_len_ceil) # (T',)
                    concate_event = np.concatenate((series_info['event'], pad_event), axis=0)
                    
                    for key in self.feats_data.keys():
                        # (T',3)
                        self.feats_data[key] = torch.cat([self.feats_data[key], 
                                                        torch.zeros(padding_len_ceil, 3)], dim=0)
                
                else:
                    concate_event = series_info['event'][:padding_len_floor] # (T',)
                    concate_gt = series_info['gt'][:padding_len_floor] # (T',)
                    for key in self.feats_data.keys():
                        # (T',3)
                        self.feats_data[key] = self.feats_data[key][:padding_len_floor,:]
                        
        
                ori_filename = series_info['ori_filename']
                # Consider whether this series is included in validation
                if random.random() < 0.6 and len(cur_val_trials_name) <= 0.1 * total_num_trials:
                    # check if this window includes fog event
                    has_one_in_second_column = (concate_gt[:] == 1).any().item()
                    if has_one_in_second_column and ori_filename not in cur_test_trials_name \
                       and ori_filename not in cur_train_trials_name:
                        cur_val_trials_name.add(ori_filename)
                    
                # Consider whether this series is included in test set
                if random.random() < 0.6 and len(cur_test_trials_name) <= 0.1 * total_num_trials:
                    # check if this window includes fog event
                    has_one_in_second_column = (concate_gt[:] == 1).any().item()
                    if has_one_in_second_column and ori_filename not in cur_val_trials_name \
                       and ori_filename not in cur_train_trials_name:    
                        cur_test_trials_name.add(ori_filename)
                    
                if ori_filename in cur_val_trials_name and ori_filename in cur_test_trials_name:
                    raise Exception(f"{ori_filename} is in both valid and test set")
                
                if ori_filename not in cur_val_trials_name \
                   and ori_filename not in cur_test_trials_name:
                    cur_train_trials_name.add(series_info['ori_filename'])
        
                #* split by window =================================================================
                for i in range(0, concate_gt.shape[0], self.window):
                    start_t_idx = int(i)
                    end_t_idx = int(i + self.window)
                    
                    assert end_t_idx <= concate_gt.shape[0], "length unmatched"
                
                    # (window,)
                    gt = concate_gt[start_t_idx:end_t_idx].detach().clone() 
                    
                    # (window,3)
                    gt = torch.nn.functional.one_hot(gt.to(torch.int64), num_classes=3)
                    
                    assert is_valid_one_hot(gt), "not valid one hot encoding"
                    
                    # (window,3)
                    gt = gt.float()
                    
                    event = concate_event[start_t_idx:end_t_idx].tolist()
                    
                    added_window = {
                        'series_name': series_name,  #                           str
                        'ori_filename': series_info['ori_filename'], #           str
                        'start_t_idx': start_t_idx, # 0,     15552, ...               int
                        'end_t_idx': end_t_idx,     # 15552, 15552+15552, ...         int
                        'idx_feats': self.idx_feats, # {0: 'lowerback_acc', ...}
                        'event': event, # ['some event', ...] (T',)
                        # 'model_input': model_input.cpu(), # (window, num_feats)  torch
                        'gt': gt.cpu(), # (window, 3) one-hot                      torch
                    }
                    
                    # (window, num_feats)
                    for idx, body_name in added_window['idx_feats'].items():
                        model_input = self.feats_data[body_name][start_t_idx:end_t_idx]\
                                            .detach().clone()
                        added_window[body_name] = model_input.to(dtype=torch.float32).cpu()

                    # add to the together data dictionary
                    if series_info['ori_filename'] in cur_val_trials_name:
                        val_data_dict[val_count] = added_window
                        val_seq_len += added_window['gt'].shape[0]
                        val_count += 1
                        cur_val_seq_len += added_window['gt'].shape[0]
                        cur_val_n_win += 1
                    elif series_info['ori_filename'] in cur_test_trials_name:
                        test_data_dict[test_count] = added_window
                        test_seq_len += added_window['gt'].shape[0]
                        test_count += 1
                        cur_test_seq_len += added_window['gt'].shape[0]
                        cur_test_n_win += 1
                    else: # train data dict
                        for idx, body_name in added_window['idx_feats'].items():
                            model_input = added_window[body_name]
                            ori_dtype = model_input.dtype   
                            if self.random_aug:                                             
                                if random.random() <= 0.5:                                  
                                    # Augment data                                          
                                    model_input = self._augment_data(model_input.cpu()\
                                                                                .detach().numpy())
                            else: # augment all data                                        
                                model_input = self._augment_data(model_input.cpu().detach().numpy())
                            
                            model_input = model_input.to(ori_dtype)    
                            assert model_input.shape == (self.window, 3), "incorrect shape"
                            added_window[body_name] = model_input
                        
                        train_data_dict[train_count] = added_window
                        train_seq_len += added_window['gt'].shape[0]
                        train_count += 1
                        cur_train_seq_len += added_window['gt'].shape[0]
                        cur_train_n_win += 1
                #* finishing spliting current trials ===============================================
            
            train_n_trials += len(cur_train_trials_name)
            val_n_trials += len(cur_val_trials_name)
            test_n_trials += len(cur_test_trials_name)
            
            print()
            print(f"{dataset}:-----------------")
            print(f"        train n trials : {len(cur_train_trials_name)}") 
            print(f"        train seq len  : {cur_train_seq_len}")     
            print(f"        train n window : {cur_train_n_win}") 
            print()
            print(f"        val n trials   : {len(cur_val_trials_name)}") 
            print(f"        val seq len    : {cur_val_seq_len}")     
            print(f"        val n window   : {cur_val_n_win}") 
            print()
            print(f"        test n trials  : {len(cur_test_trials_name)}") 
            print(f"        test seq len   : {cur_test_seq_len}")     
            print(f"        test n window  : {cur_test_n_win}")                                                 
            print()
            # go to next dataset in for loop
        
        print()
        print("TOTAL -------------------------")
        print(f"        train n trials : {train_n_trials}")
        print(f"        train seq len  : {train_seq_len}")
        print(f"        train n window : {train_count}")
        print()
        print(f"        val n trials   : {val_n_trials}")
        print(f"        val seq len    : {val_seq_len}")
        print(f"        val n window   : {val_count}")
        print()
        print(f"        test n trials  : {test_n_trials}")
        print(f"        test seq len   : {test_seq_len}")
        print(f"        test n window  : {test_count}")
        print()
                                                                                
        train_dname = f"train_{self.all_dname}_window{self.window}_allaug.p"     
        val_dname = f"val_{self.all_dname}_window{self.window}_allaug.p"         
        test_dname = f"test_{self.all_dname}_window{self.window}_allaug.p"       
                                                                                
        if self.random_aug:                                                     
            train_dname = f"train_{self.all_dname}_window{self.window}_randomaug.p"
            val_dname = f"val_{self.all_dname}_window{self.window}_randomaug.p"  
            test_dname = f"test_{self.all_dname}_window{self.window}_randomaug.p"
                                                                                
        train_dpath = os.path.join(root_dpath, 'all_data', train_dname)       
        val_dpath = os.path.join(root_dpath, 'all_data', val_dname)           
        test_dpath = os.path.join(root_dpath, 'all_data', test_dname)         
                                                                                
        joblib.dump(train_data_dict, open(train_dpath, 'wb'))                   
        joblib.dump(val_data_dict, open(val_dpath, 'wb'))                       
        joblib.dump(test_data_dict, open(test_dpath, 'wb'))    
        
        # clear self.feats_data
        for key in self.feats_data.keys():
            self.feats_data[key] = None
        torch.cuda.empty_cache()                   
                                                                                
        print(f"Finishing spliting train, val, test data for {self.all_dname}\n")                

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
        return torch.tensor(model_input, dtype=torch.float32).clone().detach()

    def _align_data(self, series_info, dataset):
        if dataset == 'daphnet':
            l_latshank_acc = torch.zeros(series_info['gt'].shape[0], 3, dtype=torch.float32)  # (N,)
            r_latshank_acc = torch.zeros(series_info['gt'].shape[0], 3, dtype=torch.float32)  # (N,)
            l_latshank_gyr = torch.zeros(series_info['gt'].shape[0], 3, dtype=torch.float32)  # (N,)
            r_latshank_gyr = torch.zeros(series_info['gt'].shape[0], 3, dtype=torch.float32)  # (N,)
            lowerback_acc = torch.cat([series_info['LowerBack_Acc_AP'][:,None],
                                       series_info['LowerBack_Acc_ML'][:,None],
                                       series_info['LowerBack_Acc_V'][:,None]], dim=1)
            l_midlatthigh_acc = torch.cat([series_info['L_MidLatThigh_Acc_AP'][:,None],
                                           series_info['L_MidLatThigh_Acc_ML'][:,None],
                                           series_info['L_MidLatThigh_Acc_V'][:,None]], dim=1)
            l_ankle_acc = torch.cat([series_info['L_Ankle_Acc_AP'][:,None],
                                     series_info['L_Ankle_Acc_ML'][:,None],
                                     series_info['L_Ankle_Acc_V'][:,None]], dim=1)
        
        elif dataset == 'kaggle_pd_data_defog':
            l_latshank_acc = torch.zeros(series_info['gt'].shape[0], 3, dtype=torch.float32)  # (N,)
            r_latshank_acc = torch.zeros(series_info['gt'].shape[0], 3, dtype=torch.float32)  # (N,)
            l_latshank_gyr = torch.zeros(series_info['gt'].shape[0], 3, dtype=torch.float32)  # (N,)
            r_latshank_gyr = torch.zeros(series_info['gt'].shape[0], 3, dtype=torch.float32)  # (N,)
            l_midlatthigh_acc = torch.zeros(series_info['gt'].shape[0], 3, 
                                            dtype=torch.float32)  # (N,)
            l_ankle_acc = torch.zeros(series_info['gt'].shape[0], 
                                      3, dtype=torch.float32)  # (N,)
            lowerback_acc = torch.cat([series_info['LowerBack_Acc_AP'][:,None],
                                       series_info['LowerBack_Acc_ML'][:,None],
                                       series_info['LowerBack_Acc_V'][:,None]], dim=1)
        
        elif dataset == 'kaggle_pd_data_tdcsfog':
            l_latshank_acc = torch.zeros(series_info['gt'].shape[0], 3, dtype=torch.float32)  # (N,)
            r_latshank_acc = torch.zeros(series_info['gt'].shape[0], 3, dtype=torch.float32)  # (N,)
            l_latshank_gyr = torch.zeros(series_info['gt'].shape[0], 3, dtype=torch.float32)  # (N,)
            r_latshank_gyr = torch.zeros(series_info['gt'].shape[0], 3, dtype=torch.float32)  # (N,)
            l_midlatthigh_acc = torch.zeros(series_info['gt'].shape[0], 3, 
                                            dtype=torch.float32)  # (N,)
            l_ankle_acc = torch.zeros(series_info['gt'].shape[0], 
                                      3, dtype=torch.float32)  # (N,)
            lowerback_acc = torch.cat([series_info['LowerBack_Acc_AP'][:,None],
                                       series_info['LowerBack_Acc_ML'][:,None],
                                       series_info['LowerBack_Acc_V'][:,None]], dim=1)
        
        elif dataset == 'turn_in_place':
            side = 'L' if 'L_LatShank_Acc_ML' in series_info.keys() else 'R'
            
            if side == 'L':
                r_latshank_acc = torch.zeros(series_info['gt'].shape[0], 3, 
                                             dtype=torch.float32)  # (N,)
                r_latshank_gyr = torch.zeros(series_info['gt'].shape[0], 3, 
                                             dtype=torch.float32)  # (N,)
                l_latshank_acc = torch.cat([series_info['L_LatShank_Acc_AP'][:,None],
                                            series_info['L_LatShank_Acc_ML'][:,None],
                                            series_info['L_LatShank_Acc_SI'][:,None]], dim=1)
                l_latshank_gyr = torch.cat([series_info['L_LatShank_Gyr_AP'][:,None],
                                            series_info['L_LatShank_Gyr_ML'][:,None],
                                            series_info['L_LatShank_Gyr_SI'][:,None]], dim=1)
                
            else: 
                l_latshank_acc = torch.zeros(series_info['gt'].shape[0], 3, 
                                             dtype=torch.float32)  # (N,)
                l_latshank_gyr = torch.zeros(series_info['gt'].shape[0], 3, 
                                             dtype=torch.float32)  # (N,)
                r_latshank_acc = torch.cat([series_info['R_LatShank_Acc_AP'][:,None],
                                            series_info['R_LatShank_Acc_ML'][:,None],
                                            series_info['R_LatShank_Acc_SI'][:,None]], dim=1)
                r_latshank_gyr = torch.cat([series_info['R_LatShank_Gyr_AP'][:,None],
                                            series_info['R_LatShank_Gyr_ML'][:,None],
                                            series_info['R_LatShank_Gyr_SI'][:,None]], dim=1)
            l_midlatthigh_acc = torch.zeros(series_info['gt'].shape[0], 3, 
                                            dtype=torch.float32)  # (N,)
            l_ankle_acc = torch.zeros(series_info['gt'].shape[0], 
                                      3, dtype=torch.float32)  # (N,)
            lowerback_acc = torch.zeros(series_info['gt'].shape[0], 3, 
                                        dtype=torch.float32)  # (N,)
        elif dataset == 'turn_in_place_l' or dataset == 'turn_in_place_r':
            side = 'L' if 'L_LatShank_Acc_ML' in series_info.keys() else 'R'
            
            if side == 'L':
                l_latshank_acc = torch.cat([series_info['L_LatShank_Acc_AP'][:,None],
                                            series_info['L_LatShank_Acc_ML'][:,None],
                                            series_info['L_LatShank_Acc_SI'][:,None]], dim=1)
                l_latshank_gyr = torch.cat([series_info['L_LatShank_Gyr_AP'][:,None],
                                            series_info['L_LatShank_Gyr_ML'][:,None],
                                            series_info['L_LatShank_Gyr_SI'][:,None]], dim=1)
            else: 
                r_latshank_acc = torch.cat([series_info['R_LatShank_Acc_AP'][:,None],
                                            series_info['R_LatShank_Acc_ML'][:,None],
                                            series_info['R_LatShank_Acc_SI'][:,None]], dim=1)
                r_latshank_gyr = torch.cat([series_info['R_LatShank_Gyr_AP'][:,None],
                                            series_info['R_LatShank_Gyr_ML'][:,None],
                                            series_info['R_LatShank_Gyr_SI'][:,None]], dim=1)
            l_midlatthigh_acc = torch.zeros(series_info['gt'].shape[0], 3, 
                                            dtype=torch.float32)  # (N,)
            l_ankle_acc = torch.zeros(series_info['gt'].shape[0], 
                                      3, dtype=torch.float32)  # (N,)
            lowerback_acc = torch.zeros(series_info['gt'].shape[0], 3, 
                                        dtype=torch.float32)  # (N,)
        
        for feat in self.feats_data.keys():
            match feat:
                case 'lowerback_acc':
                    self.feats_data['lowerback_acc'] = lowerback_acc
                case 'l_midlatthigh_acc':
                    self.feats_data['l_midlatthigh_acc'] = l_midlatthigh_acc
                case 'l_latshank_acc':
                    self.feats_data['l_latshank_acc'] = l_latshank_acc
                case 'r_latshank_acc':
                    self.feats_data['r_latshank_acc'] = r_latshank_acc
                case 'l_latshank_gyr':
                    self.feats_data['l_latshank_gyr'] = l_latshank_gyr
                case 'r_latshank_gyr':
                    self.feats_data['r_latshank_gyr'] = r_latshank_gyr
                case 'l_ankle_acc':
                    self.feats_data['l_ankle_acc'] = l_ankle_acc
                case _:
                    raise Exception('features do not match')

    def __len__(self):
        return len(self.window_data_dict)

    def __getitem__(self, idx):
        return self.window_data_dict[idx]
        # return self.window_data_dict[idx]['model_input']
    
    


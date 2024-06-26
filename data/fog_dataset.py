import joblib, math, os, random, sys

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

        self.logger = opt.mylogger
        self.logger.info(f"-------- Initialize FoG dataset: {mode} mode --------")
        
        self.mode = mode
        self.feats = opt.feats
        self.window = opt.window
        self.random_aug = opt.random_aug
        self.device = opt.device
        
        self.opt = opt
        
        self.window_data_dict = {}
        
        for data_folder in tqdm(os.listdir(opt.root_dpath), total=len(os.listdir(opt.root_dpath)), 
                                desc=f"Load {self.mode} data", file=sys.stdout):
            if data_folder not in ALL_DATASETS or data_folder not in opt.train_datasets:
                continue
            
            if self.random_aug:
                # dname = f"{self.mode}_{data_folder}_blks{self.block_size}_"\
                #         f"ps{self.patch_size}_randomaug.p"
                dname = f"{self.mode}_{data_folder}_window{self.window}_randomaug.p"  
            else:
                dname = f"{self.mode}_{data_folder}_window{self.window}_allaug.p"                
            
            mode_dpath = os.path.join(opt.root_dpath, data_folder, dname)
            
            if not os.path.exists(mode_dpath):
                self._generate_train_val_data_window(opt.root_dpath, data_folder)

            self._load_train_val_data(mode_dpath) 
        
        print(f"{self.mode}: total block {len(self.window_data_dict)}")
                    
    def _load_train_val_data(self, dpath):
        try:
            d_dict = joblib.load(dpath)
        except Exception:
            self.logger.warning(f"{dpath} cannot be loaded.")
            print(f"\n**** {dpath} cannot be loaded.")
            return
        
        count_key = len(self.window_data_dict)
        for _, value in d_dict.items():
            self.window_data_dict[count_key] = value
            
            model_input = value['model_input'] # (window, num_feats)
            
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! watch out cuda memory
            self.window_data_dict[count_key]['model_input'] = model_input.to(self.device)
                                                                    
            count_key += 1  
        self.logger.info(f"Finishing loading {dpath}")
      
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
            self.logger.warning(f"{dpath} cannot be loaded.")
            print(f"\n**** {dpath} cannot be loaded.")
            return
        
        self.logger.info(f"_generate_train_val_data_full_trialGenerate and store train or val data for {dpath}")
        
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
        
        train_dname = f"train_{dataset_name}_full_trial_allaug.p"
        val_dname = f"val_{dataset_name}_full_trial_allaug.p"
        
        if self.random_aug:
            train_dname = f"train_{dataset_name}_window{self.window}_randomaug.p"
            val_dname = f"val_{dataset_name}_window{self.window}_randomaug.p"
     
        train_dpath = os.path.join(root_dpath, dataset_name, train_dname)
        val_dpath = os.path.join(root_dpath, dataset_name, val_dname)
        
        joblib.dump(train_data_dict, open(train_dpath, 'wb'))
        joblib.dump(val_data_dict, open(val_dpath, 'wb'))
        
        self.logger.info(f"Finishing spliting train and val data for {dataset_name}")

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
    
    


import joblib, math, os, sys

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.config import ALL_DATASETS, VALIDATION_SET

class FoGDataset(Dataset):
    '''
    Freezing of Gaits dataset for parkinson disease
    '''
    
    def __init__(self, opt, mode='train'):
        """Initialize FoGDataset.

        Args:
            opt.root_dpath: 'data/rectified_data'
        """
        from train.train_transformer_bilstm_v1 import MYLOGGER
        self.logger = MYLOGGER
        self.logger.info(f"-------- Initialize FoG dataset: {mode} mode --------")
        
        self.block_data_dict = {}
        self.block_size = opt.block_size
        self.block_stride = opt.block_stride
        self.patch_size = opt.patch_size
        self.mode = mode
        self.opt = opt
        
        for data_folder in tqdm(os.listdir(opt.root_dpath), total=len(os.listdir(opt.root_dpath)), 
                                desc=f"Load {self.mode} data", file=sys.stdout):
            if data_folder not in ALL_DATASETS:
                continue
            
            dname = f"{self.mode}_{data_folder}_blks{self.block_size}_ps{self.patch_size}.p"
            
            if opt.data_name is not None and len(opt.data_name) == 2:
                dname = opt.data_name[0] if mode == 'train' else opt.data_name[1]
            
            mode_dpath = os.path.join(opt.root_dpath, data_folder, dname)
            
            if not os.path.exists(mode_dpath):
                self._generate_train_val_data(opt.root_dpath, data_folder, opt.data_name)

            self._load_train_val_data(mode_dpath) 
        
        print(f"{self.mode}: total block {len(self.block_data_dict)}")
                    
    def _load_train_val_data(self, dpath):
        try:
            d_dict = joblib.load(dpath)
        except Exception:
            self.logger.warning(f"{dpath} cannot be loaded.")
            print(f"\n**** {dpath} cannot be loaded.")
            return
        
        count_key = len(self.block_data_dict)
        for _, value in d_dict.items():
            self.block_data_dict[count_key] = value
            
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! watch out cuda memory
            self.block_data_dict[count_key]['model_input'] = value['model_input'].\
                                                                    to(self.opt.device)
                                                                    
            count_key += 1  
        self.logger.info(f"Finishing loading {dpath}")
                
    def _generate_train_val_data(self, root_dpath, dataset_name, data_name):
        """Generate and store train or val data for one specific dataset.
        
        Args:
            root_dpath: e.g. 'data/rectified_data'
            dataset_name: e.g. 'dataset_fog_release'
            data_name: e.g. ['train.p', 'val.p']
        """
        dpath = os.path.join(root_dpath, dataset_name, f"all_{dataset_name}.p")
        try:
            single_data_dict = joblib.load(dpath)
        except Exception:
            self.logger.warning(f"{dpath} cannot be loaded.")
            print(f"\n**** {dpath} cannot be loaded.")
            return
        
        self.logger.info(f"Generate and store train or val data for {dpath}")
        
        count = 0
        all_data_dict = {}
        for series_name, series_info in tqdm(single_data_dict.items(), total=len(single_data_dict),
                                             desc="Generate train/val data", file=sys.stdout):
            series_len = len(series_info['gt'])
            block_count = math.ceil(series_len / self.block_size)
                        
            # (series_len, num_feats)  e.g. (16641, 9)
            concate_feat = torch.cat([series_info[feat][:, None] \
                                      for feat in series_info if feat != "gt"], dim=1)
            
            padding_len = block_count * self.block_size - series_len
            pad_feat = torch.zeros(padding_len, concate_feat.shape[1])
            pad_gt = torch.ones(padding_len, dtype=torch.int8) * 2
            
            # (block_count*block_size, num_feats)  e.g. (31104, 9)
            concate_feat = torch.cat([concate_feat, pad_feat], dim=0)
            
            # (block_count*block_size,)
            concate_gt = torch.cat([series_info['gt'], pad_gt], dim=0)
            
            for start_t_idx in range(0, block_count * self.block_size, self.block_stride):
                
                if self.block_size + start_t_idx > concate_feat.shape[0]:
                    break
                
                end_t_idx = start_t_idx + self.block_size
                
                model_input = concate_feat[start_t_idx:end_t_idx].clone()  # (BLKS, num_feats)
                
                model_input = model_input.reshape(self.block_size // self.patch_size, 
                                                  self.patch_size, 
                                                  -1)  # (BLKS//P, P, num_feats)

                # (BLKS//P, P*num_feats) e.g. (864, 18*9)
                model_input = model_input.reshape(model_input.shape[0], -1) 
                model_input = model_input.to(dtype=torch.float32)

                gt = concate_gt[start_t_idx:end_t_idx].clone() # (BLKS,)
                
                gt = torch.nn.functional.one_hot(gt.to(torch.int64), num_classes=3)
                # (BLKS,3)
                gt = gt.float()
                # (BLKS//P, P, 3)
                gt = gt.reshape(self.block_size // self.patch_size, self.patch_size, -1)
                # (BLKS//P, 3, P)
                gt = gt.permute(0, 2, 1)
                # (BLKS//P, 3)
                gt, _ = torch.max(gt, dim=-1)

                added_block = {
                    'series_name': series_name,
                    'start_t_idx': start_t_idx, # 0,     972, ...
                    'end_t_idx': end_t_idx,     # 15552, 16524, ...
                    'model_input': model_input, # (BLKS//P, P*num_feats)
                    'gt': gt, # (BLKS//P, 3) one-hot
                }
                
                all_data_dict[count] = added_block
                count += 1
        
        # split whole data 80% for train and  20% for val ------------------------------------------
        train_data_dict = {}
        val_data_dict = {}

        train_count, val_count = 0, 0
        train_seq_len, val_seq_len = 0, 0
        for key, value in all_data_dict.items():
            if value['series_name'] in VALIDATION_SET:
                val_data_dict[val_count] = value
                val_seq_len += value['model_input'].shape[0]
                val_count += 1
            else:
                train_data_dict[train_count] = value
                train_seq_len += value['model_input'].shape[0]
                train_count += 1

        train_dname = f"train_{dataset_name}_blks{self.block_size}_ps{self.patch_size}.p"
        val_dname = f"val_{dataset_name}_blks{self.block_size}_ps{self.patch_size}.p"
        if data_name is not None and len(data_name) == 2:
            train_dname = data_name[0]
            val_dname = data_name[1]
    
        train_dpath = os.path.join(root_dpath, dataset_name, train_dname)
        val_dpath = os.path.join(root_dpath, dataset_name, val_dname)
        
        joblib.dump(train_data_dict, open(train_dpath, 'wb'))
        joblib.dump(val_data_dict, open(val_dpath, 'wb'))
        
        self.logger.info(f"Finishing spliting train and val data for {dataset_name}")

    def __len__(self):
        return len(self.block_data_dict)

    def __getitem__(self, idx):
        return self.block_data_dict[idx]
        # return self.block_data_dict[idx]['model_input']
    
    


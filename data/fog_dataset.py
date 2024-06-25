import joblib, math, os, random, sys

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.config import ALL_DATASETS, VALIDATION_SET
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
        
        self.block_data_dict = {}
        self.block_size = opt.block_size
        self.block_stride = opt.block_stride
        self.patch_size = opt.patch_size
        self.mode = mode
        self.opt = opt
        self.num_feats = opt.num_feats
        self.random_aug = opt.random_aug
        self.device = self.opt.device
        
        for data_folder in tqdm(os.listdir(opt.root_dpath), total=len(os.listdir(opt.root_dpath)), 
                                desc=f"Load {self.mode} data", file=sys.stdout):
            if data_folder not in ALL_DATASETS or data_folder not in opt.train_datasets:
                continue
            
            if self.random_aug:
                dname = f"{self.mode}_{data_folder}_blks{self.block_size}_"\
                        f"ps{self.patch_size}_randomaug.p" 
            else:
                dname = f"{self.mode}_{data_folder}_blks{self.block_size}_"\
                        f"ps{self.patch_size}_allaug.p"                
            
            mode_dpath = os.path.join(opt.root_dpath, data_folder, dname)
            
            if not os.path.exists(mode_dpath):
                self._generate_train_val_data(opt.root_dpath, data_folder)

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
            
            model_input = value['model_input'] # (BLKS//P, P*num_feats)
            
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! watch out cuda memory
            self.block_data_dict[count_key]['model_input'] = model_input.to(self.device)
                                                                    
            count_key += 1  
        self.logger.info(f"Finishing loading {dpath}")
                
    def _generate_train_val_data(self, root_dpath, dataset_name):
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
        
        self.logger.info(f"Generate and store train or val data for {dpath}")
        
        count = 0
        all_data_dict = {}
        validation_set = set() # series_name: total 970, pick 100
        for series_name, series_info in tqdm(single_data_dict.items(), 
                                             total=len(single_data_dict),
                                             desc=f"Gen train/val data: {dataset_name}", 
                                             file=sys.stdout):
            series_len = len(series_info['gt'])
            block_count = math.ceil(series_len / self.block_size)
                        
            # (series_len, num_feats)  e.g. (16641, 9)
            concate_feat = []
            for feat in series_info:
                if feat == "gt" or feat == "Annotation" or feat == "ori_filename":
                    continue
                series_info[feat] = series_info[feat].to(self.device)
                concate_feat.append(series_info[feat][:, None])
            concate_feat = torch.cat(concate_feat, dim=1)
            
            series_info['gt'] = series_info['gt'].to(self.device)
            
            padding_len = block_count * self.block_size - series_len
            pad_feat = torch.zeros(padding_len, concate_feat.shape[1], device=concate_feat.device)
            
            pad_gt = torch.zeros(padding_len, dtype=torch.int8, 
                                    device=series_info['gt'].device) * 2

            # (block_count*block_size, num_feats)  e.g. (31104, 9)
            concate_feat = torch.cat([concate_feat, pad_feat], dim=0)
            
            # (block_count*block_size,)
            concate_gt = torch.cat([series_info['gt'], pad_gt], dim=0)
            
            # Consider whether this series is included in validation
            if random.random() < 0.4 and len(validation_set) < 18:
                if series_name not in validation_set:
                    validation_set.add(series_name)
            
            #* Split by patch size and block size
            for start_t_idx in range(0, block_count * self.block_size, self.block_stride):
                
                if self.block_size + start_t_idx > concate_feat.shape[0]:
                    break
                
                end_t_idx = start_t_idx + self.block_size
                
                model_input = concate_feat[start_t_idx:end_t_idx].clone()  # (BLKS, num_feats)
                
                model_input = model_input.reshape(self.block_size // self.patch_size, 
                                                  self.patch_size, 
                                                  -1)  # (BLKS//P, P, num_feats)

                # (BLKS//P, P*num_feats) e.g. (864, 18*9) (864, 18*3)
                model_input = model_input.reshape(model_input.shape[0], -1) 
                model_input = model_input.to(dtype=torch.float32)

                gt = concate_gt[start_t_idx:end_t_idx].clone() # (BLKS,)
                
                gt = torch.nn.functional.one_hot(gt.to(torch.int64), num_classes=3)
                # (BLKS,3)
                gt = gt.float()
                # (BLKS,3)
                inference_gt = gt.clone()
                # (BLKS//P, P, 3)
                gt = gt.reshape(self.block_size // self.patch_size, self.patch_size, -1)
                
                # (BLKS//P, 3, P)
                gt = gt.permute(0, 2, 1)
                
                #* only when half of the "patch size" is one then this BLKS//P can be one
                threshold = gt.shape[2] / 2
                count_ones = torch.sum(gt, dim=-1) # (BLKS//P, 3)
                
                gt = (count_ones > threshold).float() # (BLKS//P, 3)
            
                sum_along_second_dim = torch.sum(gt, dim=1) # (BLKS//P, )
                # Check if any row has more than one `1`
                exists_more_than_one_one = torch.any(sum_along_second_dim > 1.1)
                if exists_more_than_one_one.item():
                    non_one_hot_rows = torch.where(sum_along_second_dim > 1.1)
                    print("++++++")
                    print(gt[-10:,:])
                    print()
                    print(sum_along_second_dim)
                    print(sum_along_second_dim.shape)
                    print("Rows that are not one-hot encoded:", non_one_hot_rows)
                    print(series_name)
                    print(series_info['ori_filename'])
                    print(start_t_idx)
                    self.logger.error('Not one hot encoding...')
                    exit(1)
                
                added_block = {
                    'series_name': series_name,  #                              str
                    'ori_series_name': series_info['ori_filename'], #           str
                    'start_t_idx': start_t_idx, # 0,     972, ...               int
                    'end_t_idx': end_t_idx,     # 15552, 16524, ...             int
                    'model_input': model_input.cpu(), # (BLKS//P, P*num_feats)  torch
                    'gt': gt.cpu(), # (BLKS//P, 3) one-hot                      torch
                    'inference_gt': inference_gt.cpu(),  # (BLKS, 3) one-hot
                }
                
                all_data_dict[count] = added_block
                count += 1
                
        tmp_count = 0
        
        # Initialize counters and sequence lengths
        train_count, val_count = 0, 0
        train_seq_len, val_seq_len = 0, 0
        # Initialize dictionaries for training and validation data
        train_data_dict = {}
        val_data_dict = {}
        for key, value in tqdm(all_data_dict.items(),
                               total=len(all_data_dict.keys()), 
                               desc="split train val"):
            
            if value['series_name'] in validation_set:
                tmp_count += 1
                val_data_dict[val_count] = value
                val_seq_len += value['model_input'].shape[0]
                val_count += 1
            else:  # training set
                #* data augmentation ***********************************************
                model_input = value['model_input'] # (BLKS//P, P*num_feats) (864,18*3)
                ori_dtype = model_input.dtype
                if self.random_aug:
                    if random.random() <= 0.5:
                        # Augment data
                        model_input = self._augment_data(model_input.cpu().detach().numpy())
                else: # augment all data
                    model_input = self._augment_data(model_input.cpu().detach().numpy())
                # torch.Size([864, 18, 3])
                model_input = model_input.to(ori_dtype) 
                # (BLKS//P, P*num_feats)
                model_input = model_input.reshape(model_input.shape[0], -1)
                value['model_input'] = model_input
                #*******************************************************************
                
                train_data_dict[train_count] = value
                train_seq_len += value['model_input'].shape[0]
                train_count += 1
                
        print(f"++++++++++ {tmp_count} +++++++: ")
        print()
        print(validation_set)
        print()
        
        print(f"----- train seq len: {train_seq_len} -----")
        print(f"----- val seq len: {val_seq_len} -----")
        print()
        
        train_dname = f"train_{dataset_name}_blks{self.block_size}_ps{self.patch_size}_allaug.p"
        val_dname = f"val_{dataset_name}_blks{self.block_size}_ps{self.patch_size}_allaug.p"
        
        if self.random_aug:
            train_dname = f"train_{dataset_name}_blks{self.block_size}" \
                          f"_ps{self.patch_size}_randomaug.p"
            val_dname = f"val_{dataset_name}_blks{self.block_size}"\
                        f"_ps{self.patch_size}_randomaug.p"
     
        train_dpath = os.path.join(root_dpath, dataset_name, train_dname)
        val_dpath = os.path.join(root_dpath, dataset_name, val_dname)
        
        joblib.dump(train_data_dict, open(train_dpath, 'wb'))
        joblib.dump(val_data_dict, open(val_dpath, 'wb'))
        
        self.logger.info(f"Finishing spliting train and val data for {dataset_name}")

    def _augment_data(self, model_input):
        model_input = model_input.reshape(self.block_size//self.patch_size, 
                                                    self.patch_size, 
                                                    self.num_feats)
        # model_input = model_input.reshape(self.block_size//self.patch_size, 
        #                                 self.patch_size,
        #                                 3,3) # (864,18,3,3)
        theta = random.random()*math.pi*2
        theta = random.random()*360
        a=random.random()
        b=random.random()
        c=random.random()
        model_input = rotateC(model_input, theta, a, b, c)
        return torch.tensor(model_input).clone().detach()

    def __len__(self):
        return len(self.block_data_dict)

    def __getitem__(self, idx):
        return self.block_data_dict[idx]
        # return self.block_data_dict[idx]['model_input']
    
    


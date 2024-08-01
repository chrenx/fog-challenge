import argparse, csv, joblib, math, os, shutil, torch
from collections import OrderedDict

import pandas as pd
import numpy as np
from tqdm import tqdm


# DATASETS_FEATS = {
#     'turn': ['L_LatShank_Acc_X', 'L_LatShank_Acc_Y', 'L_LatShank_Acc_Z', 
#                       'R_LatShank_Acc_X', 'R_LatShank_Acc_Y', 'R_LatShank_Acc_Z', 
#                       'L_LatShank_Gyr_X', 'L_LatShank_Gyr_Y', 'L_LatShank_Gyr_Z', 
#                       'R_LatShank_Gyr_X', 'R_LatShank_Gyr_Y', 'R_LatShank_Gyr_Z'],
#     'kaggle': ['LowerBack_Acc_X', 'LowerBack_Acc_Y', 'LowerBack_Acc_Z'],
#     'daphnet': ['LowerBack_Acc_X', 'LowerBack_Acc_Y', 'LowerBack_Acc_Z', 
#                 'L_MidLatThigh_Acc_X', 'L_MidLatThigh_Acc_Y', 'L_MidLatThigh_Acc_X', 
#                 'L_Ankle_Acc_X', 'L_Ankle_Acc_Y', 'L_Ankle_Acc_Z']
# }

DATASETS_FEATS_MODEL = {
    'kaggle': ['lowerback_acc'],
    'turn': ['l_latshank_acc', 'r_latshank_acc', 'l_latshank_gyr', 'r_latshank_gyr'],
    'daphnet': ['lowerback_acc', 'l_midlatthigh_acc', 'l_ankle_acc']
}

CAT_FEATS = {
    'lowerback_acc': ['LowerBack_Acc_X', 'LowerBack_Acc_Y', 'LowerBack_Acc_Z'],
    'l_midlatthigh_acc': ['L_MidLatThigh_Acc_X', 'L_MidLatThigh_Acc_Y', 'L_MidLatThigh_Acc_X'],
    'l_ankle_acc': ['L_Ankle_Acc_X', 'L_Ankle_Acc_Y', 'L_Ankle_Acc_Z'],
    'l_latshank_acc': ['L_LatShank_Acc_X', 'L_LatShank_Acc_Y', 'L_LatShank_Acc_Z'],
    'r_latshank_acc': ['R_LatShank_Acc_X', 'R_LatShank_Acc_Y', 'R_LatShank_Acc_Z'],
    'l_latshank_gyr': ['L_LatShank_Gyr_X', 'L_LatShank_Gyr_Y', 'L_LatShank_Gyr_Z'],
    'r_latshank_gyr': ['R_LatShank_Gyr_X', 'R_LatShank_Gyr_Y', 'R_LatShank_Gyr_Z']
}

PERMUTATIONS = [[0,1,2], [0,2,1], [1,0,2], [1,2,0], [2,0,1], [2,1,0]]
FEATURES = ['LowerBack_Acc_X', 'LowerBack_Acc_Y', 'LowerBack_Acc_Z']
WINDOW = 1024

WEIGHTS = {
    'kaggle': 0.8,
    'turn': 0.15,
    'daphnet': 0.05,
}

def dict_to_csv(data_dict, csv_file_path):
    # Determine the maximum length among all tensors
    max_length = max(tensor.size(0) for tensor in data_dict.values())

    # Create a dictionary with lists, padding shorter lists with None
    csv_dict = {}
    for key, tensor in data_dict.items():
        list_values = tensor.tolist()
        list_values.extend([None] * (max_length - len(list_values)))
        csv_dict[key] = list_values
    # Create a DataFrame from the dictionary
    df = pd.DataFrame(csv_dict)
    # Write the DataFrame to a CSV file
    df.to_csv(csv_file_path, index=False)

def check_and_find_length(tensor):
    B, window, _ = tensor.shape
    lengths = []

    for b in range(B):
        # Find the first position with 1 in the third column
        first_one_pos = (tensor[b, :, 2] == 1).nonzero(as_tuple=True)[0]
        
        if first_one_pos.numel() == 0:
            # raise ValueError(f"Batch {b} does not contain any '1' in the third column.")
            lengths.append(window)
            continue
        
        first_one_pos = first_one_pos[0].item()
        
        # Check if all subsequent positions in the third column are 1
        if not torch.all(tensor[b, first_one_pos:, 2] == 1):
            raise ValueError(f"Batch {b} contains values other than '1' in the third column after position {first_one_pos}.")
        
        lengths.append(first_one_pos)

    return lengths

def rearrange_output(data_dict):
    for file, value in data_dict.items():
        start_t_idx = value['start_t_idx']
        end_t_idx = value['end_t_idx']
        output = value['output']
        actual_len = value['actual_len']

        # Zip the lists together
        combined = list(zip(start_t_idx, end_t_idx, actual_len, output))

        # Sort by start_t_idx
        combined.sort(key=lambda x: x[0])

        # Unzip the sorted list back into separate lists
        sorted_start_t_idx, sorted_end_t_idx, sorted_actual_len, sorted_output = zip(*combined)

        # Update the dictionary with the sorted lists
        value['start_t_idx'] = list(sorted_start_t_idx)
        value['end_t_idx'] = list(sorted_end_t_idx)
        value['actual_len'] = list(sorted_actual_len)
        value['output'] = list(sorted_output)

def cycle_dataloader(dl):
    while True:
        for data in dl:
            yield data

def is_valid_one_hot(matrix):
    # Check if the matrix has shape (N, 3)
    if matrix.size(1) != 3:
        return False

    # Check if each row has exactly one '1' and the rest '0'
    for row in matrix:
        if torch.sum(row) != 1 or not torch.all((row == 0) | (row == 1)):
            return False

    return True

def sample_normalize(sample):
    """Mean-std normalization function. 

    Args:
        sample: (N,)
    Returns:
        normalized_sample: (N,)
    """
    if not isinstance(sample, torch.Tensor):
        sample = torch.tensor(sample)
    mean = torch.mean(sample)
    std = torch.std(sample)
    # Normalize the sample and handle division by zero
    eps = 1e-8
    normalized_sample = (sample - mean) / (std + eps)
    return normalized_sample  # (N,)

def is_valid_one_hot(matrix):
    # Check if the matrix has shape (N, 3)
    if matrix.size(1) != 3:
        return False

    # Check if each row has exactly one '1' and the rest '0'
    for row in matrix:
        if torch.sum(row) != 1 or not torch.all((row == 0) | (row == 1)):
            return False

    return True

def evaluation_metrics(output, gt):
    """Generate precision, recall, and f1 score.

    Args:
        output: (B, window, 1)   # prob class
        gt (inference):   (B, window, 3)   # one hot
    """
    # Convert the model output probabilities to class predictions
    pred = torch.round(output)  # (B, window, 1)

    # Extract the first two classes from the ground truth
    real = torch.argmax(gt[:, :, :2], dim=-1, keepdim=True)  # (B, window, 1)

    # Create a mask to ignore the positions where the ground truth class is 2
    mask = (gt[:, :, 2] != 1).unsqueeze(-1)  # (B, window, 1)

    # Apply the mask to the predictions and ground truth
    pred = (pred * mask.float()).squeeze() # (B, window)
    real = (real * mask.float()).squeeze() # (B, window)
    

    # Calculate true positives, false positives, and false negatives
    tp = ((pred == 1) & (real == 1)).float().sum().item()
    fp = ((pred == 1) & (real == 0)).float().sum().item()
    fn = ((pred == 0) & (real == 1)).float().sum().item()

    return tp, fp, fn


def split_by_window(opt, model_name):
    all_data = joblib.load(os.path.join(opt.test_dpath, f"all_{model_name}.p"))
    
    test_data_dict = {}
    test_count = 0

    for series_idx, series_info in tqdm(all_data.items(), total=len(all_data.keys()), 
                                        desc=f"split {model_name}"):
    
        series_len = len(series_info['gt'])

        padding_len_ceil = math.ceil(series_len / WINDOW) * WINDOW - series_len
        padding_len_floor = math.floor(series_len / WINDOW) * WINDOW

        ############################################################################################
        # if padding_len_ceil < 0.3 * WINDOW:            
        #     pad_gt = torch.ones(padding_len_ceil, dtype=torch.int8, 
        #                             device=series_info['gt'].device) * 2
        #     concate_gt = torch.cat([series_info['gt'], pad_gt], dim=0) # (T',)
            
        #     for cat_feat in DATASETS_FEATS_MODEL[model_name]:
        #         # (T',3)
        #         all_data[series_idx][cat_feat] = torch.cat([all_data[series_idx][cat_feat], 
        #                                                 torch.zeros(padding_len_ceil, 3)], dim=0)
        # else:
        #     concate_gt = series_info['gt'][:padding_len_floor] # (T',)
        #     for cat_feat in DATASETS_FEATS_MODEL[model_name]:
        #         # (T',3)
        #         all_data[series_idx][cat_feat] = all_data[series_idx][cat_feat][:padding_len_floor,:]
        ############################################################################################
        pad_gt = torch.ones(padding_len_ceil, dtype=torch.int8, 
                                device=series_info['gt'].device) * 2
        concate_gt = torch.cat([series_info['gt'], pad_gt], dim=0) # (T',)

        for cat_feat in DATASETS_FEATS_MODEL[model_name]:
            # (T',3)
            all_data[series_idx][cat_feat] = torch.cat([all_data[series_idx][cat_feat], 
                                                        torch.zeros(padding_len_ceil, 3)], dim=0)
        ############################################################################################

        #* split by window =================================================================
        for i in range(0, concate_gt.shape[0], WINDOW):
            start_t_idx = int(i)
            end_t_idx = int(i + WINDOW)
            
            assert end_t_idx <= concate_gt.shape[0], "length unmatched"
        
            # (window,)
            gt = concate_gt[start_t_idx:end_t_idx].detach().clone() 
            
            # (window,3)
            gt = torch.nn.functional.one_hot(gt.to(torch.int64), num_classes=3)
            
            assert is_valid_one_hot(gt), "not valid one hot encoding"
            
            # (window,3)
            gt = gt.float()
                        
            added_window = {
                'ori_filename': series_info['ori_filename'], #           str
                'start_t_idx': start_t_idx, # 0,     15552, ...               int
                'end_t_idx': end_t_idx,     # 15552, 15552+15552, ...         int
                # 'model_input': model_input.cpu(), # (window, num_feats)  torch
                'gt': gt.cpu(), # (window, 3) one-hot                      torch
            }

            for body_name in DATASETS_FEATS_MODEL[model_name]:
                model_input = all_data[series_idx][body_name][start_t_idx:end_t_idx]\
                                    .detach().clone()
                added_window[body_name] = model_input.to(dtype=torch.float32).cpu()
            
            # (window, num_feats)
            test_data_dict[test_count] = added_window
            test_count += 1

    return test_data_dict

def update_all_data_from_csv(opt, df, model_name):
    res = {}
    for body_name in DATASETS_FEATS_MODEL[model_name]:
        tmp = []
        for ele in CAT_FEATS[body_name]:
            if opt.enable_self_test:
                tmp.append(torch.tensor(df[ele], dtype=torch.float32)[:, None])
            else:
                tmp.append(sample_normalize(torch.tensor(df[ele], dtype=torch.float32))[:, None])
        res[body_name] = torch.cat(tmp, dim=1) # (N,3)
           
    return res # {'lowerback_acc': (N,3), ...}

def process_csv_files(opt, model_name):
    test_csv_files = os.listdir(opt.test_dpath)

    all_data = {}
    count = 0

    gt_file = os.listdir(opt.gt_dpath)
    assert len(gt_file) == 1, "too many ground truth files, suppose to have only one file"
    gt_data = pd.read_csv(os.path.join(opt.gt_dpath, gt_file[0]))

    for filename in tqdm(test_csv_files):
        if not filename.endswith('.csv'):
            continue
        
        df_filtered = pd.read_csv(os.path.join(opt.test_dpath, filename))

        all_data[count] = update_all_data_from_csv(opt, df_filtered, model_name)
     
        tmp_gt = torch.tensor(gt_data[filename[:-4]].dropna(), dtype=torch.float32)

        if 'lowerback_acc' in all_data[count]:
            assert tmp_gt.shape[0] == all_data[count]['lowerback_acc'].shape[0], "unmatched length"
        elif 'l_latshank_acc' in all_data[count]:
            assert tmp_gt.shape[0] == all_data[count]['l_latshank_acc'].shape[0], "unmatched length"
        elif 'r_latshank_acc' in all_data[count]:
            assert tmp_gt.shape[0] == all_data[count]['r_latshank_acc'].shape[0], "unmatched length"
        else:
            raise "No proper features."

        all_data[count]['ori_filename'] = filename
        all_data[count]['gt'] = tmp_gt

        count += 1

    joblib.dump(all_data, open(os.path.join(opt.test_dpath, f"all_{model_name}.p"), 'wb'))
    test_data = split_by_window(opt, model_name)

    joblib.dump(test_data, open(os.path.join(opt.test_dpath, 
                                            f"all_test_data_{model_name}_window{WINDOW}.p"), 'wb'))

def generate_gt_csv(opt):
    gt_file = os.listdir(opt.gt_dpath)[0]
    gt_path = os.path.join(opt.gt_dpath, gt_file)
    shutil.copy2(gt_path, 'submission/GuanRen_FoG_Ground_Truth_Data.csv')
    print('Generate ground truth csv.')
        
def generate_model_output_csv(opt):

    from codes.fog_dataset import FoGDataset
    test_ds_kaggle = FoGDataset(opt, 'kaggle')
    dl = torch.utils.data.DataLoader(test_ds_kaggle, 
                        batch_size=opt.batch_size, 
                        shuffle=False, 
                        pin_memory=False, 
                        num_workers=0)
    test_n_batch_kaggle = len(dl)
    test_dl_kaggle = cycle_dataloader(dl)

    test_ds_turn = FoGDataset(opt, 'turn')
    dl = torch.utils.data.DataLoader(test_ds_turn, 
                        batch_size=opt.batch_size, 
                        shuffle=False, 
                        pin_memory=False, 
                        num_workers=0)
    test_n_batch_turn = len(dl)
    test_dl_turn = cycle_dataloader(dl)

    test_ds_daphnet = FoGDataset(opt, 'daphnet')
    dl = torch.utils.data.DataLoader(test_ds_daphnet, 
                        batch_size=opt.batch_size, 
                        shuffle=False, 
                        pin_memory=False, 
                        num_workers=0)
    test_n_batch_daphnet = len(dl)
    test_dl_daphnet = cycle_dataloader(dl)

    assert test_n_batch_kaggle == test_n_batch_turn and test_n_batch_turn == test_n_batch_daphnet, \
            "unequal length of dataset"

    ############################################################################
    from codes.unet_v4 import UNet
    weights_path = f"codes/kaggle.pt"
    model_kaggle = UNet(channel=len(DATASETS_FEATS_MODEL['kaggle'])*3, 
                 feats=DATASETS_FEATS_MODEL['kaggle'])
    model_kaggle = model_kaggle.to(opt.device)
    weights = torch.load(weights_path, map_location=opt.device, weights_only=True)['model']
    model_kaggle.load_state_dict(weights)

    weights_path = f"codes/turn.pt"
    model_turn = UNet(channel=len(DATASETS_FEATS_MODEL['turn'])*3, 
                 feats=DATASETS_FEATS_MODEL['turn'])
    model_turn = model_turn.to(opt.device)
    weights = torch.load(weights_path, map_location=opt.device, weights_only=True)['model']
    model_turn.load_state_dict(weights)

    weights_path = f"codes/daphnet.pt"
    model_daphnet = UNet(channel=len(DATASETS_FEATS_MODEL['daphnet'])*3, 
                 feats=DATASETS_FEATS_MODEL['daphnet'])
    model_daphnet = model_daphnet.to(opt.device)
    weights = torch.load(weights_path, map_location=opt.device, weights_only=True)['model']
    model_daphnet.load_state_dict(weights)

    ############################################################################


    model_output = {}

    cum_tp, cum_fp, cum_fn = 0.0, 0.0, 0.0     

    model_kaggle.eval(), model_turn.eval(), model_daphnet.eval() 

    with torch.no_grad():
        for _ in tqdm(range(test_n_batch_kaggle), desc=f"test data"):
            test_data_kaggle = next(test_dl_kaggle) 
            test_data_turn = next(test_dl_turn) 
            test_data_daphnet = next(test_dl_daphnet)  

            assert test_data_kaggle['ori_filename'] == test_data_turn['ori_filename'] \
                and test_data_daphnet['ori_filename'] == test_data_turn['ori_filename'], \
                "test dataloader ori_filename doesn't match"
            assert test_data_kaggle['start_t_idx'] == test_data_turn['start_t_idx'] \
                and test_data_daphnet['start_t_idx'] == test_data_turn['start_t_idx'], \
                "test dataloader start_t_idx doesn't match"
            assert test_data_kaggle['end_t_idx'] == test_data_turn['end_t_idx'] \
                and test_data_daphnet['end_t_idx'] == test_data_turn['end_t_idx'], \
                "test dataloader end_t_idx doesn't match"

            test_gt = test_data_kaggle['gt'] # (B, window, 3)
            
            ############################################################################
            test_pred_kaggle = 0
            for permutation in PERMUTATIONS:
                test_input = {}  
                for body_name in DATASETS_FEATS_MODEL['kaggle']:  
                    # (BS, window, 3)
                    test_input[body_name] = test_data_kaggle[body_name][:,:,permutation]
                    if not opt.preload_gpu:
                        test_input[body_name] = test_input[body_name].to(opt.device)
                test_pred_kaggle += model_kaggle(test_input) # (B, window, 1)
            test_pred_kaggle /= len(PERMUTATIONS)
            test_pred_kaggle *= WEIGHTS['kaggle']

            # --------------------------------------------------
            test_pred_turn_l = 0
            side = 'l'
            for permutation in PERMUTATIONS:
                test_input = {}  
                for body_name in DATASETS_FEATS_MODEL['turn']:  
                    # (BS, window, 3)
                    if body_name[0] != side:
                        test_input[body_name] = torch.zeros_like(test_data_turn[body_name])\
                                                .to(test_data_turn[body_name].device)
                    else:
                        test_input[body_name] = test_data_turn[body_name][:,:,permutation]
                    if not opt.preload_gpu:
                        test_input[body_name] = test_input[body_name].to(opt.device)
                test_pred_turn_l += model_turn(test_input) # (B, window, 1)
            test_pred_turn_l /= len(PERMUTATIONS)

            test_pred_turn_r = 0
            side = 'r'
            for permutation in PERMUTATIONS:
                test_input = {}  
                for body_name in DATASETS_FEATS_MODEL['turn']:  
                    # (BS, window, 3)
                    if body_name[0] != side:
                        test_input[body_name] = torch.zeros_like(test_data_turn[body_name])\
                                                .to(test_data_turn[body_name].device)
                    else:
                        test_input[body_name] = test_data_turn[body_name][:,:,permutation]
                    if not opt.preload_gpu:
                        test_input[body_name] = test_input[body_name].to(opt.device)
                test_pred_turn_r += model_turn(test_input) # (B, window, 1)
            test_pred_turn_r /= len(PERMUTATIONS)

            test_pred_turn = (test_pred_turn_l + test_pred_turn_r) / 2
            test_pred_turn *= WEIGHTS['turn']
            # --------------------------------------------------

            test_pred_daphnet = 0
            for permutation in PERMUTATIONS:
                test_input = {}  
                for body_name in DATASETS_FEATS_MODEL['daphnet']:  
                    # (BS, window, 3)
                    test_input[body_name] = test_data_daphnet[body_name][:,:,permutation]
                    if not opt.preload_gpu:
                        test_input[body_name] = test_input[body_name].to(opt.device)
                test_pred_daphnet += model_daphnet(test_input) # (B, window, 1)
            test_pred_daphnet /= len(PERMUTATIONS)
            test_pred_daphnet *= WEIGHTS['daphnet']
            ############################################################################


            test_pred = test_pred_kaggle + test_pred_turn + test_pred_daphnet

            tp, fp, fn = evaluation_metrics(test_pred, test_gt.to(opt.device))
            
            cum_tp += tp
            cum_fp += fp
            cum_fn += fn

            gt_lengths = check_and_find_length(test_gt) # [...] w/ length B

            # keep track of model output #######################################
            for i in range(test_pred.shape[0]):
                ori_filename = test_data_kaggle['ori_filename'][i][:-4]
                if ori_filename not in model_output:
                    model_output[ori_filename] = {
                        'start_t_idx': [], # [0, 1024, ...]
                        'end_t_idx': [], # [1024, 2048, ...]
                        'output': [], # (windows)
                        'actual_len': [] # w/ length batch_size
                    }
    
                model_output[ori_filename]['start_t_idx'].append(
                    test_data_kaggle['start_t_idx'][i].item()  
                )
                model_output[ori_filename]['end_t_idx'].append(
                    test_data_kaggle['end_t_idx'][i].item()
                )
                model_output[ori_filename]['output'].append(
                    test_pred[i, :, :].squeeze().cpu()
                )
                model_output[ori_filename]['actual_len'].append(gt_lengths[i])


        prec = 0 if cum_tp == 0 else cum_tp / (cum_tp + cum_fp)
        recall = 0 if cum_tp == 0 else cum_tp / (cum_tp + cum_fn)
        f1 = 0 if prec == 0 or recall == 0 else 2 * (prec * recall) / (prec + recall)
    
    print('prec, recall, f1', prec, recall, f1)

    statistics = {'F1 score': round(f1, 4), 'precision': round(prec, 4), 'recall': round(recall, 4)}
    with open('submission/GuanRen_Statistics.csv', 'w', newline='') as csvfile:
        fieldnames = ['F1 score', 'precision','recall']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(statistics)
        print('Generate Statistics csv.')

    rearrange_output(model_output)

    for key, value in model_output.items():
        value['output'] = torch.cat(value['output']) # (N*window,)
        actual_len = sum(value['actual_len'])
        value['output'] = torch.round(value['output'][:actual_len])
        model_output[key] = value['output']

    dict_to_csv(model_output, 'submission/GuanRen_Model_Output.csv')
    print('Generate model output csv.')


def self_test_generate_model_output_csv(opt):
    model_name = 'kaggle'
    from codes.fog_dataset import FoGDataset
    test_ds = FoGDataset(opt, model_name)
    dl = torch.utils.data.DataLoader(test_ds, 
                        batch_size=opt.batch_size, 
                        shuffle=False, 
                        pin_memory=False, 
                        num_workers=0)
    test_n_batch = len(dl)
    test_dl = cycle_dataloader(dl)

    from codes.unet_v4 import UNet
    weights_path = f"codes/{model_name}.pt"
    model = UNet(channel=len(DATASETS_FEATS_MODEL[model_name])*3, 
                 feats=DATASETS_FEATS_MODEL[model_name])
    model = model.to(opt.device)
    weights = torch.load(weights_path, map_location=opt.device, weights_only=True)['model']
    model.load_state_dict(weights)


    model_output = {}

    cum_tp, cum_fp, cum_fn = 0.0, 0.0, 0.0      
    model.eval()
    with torch.no_grad():
        for _ in tqdm(range(test_n_batch), desc=f"test data"):
            test_data = next(test_dl) 
            test_gt = test_data['gt'] # (B, window, 3)
            
            test_pred = 0
            for permutation in PERMUTATIONS:
                test_input = {}  

                for body_name in DATASETS_FEATS_MODEL[model_name]:  
                    # (BS, window, 3)
                    test_input[body_name] = test_data[body_name][:,:,permutation]

                    if not opt.preload_gpu:
                        test_input[body_name[0]] = test_input[body_name[0]].to(opt.device)

                test_pred += model(test_input) # (B, window, 1)
            
            test_pred /= len(PERMUTATIONS)
            
            tp, fp, fn = evaluation_metrics(test_pred, test_gt.to(opt.device))
            
            cum_tp += tp
            cum_fp += fp
            cum_fn += fn

            gt_lengths = check_and_find_length(test_gt) # [...] w/ length B

            # keep track of model output #######################################
            for i in range(test_pred.shape[0]):
                ori_filename = test_data['ori_filename'][i][:-4]
                if ori_filename not in model_output:
                    model_output[ori_filename] = {
                        'start_t_idx': [], # [0, 1024, ...]
                        'end_t_idx': [], # [1024, 2048, ...]
                        'output': [], # (windows)
                        'actual_len': [] # w/ length batch_size
                    }
    
                model_output[ori_filename]['start_t_idx'].append(
                    test_data['start_t_idx'][i].item()  
                )
                model_output[ori_filename]['end_t_idx'].append(
                    test_data['end_t_idx'][i].item()
                )
                model_output[ori_filename]['output'].append(
                    test_pred[i, :, :].squeeze().cpu()
                )
                model_output[ori_filename]['actual_len'].append(gt_lengths[i])

                # print(model_output.keys())
                # print(model_output[ori_filename]['start_t_idx'])
                # print(model_output[ori_filename]['end_t_idx'])
                # print(model_output[ori_filename]['output'][0].shape)      

        prec = 0 if cum_tp == 0 else cum_tp / (cum_tp + cum_fp)
        recall = 0 if cum_tp == 0 else cum_tp / (cum_tp + cum_fn)
        f1 = 0 if prec == 0 or recall == 0 else 2 * (prec * recall) / (prec + recall)
    
    print('prec, recall, f1', prec, recall, f1)

    statistics = {'F1 score': round(f1, 4), 'precision': round(prec, 4), 'recall': round(recall, 4)}
    with open('submission/GuanRen_Statistics.csv', 'w', newline='') as csvfile:
        fieldnames = ['F1 score', 'precision','recall']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(statistics)
        print('Generate Statistics csv.')

    rearrange_output(model_output)

    for key, value in model_output.items():
        value['output'] = torch.cat(value['output']) # (N*window,)
        actual_len = sum(value['actual_len'])
        value['output'] = torch.round(value['output'][:actual_len])
        model_output[key] = value['output']

    dict_to_csv(model_output, 'submission/GuanRen_Model_Output.csv')
    print('Generate model output csv.')

def parse_opt():
    parser = argparse.ArgumentParser()
    
    # project information: names ===============================================
    parser.add_argument('--test_dpath', default='test_data', help='path of test data folder')
    parser.add_argument('--gt_dpath', default='gt', help='path of ground truth csv folder')
    parser.add_argument('--device', default='0', help='cuda id')
    parser.add_argument('--batch_size', type=int, default=1, help='cuda id')
    
    parser.add_argument('--enable_self_test', action='store_true')

    opt = parser.parse_args()
    opt.device = f"cuda:{opt.device}"
    if opt.enable_self_test:
        opt.test_dpath = 'self_test_data'
        opt.gt_dpath = 'self_gt'

    opt.preload_gpu = True
    opt.root_dpath = opt.test_dpath
    opt.window = WINDOW

    return opt

if __name__ == "__main__":
    assert torch.cuda.is_available(), "**** No available GPUs."
    opt = parse_opt()

    if opt.enable_self_test:
        process_csv_files(opt, "kaggle")
        generate_gt_csv(opt)
        self_test_generate_model_output_csv(opt)
    else:
        for model_name in DATASETS_FEATS_MODEL.keys():
            process_csv_files(opt, model_name)
        generate_gt_csv(opt)
        generate_model_output_csv(opt)


    
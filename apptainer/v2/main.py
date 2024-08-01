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

        assert tmp_gt.shape[0] == all_data[count]['lowerback_acc'].shape[0], "unmatched length"

        all_data[count]['ori_filename'] = filename
        all_data[count]['gt'] = tmp_gt

        count += 1

    joblib.dump(all_data, open(os.path.join(opt.test_dpath, f"all_kaggle.p"), 'wb'))
    test_data = split_by_window(opt, 'kaggle')

    joblib.dump(test_data, open(os.path.join(opt.test_dpath, 
                                            f"all_test_data_{model_name}_window{WINDOW}.p"), 'wb'))

def generate_gt_csv(opt):
    gt_file = os.listdir(opt.gt_dpath)[0]
    gt_path = os.path.join(opt.gt_dpath, gt_file)
    shutil.copy2(gt_path, 'submission/GuanRen_FoG_Ground_Truth_Data.csv')
    print('Generate ground truth csv.')
        
def generate_model_output_csv(opt):
    for model_name in DATASETS_FEATS_MODEL.keys():
        if opt.enable_self_test and (model_name == 'daphnet' or model_name == 'turn'):
            continue

        test_data = joblib.load(os.path.join(opt.test_dpath, 
                                             f"all_test_data_{model_name}_window{WINDOW}.p"))

        gt = torch.stack([test_data[i]['gt'] for i in range(len(test_data.keys()))])
        
        from codes.unet_v4 import UNet
        weights_path = f"codes/{model_name}.pt"
        model = UNet(channel=len(DATASETS_FEATS_MODEL[model_name])*3, 
                     feats=DATASETS_FEATS_MODEL[model_name])
        model = model.to(opt.device)
        weights = torch.load(weights_path, map_location=opt.device)['model']
        model.load_state_dict(weights)
        
        model.eval()
        avg_prec, avg_recall, avg_f1 = 0, 0, 0
        with torch.no_grad():
            for batch_idx in range(gt.shape[0]):
                test_input = {}
                for body_name in DATASETS_FEATS_MODEL[model_name]:
                    model_input = test_data[batch_idx][body_name].unsqueeze(0) # (1, window, 3)
                    test_input[body_name] = model_input[:,:,[2, 0, 1]].to(opt.device)
                test_pred = model(test_input) # (B, window, 1)
                prec, recall, f1 = evaluation_metrics(test_pred, gt.to(opt.device))
                avg_prec += prec
                avg_recall += recall
                avg_f1 += f1

                # print(test_pred.shape)
                # print(prec, recall, f1)
                # exit(0)
        total_num = gt.shape[0]
        print(avg_prec / total_num, avg_recall / total_num, avg_f1 / total_num)
        exit(0)
        torch.cuda.empty_cache()

        statistics = {'F1 score': f1.item(), 'precision': prec.item(), 'recall': recall.item()}
        with open('GuanRen_Statistics.csv', 'w', newline='') as csvfile:
            fieldnames = ['F1 score', 'precision','recall']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(statistics)
            print('Generate Statistics csv.')

        #*==============================================================================================
        pred = torch.round(test_pred)  # (B, window, 1)
        real = torch.argmax(gt[:, :, :2], dim=-1, keepdim=True)  # (B, window, 1)
        mask = (gt[:, :, 2] != 1).unsqueeze(-1)  # (B, window, 1)
        output = (pred * mask.float()).squeeze() # (B, window)

        grouped_data = {}
        for i in range(output.shape[0]):
            sample = test_data[i]
            trial_id = sample['trial_id']
            
            a = output[i].cpu().numpy()
            a[a == 0.0] = 0
            a[a == 1.0] = 1
            a[a == 2.0] = 2
            
            if trial_id not in grouped_data:
                grouped_data[trial_id] = a
            else:
                
                grouped_data[trial_id] = np.concatenate((grouped_data[trial_id], a))

        data = {f"ModelOutput_Trial{key}": value.astype(int).tolist() \
                for key, value in grouped_data.items()}

        max_len = max(len(v) for v in data.values())

        for key in data:
            data[key] += [None] * (max_len - len(data[key]))
            
        for key in data:
            for i in range(len(data[key])):
                if data[key][i] is not None:
                    data[key][i] = int(data[key][i])

        df = pd.DataFrame(data)

        # Save the DataFrame to a CSV file
        df.to_csv('GuanRen_FoG_Model_Output_Data.csv', index=False)

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
    weights = torch.load(weights_path, map_location=opt.device)['model']
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
                ori_filename = test_data['ori_filename'][i]
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
        value['output'] = torch.cat(value['output'])
        
        print(value['actual_len'])
        exit(0)

    exit(0)
    for model_name in DATASETS_FEATS_MODEL.keys():
        if opt.enable_self_test and (model_name == 'daphnet' or model_name == 'turn'):
            continue

        test_data = joblib.load(os.path.join(opt.test_dpath, 
                                             f"all_test_data_{model_name}_window{WINDOW}.p"))

        gt = torch.stack([test_data[i]['gt'] for i in range(len(test_data.keys()))])
        
        
        
        model.eval()
        avg_prec, avg_recall, avg_f1 = 0, 0, 0
        with torch.no_grad():
            for batch_idx in range(gt.shape[0]):
                test_input = {}
                for body_name in DATASETS_FEATS_MODEL[model_name]:
                    model_input = test_data[batch_idx][body_name].unsqueeze(0) # (1, window, 3)
                    test_input[body_name] = model_input[:,:,[2, 0, 1]].to(opt.device)
                test_pred = model(test_input) # (B, window, 1)
                prec, recall, f1 = evaluation_metrics(test_pred, gt.to(opt.device))
                avg_prec += prec
                avg_recall += recall
                avg_f1 += f1

                # print(test_pred.shape)
                # print(prec, recall, f1)
                # exit(0)
        total_num = gt.shape[0]
        print(avg_prec / total_num, avg_recall / total_num, avg_f1 / total_num)
        exit(0)
        torch.cuda.empty_cache()

        statistics = {'F1 score': f1.item(), 'precision': prec.item(), 'recall': recall.item()}
        with open('GuanRen_Statistics.csv', 'w', newline='') as csvfile:
            fieldnames = ['F1 score', 'precision','recall']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(statistics)
            print('Generate Statistics csv.')

        #*==============================================================================================
        pred = torch.round(test_pred)  # (B, window, 1)
        real = torch.argmax(gt[:, :, :2], dim=-1, keepdim=True)  # (B, window, 1)
        mask = (gt[:, :, 2] != 1).unsqueeze(-1)  # (B, window, 1)
        output = (pred * mask.float()).squeeze() # (B, window)

        grouped_data = {}
        for i in range(output.shape[0]):
            sample = test_data[i]
            trial_id = sample['trial_id']
            
            a = output[i].cpu().numpy()
            a[a == 0.0] = 0
            a[a == 1.0] = 1
            a[a == 2.0] = 2
            
            if trial_id not in grouped_data:
                grouped_data[trial_id] = a
            else:
                
                grouped_data[trial_id] = np.concatenate((grouped_data[trial_id], a))

        data = {f"ModelOutput_Trial{key}": value.astype(int).tolist() \
                for key, value in grouped_data.items()}

        max_len = max(len(v) for v in data.values())

        for key in data:
            data[key] += [None] * (max_len - len(data[key]))
            
        for key in data:
            for i in range(len(data[key])):
                if data[key][i] is not None:
                    data[key][i] = int(data[key][i])

        df = pd.DataFrame(data)

        # Save the DataFrame to a CSV file
        df.to_csv('GuanRen_FoG_Model_Output_Data.csv', index=False)


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
        # process_csv_files(opt, "kaggle")
        # generate_gt_csv(opt)
        self_test_generate_model_output_csv(opt)
    else:
        for model_name in DATASETS_FEATS_MODEL.keys():
            process_csv_files(opt, model_name)
        generate_gt_csv(opt)
        generate_model_output_csv(opt)


    
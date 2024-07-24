import argparse, csv, joblib, math, os, torch
from collections import OrderedDict

import pandas as pd
import numpy as np
from tqdm import tqdm


FEATURES = ['LowerBack_Acc_X', 'LowerBack_Acc_Y', 'LowerBack_Acc_Z']
WINDOW = 6976


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

def split_by_window(opt):
    all_data = joblib.load(os.path.join(opt.test_dpath, "all_test_data.p"))

    # split by window
    test_data = {}
    count = 0
    for series_name, series_info in all_data.items():
        series_len = len(series_info['gt'])
        concate_feat = []
        for feat in FEATURES:
            concate_feat.append(series_info[feat][:,None])
        
        concate_feat = torch.cat(concate_feat, dim=1) # (series_len, num_feats)
        padding_len = math.ceil(series_len / WINDOW) * WINDOW - series_len
        pad_feat = torch.zeros(padding_len, concate_feat.shape[1], device=concate_feat.device)
        pad_gt = torch.ones(padding_len, dtype=torch.int8, 
                                device=series_info['gt'].device) * 2
        # (T', num_feats)  e.g. (31104, 9)
        concate_feat = torch.cat([concate_feat, pad_feat], dim=0)
        # (T',)
        concate_gt = torch.cat([series_info['gt'], pad_gt], dim=0)

        for i in range(0, concate_feat.shape[0], WINDOW):
            start_t_idx = int(i)
            end_t_idx = int(i + WINDOW)
            
            assert end_t_idx <= concate_feat.shape[0], "length unmatched"
        
            # (window, num_feats)
            model_input = concate_feat[start_t_idx:end_t_idx].detach().clone()
            model_input = model_input.to(dtype=torch.float32)

            # (window,)
            gt = concate_gt[start_t_idx:end_t_idx].detach().clone() 
            
            gt = torch.nn.functional.one_hot(gt.to(torch.int64), num_classes=3)
            
            assert is_valid_one_hot(gt), "not valid one hot encoding"
            
            # (window,3)
            gt = gt.float()

            added_window = {
                'series_name': series_name,  #                              str
                'trial_id': series_info['trial_id'], #           str
                'start_t_idx': start_t_idx, # 0,     15552, ...               int
                'end_t_idx': end_t_idx,     # 15552, 15552+15552, ...         int
                'model_input': model_input.cpu(), # (window, num_feats)  torch
                'gt': gt.cpu(), # (window, 3) one-hot                      torch
            }
            
            test_data[count] = added_window
            count += 1
    return test_data

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
    tp = ((pred == 1) & (real == 1)).float().sum()
    fp = ((pred == 1) & (real == 0)).float().sum()
    fn = ((pred == 0) & (real == 1)).float().sum()

    # Calculate precision, recall, and F1 score
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    return precision, recall, f1

def process_csv_files(opt):
    test_csv_files = os.listdir(opt.test_dpath)

    all_test_data = {}
    count = 0

    gt_file = os.listdir(opt.gt_dpath)
    assert len(gt_file) == 1, "too many ground truth files, suppose to have only one file"
    gt_data = pd.read_csv(os.path.join(opt.gt_dpath, gt_file[0]))

    for filename in tqdm(test_csv_files):
        if not filename.endswith('.csv'):
            continue
        
        df_filtered = pd.read_csv(os.path.join(opt.test_dpath, filename))

        trial_id = filename[5:-4] 

        if opt.enable_self_test:
            df_filtered.rename(columns={
                'AccV': 'LowerBack_Acc_Z',
                'AccML': 'LowerBack_Acc_Y',
                'AccAP': 'LowerBack_Acc_X',
            }, inplace=True)
            # df_filtered['Event'] = df_filtered.apply(lambda row: 1 
            #                                          if row[['StartHesitation', 
            #                                                  'Turn', 'Walking']].any() 
            #                                          else 0, axis=1)
        
        desired_order = ['LowerBack_Acc_X', 'LowerBack_Acc_Y', 'LowerBack_Acc_Z']
        df_filtered = df_filtered[desired_order]
        
        all_test_data[count] = {
            'ori_filename': filename,
            'trial_id': trial_id,
            'gt': torch.tensor(gt_data[f'GroundTruth_Trial{trial_id}'].dropna(), 
                               dtype=torch.float32),
            'LowerBack_Acc_X': sample_normalize(torch.tensor(df_filtered['LowerBack_Acc_X'], 
                                                             dtype=torch.float32)),
            'LowerBack_Acc_Y': sample_normalize(torch.tensor(df_filtered['LowerBack_Acc_Y'], 
                                                             dtype=torch.float32)),
            'LowerBack_Acc_Z': sample_normalize(torch.tensor(df_filtered['LowerBack_Acc_Z'], 
                                                             dtype=torch.float32))
        }
        count += 1
        # print(f"{trial_id}: {all_test_data[count-1]['gt'].shape}")
    joblib.dump(all_test_data, open(os.path.join(opt.test_dpath, f"all_test_data.p"), 'wb'))

    test_data = split_by_window(opt)
    joblib.dump(test_data, open(os.path.join(opt.test_dpath, 
                                                 f"test_data_window{WINDOW}.p"), 'wb'))

def generate_gt_csv(opt):
    test_data = joblib.load(os.path.join(opt.test_dpath, "all_test_data.p"))

    # Group samples by 'ori_series_name'
    grouped_data = {}

    for i in range(len(test_data.keys())):
        sample = test_data[i]
        trial_id = sample['trial_id']
        gt_label = sample['gt']
        
        if trial_id not in grouped_data:
            grouped_data[trial_id] = gt_label.cpu().numpy().astype(int)
        else:
            grouped_data[trial_id] = np.concatenate((grouped_data[trial_id], 
                                                            gt_label.cpu().numpy().astype(int)))

    # print(len(grouped_data.keys()))
    # for key, value in grouped_data.items():
    #     print(f"sample: {key}, length: {value.size}")


    data = {f"GroundTruth_Trial{key}": value.astype(int).tolist() \
            for key, value in grouped_data.items()}

    max_len = max(len(v) for v in data.values())

    for key in data:
        data[key] += [None] * (max_len - len(data[key]))

    df = pd.DataFrame(data)

    df.to_csv('GuanRen_FoG_Ground_Truth_Data.csv', index=False)
    print('Generate ground truth csv.')
        
def generate_model_output_csv(opt):
    test_data = joblib.load(os.path.join(opt.test_dpath, f"test_data_window{WINDOW}.p"))
    model_input = torch.stack([test_data[i]['model_input'] for i in range(len(test_data.keys()))])
    gt = torch.stack([test_data[i]['gt'] for i in range(len(test_data.keys()))])
    model_input = model_input.to(opt.device)
    gt = gt.to(opt.device)
    
    from codes.unet_v3 import UNet
    weights_path = "codes/best_model_f1_3150.pt"
    model = UNet(3)
    model = model.to(opt.device)
    weights = torch.load(weights_path, map_location=opt.device)['model']
    model.load_state_dict(weights)
    
    model.eval()
    with torch.no_grad():
        test_input = torch.permute(model_input, (0,2,1)) # (B, num_feats, window)
        test_pred = model(test_input) # (B, 1, window)
        test_pred = torch.permute(test_pred, (0, 2, 1)) # (B, window, 1)
        prec, recall, f1 = evaluation_metrics(test_pred, gt)

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
    
    parser.add_argument('--enable_self_test', action='store_true')

    opt = parser.parse_args()
    opt.device = f"cuda:{opt.device}"
    return opt

if __name__ == "__main__":
    assert torch.cuda.is_available(), "**** No available GPUs."
    opt = parse_opt()
    process_csv_files(opt)
    generate_gt_csv(opt)
    generate_model_output_csv(opt)


    
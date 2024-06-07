'''
Process all datasets to match the CSV data format in phase 2 of FoG challenge.
https://www.synapse.org/Synapse:syn52540892/wiki/623753
Sample data is reference_data/sample_pfda/ValidationDataset-sampledata.csv
'''

import argparse, csv, joblib, os, random

import torch
import numpy as np
import pandas as pd
from scipy.constants import physical_constants
from tqdm import tqdm

from utils import config


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

def make_single_data(dpath, dataset_name):
    csv_files = os.listdir(dpath)
    single_csv = {}
    for csv_file in tqdm(csv_files, total=len(csv_files), desc="Make a single file"):
        if not csv_file.startswith('rectified') or not csv_file.endswith('.csv'):
            continue
        series = pd.read_csv(os.path.join(dpath, csv_file))
        series = series[config.FEATURES_LIST]
        
        filename = csv_file[:-4]
        
        single_csv[filename] = {}
        gt_csv = pd.read_csv(os.path.join(dpath, f"gt_{dataset_name}.csv"))
        gt_csv = gt_csv.fillna(2)
        csv_id = csv_file.split('_')[1]
        actual_data_len = len(series[config.FEATURES_LIST[0]].values)
        gt_data = torch.tensor(gt_csv[f"GroundTruth_Trial{csv_id}"].values[:actual_data_len], 
                               dtype=torch.int8)
        single_csv[filename]['gt'] = gt_data
        for feat in config.FEATURES_LIST:
            single_csv[filename][feat] = sample_normalize(series[feat].values)
        joblib.dump(single_csv, open(os.path.join(dpath, f"all_{dataset_name}.p"), 'wb'))

def process_dataset_fog_release(dpath=None):
    print('---- Processing dataset_fog_release')
    dpath = 'data/dataset_fog_release/dataset' if dpath is None else dpath
    files_list = sorted(os.listdir(dpath))
    
    csv_path = os.path.join('data', 'rectified_data', 'dataset_fog_release')
    
    csv_count = 1
    
    trial_gt = []
    max_time_series = 0
    
    for filename in tqdm(files_list):
        if not filename.endswith('.txt'):
            continue
        file_path = os.path.join(dpath, filename)
        
        cur_trial_gt = [] # [ ['GroundTruth_Trial1'], [0], [1], [1], ... ]
        merged_lines = []
        
        not_start_experiment = True
        
        with open(file_path, 'r') as f:
            cutoff_line = 0

            cur_trial_gt.append(f"GroundTruth_Trial{csv_count}")
            
            while True:
                line = f.readline()
                
                if not line:
                    break

                line = np.array(list(map(float, line.strip().split())))                
                line[:-1] /= 1000
                
                # make data that was not in experiment all zero
                annotation = int(line[-1])
                
                if not_start_experiment:
                    if annotation == 0:
                        continue
                    else:
                        not_start_experiment = False
                
                # g --> m/s^2
                line[1:-1] *= physical_constants['standard acceleration of gravity'][0]
                
                if annotation == 0:
                    annotation = ''  # out of experiment
                    cutoff_line += 1
                elif annotation == 1:
                    annotation = 0  # no freeze
                    cutoff_line = 0
                else: 
                    annotation = 1  # freeze
                    cutoff_line = 0
                    
                cur_trial_gt.append(annotation)
            
                merged_lines.append({
                    'Time': str(line[0]) + ' sec',
                    'GeneralEvent': 'Walk',
                    'ClinicalEvent': 'unlabeled',
                    'L_Ankle_Acc_X': str(line[1]),
                    'L_Ankle_Acc_Y': str(line[3]),
                    'L_Ankle_Acc_Z': str(line[2]),
                    'L_MidLatThigh_Acc_X': str(line[4]),
                    'L_MidLatThigh_Acc_Y': str(line[6]),
                    'L_MidLatThigh_Acc_Z': str(line[5]),
                    'LowerBack_Acc_X': str(line[7]),
                    'LowerBack_Acc_Y': str(line[9]),
                    'LowerBack_Acc_Z': str(line[8]),
                })
            
        # cut off post data that are out of experiment
        merged_lines = merged_lines[:-cutoff_line]
        cur_trial_gt = cur_trial_gt[:-cutoff_line]            
                    
        # write to the training data     
        csv_file_path = os.path.join(csv_path, f"rectified_{csv_count}_dataset_fog_release.csv")           
        with open(csv_file_path, 'w') as rectified_csv:
            writer = csv.DictWriter(rectified_csv, fieldnames=config.PHASE2_DATA_HEAD)
            writer.writeheader()
            writer.writerows(merged_lines)
        
        if len(cur_trial_gt) > max_time_series:
            max_time_series = len(cur_trial_gt)
        else:
            while len(cur_trial_gt) < max_time_series:
                cur_trial_gt.append('')    
            
        trial_gt.append(cur_trial_gt)            
        
        csv_count += 1
    
    # pad and write to ground truth file
    for i in range(len(trial_gt)):
        while len(trial_gt[i]) < max_time_series:
            trial_gt[i].append('')
    gt_csv_file_path = os.path.join(csv_path, "gt_dataset_fog_release.csv")
    with open(gt_csv_file_path, 'w', newline='') as gt_csv:
        writer = csv.writer(gt_csv)
        transposed_data = list(zip(*trial_gt))
        for row in transposed_data:
            writer.writerow(row)

def process_kaggle_pd_data():
    print('---- Processing kaggle_pd_data')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, nargs='+', default=config.ALL_DATASETS, 
                                      help='Which datasets to process. By default process all.')
    opt = parser.parse_known_args()
    opt = opt[0]

    for dataset in opt.datasets:
        match dataset:
            case 'dataset_fog_release':
                # process_dataset_fog_release()
                # make_single_data(dpath='data/rectified_data/dataset_fog_release', 
                #                  dataset_name="dataset_fog_release")
                # split_train_val(dpath='data/rectified_data/dataset_fog_release', 
                #                 dataset_name="dataset_fog_release")
                pass
            case 'kaggle_pd_data':
                # process_kaggle_pd_data()
                pass
            case _:
                print(f"**** {dataset} dataset doesn\'t exist")
         
                
        

'''
Process all datasets to match the CSV data format in phase 2 of FoG challenge.
https://www.synapse.org/Synapse:syn52540892/wiki/623753
Sample data is reference_data/sample_pfda/ValidationDataset-sampledata.csv
'''

import argparse, csv, os

import numpy as np
from scipy.constants import physical_constants
from tqdm import tqdm

from utils import config


def process_dataset_fog_release(dpath=None):
    print('---- Processing dataset_fog_release')
    dpath = 'data/dataset_fog_release/dataset' if dpath is None else dpath
    files_list = sorted(os.listdir(dpath))
    csv_path = 'data/rectified_dataset_fog_release.csv'
    
    data_folder = os.listdir('data')
    rectified_csv_files = [file for file in data_folder if file.startswith('rectified') \
                                                           and file.endswith('.csv')]
    rectified_csv_files = sorted(rectified_csv_files, key=lambda x: (int(x[:-4].split('_')[1])))
    if len(rectified_csv_files) > 0:
        csv_count = rectified_csv_files[-1][:-4].split('_')[1]
        csv_count += 1
    else:
        csv_count = 0
    
    trial_gt = []
    max_time_series = 0
    
    for filename in tqdm(files_list):
        if not filename.endswith('.txt'):
            continue
        file_path = os.path.join(dpath, filename)
        
        cur_trial_gt = []
        merged_lines = []
        with open(file_path, 'r') as f:
            line_count = 0
            count_freeze = 0
            
            while True:
                line = f.readline()
                if not line:
                    while line_count % config.ONE_TIME_TRIAL != 0:
                        # pad the csv with all zero for an incomplete trial
                        merged_lines.append({
                            'Time': '0 sec',
                            'GeneralEvent': 'unlabeled',
                            'ClinicalEvent': 'unlabeled',
                            'L_Ankle_Acc_X': 0,
                            'L_Ankle_Acc_Y': 0,
                            'L_Ankle_Acc_Z': 0,
                            'L_MidLatThigh_Acc_X': 0,
                            'L_MidLatThigh_Acc_Y': 0,
                            'L_MidLatThigh_Acc_Z': 0,
                            'LowerBack_Acc_X': 0,
                            'LowerBack_Acc_Y': 0,
                            'LowerBack_Acc_Z': 0,
                        })
                        line_count += 1
                        
                    binary_label = 1 if count_freeze / config.ONE_TIME_TRIAL >= 0.5 else 0
                    cur_trial_gt.append(binary_label)
                    break
                
                line_count += 1
                
                line = np.array(list(map(float, line.strip().split())))                
                line[:-1] /= 1000
                
                # make data that was not in experiment all zero
                annotation = int(line[-1])
                if annotation == 0:
                    line[1:-1] = 0
                else:
                    # g --> m/s^2
                    line[1:-1] *= physical_constants['standard acceleration of gravity'][0]
            
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
                
                if line[-1] == 2: count_freeze += 1
                    
                if line_count % config.ONE_TIME_TRIAL == 0:
                    binary_label = 1 if count_freeze / config.ONE_TIME_TRIAL >= 0.5 else 0
                    cur_trial_gt.append(binary_label)
                    count_freeze = 0
                    
        # write to the training data                
        with open(f'data/rectified_data/recitifed_{csv_count}_dataset_fog_release.csv', 'w') as rectified_csv:
            writer = csv.DictWriter(rectified_csv, fieldnames=config.PHASE2_DATA_HEAD)
            writer.writeheader()
            writer.writerows(merged_lines)
        
        if len(cur_trial_gt) > max_n_trial:
            max_n_trial = len(cur_trial_gt)
        else:
            while len(cur_trial_gt) < max_n_trial:
                cur_trial_gt.append('')    
            
        trial_gt.append(cur_trial_gt)            
        
        csv_count += 1
    
    # write to ground truth file
    with open(f'data/rectified_data/gt_dataset_fog_release.csv', 'w') as gt_csv:
        writer = csv.writer(gt_csv)
        header = [f'GroundTruth_Trial{i + 1}' for i in range(max_n_trial)]
        writer.writerow(header)
        
        for i in range(len(trial_gt)):
            while len(trial_gt[i]) < max_n_trial:
                trial_gt[i].append('')
            writer.writerow(trial_gt[i])

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
                # continue
                process_dataset_fog_release()
            case 'kaggle_data':
                continue
                process_kaggle_pd_data()
            case _:
                print('**** dataset doesn\'t exist')
         
                
        

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
    
    csv_path = os.path.join('data', 'rectified_data')
    
    data_folder = os.listdir(csv_path)
    rectified_csv_files = [file for file in data_folder if file.startswith('rectified') \
                                                           and file.endswith('.csv')]
    rectified_csv_files = sorted(rectified_csv_files, key=lambda x: (int(x[:-4].split('_')[1])))
    if len(rectified_csv_files) > 0:
        csv_count = rectified_csv_files[-1][:-4].split('_')[1]
        csv_count += 1
    else:
        csv_count = 1
    
    trial_gt = []
    max_time_series = 0
    
    for filename in tqdm(files_list):
        if not filename.endswith('.txt'):
            continue
        file_path = os.path.join(dpath, filename)
        
        cur_trial_gt = [] # [ ['GroundTruth_Trial1'], [0], [1], [1], ... ]
        merged_lines = []
        with open(file_path, 'r') as f:
            line_count = 0

            cur_trial_gt.append(f"GroundTruth_Trial{csv_count}")
            
            while True:
                line = f.readline()
                
                if not line:
                    break
                
                line_count += 1
                line = np.array(list(map(float, line.strip().split())))                
                line[:-1] /= 1000
                
                # make data that was not in experiment all zero
                annotation = int(line[-1])
                if annotation == 0:
                    line[1:-1] = 0.0
                else:
                    # g --> m/s^2
                    line[1:-1] *= physical_constants['standard acceleration of gravity'][0]
                
                annotation = 1 if annotation == 2 else 0
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
                    
        # write to the training data     
        csv_file_path = os.path.join(csv_path, f"rectifed_{csv_count}_dataset_fog_release.csv")           
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
    
    # write to ground truth file
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
                # continue
                process_dataset_fog_release()
            case 'kaggle_pd_data':
                continue
                process_kaggle_pd_data()
            case _:
                print(f"**** {dataset} dataset doesn\'t exist")
         
                
        

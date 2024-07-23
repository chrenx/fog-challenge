import argparse, joblib, os, torch

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


def main():
    pass

def parse_opt():
    parser = argparse.ArgumentParser()
    
    # project information: names ===============================================
    parser.add_argument('--test_dpath', default='test_data', help='path of test data folder')
    parser.add_argument('--gt_dpath', default='gt', help='path of ground truth csv folder')
    parser.add_argument('--device', default='0', help='cuda id')
    opt = parser.parse_args()
    opt.device = f"cuda:{opt.device}"
    return opt

if __name__ == "__main__":
    assert torch.cuda.is_available(), "**** No available GPUs."
    opt = parse_opt()
    process_csv_files(opt)
    # main()

    
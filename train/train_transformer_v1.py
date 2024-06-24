import argparse, glob, logging, os, random, shutil
from datetime import datetime

import torch, wandb
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
from torch.optim import Adam, AdamW
from torch.utils import data

from data.fog_dataset_v2 import FoGDataset
from models.transformer_v1 import Transformer
from tqdm import tqdm
from utils.config import ALL_DATASETS, FEATURES_LIST
from utils.train_util_v2 import cycle_dataloader, save_group_args


MYLOGGER = logging.getLogger()


class Trainer(object):
    def __init__(self, model, opt):
        super().__init__()
        
        self.use_wandb = not opt.disable_wandb
        self.save_best_model = opt.save_best_model
        self.weights_dir = opt.weights_dir
        self.warmup_steps = opt.lr_scheduler_warmup_steps
        self.penalty_cost = opt.penalty_cost
        
        if self.use_wandb:
            MYLOGGER.info("Initialize W&B")
            wandb.init(config=opt, project=opt.wandb_pj_name, entity=opt.entity, 
                       name=opt.exp_name, dir=opt.save_dir)
        
        #* NEW: self.train_dl, self.train_n_batch, self.val_dl, self.val_n_batch
        self._prepare_dataloader(opt) 
        
        self.model = model
        self.optimizer = None
        self.cur_step = 0
        self.train_num_steps = opt.train_num_steps
        self.device = opt.device
        self.bce_loss = torch.nn.BCELoss(reduction='none')
        self.opt = opt
        
        if opt.optimizer == 'adam':
            self.optimizer = Adam(self.model.parameters(), lr=opt.learning_rate, 
                                  betas=opt.adam_betas, eps=opt.adam_eps, 
                                  weight_decay=opt.weight_decay)
        elif opt.optimizer == 'adamw':
            self.optimizer = AdamW(self.model.parameters(), lr=opt.learning_rate,
                                   betas=opt.adam_betas, eps=opt.adam_eps, 
                                   weight_decay=opt.weight_decay)
        assert self.optimizer is not None, "Error: optimizer is not given."
        
        self.scheduler_optim = torch.optim.lr_scheduler.ReduceLROnPlateau(
                                                optimizer=self.optimizer,
                                                factor=opt.lr_scheduler_factor,
                                                patience=opt.lr_scheduler_patience)

        MYLOGGER.info(f"---- train n batch: {self.train_n_batch}")
        MYLOGGER.info(f"---- val n batch: {self.val_n_batch}")
        
        # tmp = next(iter(self.train_dl))
        # print(tmp.keys())
        # print(tmp['model_input'].device)
        # print(tmp['model_input'].shape)
        # print(tmp['gt'].device)
        # print(tmp['gt'].shape)
        # print(type(tmp['series_name']))
        # print(type(tmp['start_t_idx']))
        # exit(0)
        
    def _prepare_dataloader(self, opt):
        MYLOGGER.info("Loading training data ...")
        
        self.train_ds = FoGDataset(opt, mode='train')
        
        self.val_ds = FoGDataset(opt, mode='val')
        
        dl = data.DataLoader(self.train_ds, 
                            batch_size=opt.batch_size, 
                            shuffle=True, 
                            pin_memory=False, 
                            num_workers=0)
        self.train_n_batch = len(dl) 
        self.train_dl = cycle_dataloader(dl)
        
        dl = data.DataLoader(self.val_ds, 
                            batch_size=opt.batch_size, 
                            shuffle=False, 
                            pin_memory=False, 
                            num_workers=0)
        self.val_n_batch = len(dl)
        self.val_dl = cycle_dataloader(dl)
        
    def _save_model(self, step, base, best=False):
        
        data = {
            'step': step,
            'model': self.model.state_dict(),
        }
        # delete previous best* or model*
        if best: 
            search_pattern = os.path.join(self.weights_dir, f"best_model_{base}*")
        else:
            search_pattern = os.path.join(self.weights_dir, "model*")
        files_to_delete = glob.glob(search_pattern)

        for file_path in files_to_delete:
            try:
                os.remove(file_path)
            except OSError as e:
                MYLOGGER.error(f"Error deleting file {file_path}: {e}")
            
        filename = f"best_model_{base}_{step}.pt" if best else f"model_{base}_{step}.pt"
        torch.save(data, os.path.join(self.weights_dir, filename))      
        
    def _loss_func(self, pred, gt):
        """Compute the Binary Cross-Entropy loss for each class and sum over the class dimension

        Args:
            pred: (B, BLKS//P, 2) prob
            gt: (B, BLKS//P, 3) one hot
        """
        loss = self.bce_loss(pred, gt[:,:,:2]) # (B, BLKS//P, 2)
        mask = (gt[:,:,2] != 1).float() # (B, BLKS//P)
        mask = mask.unsqueeze(-1).expand(-1, -1, 2) # (B, BLKS//P, 2)
        
        # Additional cost for misclassifying the minority class
        minority_mask = (gt[:,:,1] == 1).float() # (B, BLKS//P)
        minority_mask = minority_mask.unsqueeze(-1).expand(-1, -1, 2) # (B, BLKS//P, 2)
        loss = loss * (mask + self.penalty_cost * minority_mask)
   
        return loss.sum() / mask.sum()

    def _evaluation_metrics(self, output, gt):
        """Generate precision, recall, and f1 score.

        Args:
            output: (B, BLKS, 2)   # prob class
            gt (inference):   (B, BLKS, 3)   # one hot
        """
        # Convert the model output probabilities to class predictions
        pred = torch.argmax(output, dim=-1)  # (B, BLKS//P)

        # Extract the first two classes from the ground truth
        real = torch.argmax(gt[:, :, :2], dim=-1)  # (B, BLKS//P)

        # Create a mask to ignore the positions where the ground truth class is 2
        mask = (gt[:, :, 2] != 1)  # (B, BLKS//P)

        # Apply the mask to the predictions and ground truth
        pred = pred * mask.long() # (B, BLKS//P)
        real = real * mask.long() # (B, BLKS//P)

        # Calculate true positives, false positives, and false negatives
        tp = ((pred == 1) & (real == 1)).float().sum()
        fp = ((pred == 1) & (real == 0)).float().sum()
        fn = ((pred == 0) & (real == 1)).float().sum()

        # Calculate precision, recall, and F1 score
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        # Calculate mean average precision (mAP)
        # Create a list to store the precision values at different thresholds
        precisions = []
        recalls = []
        
        # Iterate over thresholds from 0 to 1 with a step size
        thresholds = torch.linspace(0, 1, steps=101)
        for t in thresholds:
            # Binarize predictions based on the threshold
            pred_binary = (output[:, :, 1] >= t).float()
            
            # Apply the mask to the binary predictions and ground truth
            pred_binary = pred_binary * mask # (B, BLKS//P)
            real_binary = real * mask        
            
            # Calculate true positives, false positives, and false negatives
            tp_t = ((pred_binary == 1) & (real_binary == 1)).float().sum()
            fp_t = ((pred_binary == 1) & (real_binary == 0)).float().sum()
            fn_t = ((pred_binary == 0) & (real_binary == 1)).float().sum()

            # Calculate precision and recall for the current threshold
            precision_t = tp_t / (tp_t + fp_t + 1e-8)
            recall_t = tp_t / (tp_t + fn_t + 1e-8)
            
            precisions.append(precision_t)
            recalls.append(recall_t)
        
        # Convert lists to tensors
        precisions = torch.tensor(precisions)
        recalls = torch.tensor(recalls)
        
        # Sort by recall
        recall_sorted, indices = torch.sort(recalls)
        precision_sorted = precisions[indices]
        
        # Calculate average precision
        ap = torch.trapz(precision_sorted, recall_sorted)
        
        return precision, recall, f1, ap          

    def train(self):
        best_f1 = 0
        best_loss = float('inf')
        best_mAP = 0
        for step_idx in tqdm(range(0, self.train_num_steps), desc="Train"):
            self.model.train()
            self.optimizer.zero_grad()
            
            #* training part -----------------------------------------------------------------------
            train_data = next(self.train_dl)
            train_input = train_data['model_input'] # (B, BLKS//P, P*num_feats)
            train_gt = train_data['gt'] # (B, BLKS//P, 3)
            train_pred = self.model(train_input) # (B, BLKS//P, 2)
            
            train_loss = self._loss_func(train_pred, train_gt.to(self.device))
            
            train_loss.backward()
            
            # check gradients
            parameters = [p for p in self.model.parameters() if p.grad is not None]
            total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2.0).\
                                        to(self.device) for p in parameters]), 2.0)

            if torch.isnan(total_norm):
                MYLOGGER.warning('NaN gradients. Skipping to next data...')
                torch.cuda.empty_cache()
                continue
            
            self.optimizer.step()

            if self.use_wandb:
                log_dict = {
                    "Train/loss": train_loss.item(),
                }
                wandb.log(log_dict, step=step_idx+1)
                
            #* validation part ---------------------------------------------------------------------
            cur_epoch = (step_idx + 1) // self.train_n_batch
            if (step_idx + 1) % self.train_n_batch == 0: #* an epoch
            # if True:
                avg_val_f1 = 0.0
                avg_val_loss = 0.0
                avg_val_prec = 0.0
                avg_val_recall = 0.0
                avg_val_mAP = 0.0
                # all_recalls, all_precisions = [], []
                all_val_gt, all_val_pred = [], []
                
                self.model.eval()
                with torch.no_grad():
                    for _ in tqdm(range(self.val_n_batch), desc=f"Validation at epoch {cur_epoch}"):
                        val_data = next(self.val_dl) 
                        val_input = val_data['model_input']
                        val_inference_gt = val_data['inference_gt'] # (B, BLKS, 3)
                        
                        val_pred = self.model(val_input, training=False) # (B, BLKS//P, 2)
                        
                        val_pred = val_pred.unsqueeze(-1) # (B, BLKS//P, 2, 1)
                        val_pred = val_pred.permute(0,1,3,2) # (B, BLKS//P, 1, 2)
                        
                        # (B, BLKS//P, P, 2)
                        val_pred = val_pred.repeat(1, 1, self.opt.patch_size, 1)
                        # (B, BLKS, 2)
                        val_pred = val_pred.reshape(val_pred.shape[0], self.opt.block_size, 2)
                        
                        assert val_pred.shape == val_inference_gt[:,:,:-1].shape, \
                               f"val_pred {val_pred.shape} and val_inference_gt "\
                               f"{val_inference_gt.shape} have different shape"
                        
                        prec, recall, f1, ap = self._evaluation_metrics(val_pred, 
                                                                val_inference_gt.to(self.device))
                        val_loss = self._loss_func(val_pred, val_inference_gt.to(self.device))
                        
                        avg_val_f1 += f1
                        avg_val_loss += val_loss
                        avg_val_prec += prec
                        avg_val_recall += recall
                        avg_val_mAP += ap
                        
                        # all_recalls.append(recall.item())
                        # all_precisions.append(prec.item())
                        
                        tmp_gt = val_inference_gt.reshape(-1, 3)
                        mask = tmp_gt[:,2] != 1
                        tmp_gt = tmp_gt[mask] # (B*BLKS//P',3)
                        tmp_pred = val_pred.reshape(-1, 2)[mask] # (B*BLKS//P',2)
                        all_val_gt.append(tmp_gt.cpu().numpy()) # (B*BLKS//P',3)
                        all_val_pred.append(tmp_pred.cpu().numpy()) # (B*BLKS//P',2)
                        
                    avg_val_f1 /= self.val_n_batch
                    avg_val_loss /= self.val_n_batch
                    avg_val_prec /= self.val_n_batch
                    avg_val_recall /= self.val_n_batch
                    avg_val_mAP /= self.val_n_batch
                    
                    # pr_auc = auc(all_recalls, all_precisions)
                    
                    all_val_gt = np.concatenate(all_val_gt, axis=0)  # (B*BLKS//P',3)
                    all_val_pred = np.concatenate(all_val_pred, axis=0)  # (B*BLKS//P',2)
                    
                    if self.use_wandb:
                        log_dict = {
                            "Val/avg_val_loss": avg_val_loss.item(),
                            "Val/avg_val_f1": avg_val_f1.item(),
                            "Val/avg_val_prec": avg_val_prec.item(),
                            "Val/avg_val_recall": avg_val_recall.item(),
                            "Val/avg_val_mAP": avg_val_mAP.item(),
                            # "Val/pr_auc": pr_auc,
                        }
                        wandb.log(log_dict, step=step_idx+1)
                        wandb.log({'precision_recall_curve': 
                                        wandb.plot.pr_curve(y_true=all_val_gt[:, 1].astype(int),
                                                            y_probas=all_val_pred,
                                                            labels=['normal', 'freeze'])},
                                  step=step_idx+1)
                    MYLOGGER.info(f"avg_val_loss: {avg_val_loss.item():4f}")
                    MYLOGGER.info(f"avg_val_f1: {avg_val_f1.item():4f}")
                    MYLOGGER.info(f"avg_val_prec: {avg_val_prec.item():4f}")
                    MYLOGGER.info(f"avg_val_recall: {avg_val_recall.item():4f}")
                    MYLOGGER.info(f"avg_val_mAP: {avg_val_mAP.item():4f}")
                    # MYLOGGER.info(f"pr_auc: {pr_auc.item():4f}")
                
                # Log learning rate
                if self.use_wandb:
                    wandb.log({'learning_rate': self.scheduler_optim.get_last_lr()[0]}, 
                              step=step_idx+1)
                
                if self.save_best_model and avg_val_f1 > best_f1:
                    best_f1 = avg_val_f1
                    wandb.run.summary['best_f1'] = avg_val_f1.item()
                    self._save_model(step_idx, base='f1', best=True)
                    
                if self.save_best_model and avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    wandb.run.summary['best_loss'] = avg_val_loss.item()
                    self._save_model(step_idx, base='loss', best=True)
                    
                if self.save_best_model and avg_val_mAP > best_mAP:
                    best_mAP = avg_val_mAP
                    if self.use_wandb:
                        wandb.run.summary['best_mAP'] = avg_val_mAP.item()
                    self._save_model(step_idx, base='mAP', best=True)
                    
                self._save_model(step_idx, base='regular', best=False)
            
                #* learning rate scheduler ---------------------------------------------------------
                if cur_epoch > self.warmup_steps:    
                    self.scheduler_optim.step(avg_val_loss)


def run_train(opt):
    model = Transformer(opt)
    model.to(opt.device)
    trainer = Trainer(model, opt)
    trainer.train()
    torch.cuda.empty_cache()

def parse_opt():
    parser = argparse.ArgumentParser()
    
    # project information: names ===============================================
    parser.add_argument('--version', type=int, default=None)
    parser.add_argument('--project', default='runs/train', help='project/name')
    parser.add_argument('--exp_name', default='transfomer_bilstm', 
                                            help='save to project/name')
    parser.add_argument('--cur_time', default=None, help='Time running this program')
    parser.add_argument('--description', type=str, default=None, help='important notes')

    # wandb setup ==============================================================
    parser.add_argument('--disable_wandb', action='store_true')
    parser.add_argument('--wandb_pj_name', type=str, default='fog-challenge', 
                                           help='wandb project name')
    parser.add_argument('--entity', default='', help='W&B entity or username')

    # data path
    parser.add_argument('--root_dpath', default='data/rectified_data', 
                                        help='directory that contains different processed datasets')
    parser.add_argument('--train_datasets', type=str, nargs='+', default=ALL_DATASETS, 
                                       help='provided dataset_name, e.g. kaggle, ...')
    
    # GPU ======================================================================
    parser.add_argument('--device', default='0', help='assign gpu')
    parser.add_argument('--device_info', type=str, default='')
    
    # training monitor =========================================================
    parser.add_argument('--save_best_model', action='store_true', 
                                                  help='save best model during training')
    # parser.add_argument('--save_every_n_epoch', type=int, default=50, 
    #                                               help='save model during training')

    # hyperparameters ==========================================================
    parser.add_argument('--seed', type=int, default=42, 
                                      help='set up seed for torch, numpy, random')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--optimizer', type=str, default="adam", 
                                       help="Choice includes [adam, adamw]")
    parser.add_argument('--learning_rate', type=float, default=26e-5, # 0.00026 
                                           help='generator_learning_rate')
    parser.add_argument('--adam_betas', default=(0.9, 0.98), help='betas for Adam optimizer')
    parser.add_argument('--adam_eps', default=1e-9, help='epsilon for Adam optimizer')
    parser.add_argument('--weight_decay', type=float, default=0, help='for adam optimizer')
    parser.add_argument('--lr_scheduler_factor', type=float, default=0.1, help='lr scheduler')
    parser.add_argument('--lr_scheduler_patience', type=int, default=20, help='for adam optimizer')
    parser.add_argument('--lr_scheduler_warmup_steps', type=int, default=64, help='lr scheduler')

    parser.add_argument('--train_num_steps', type=int, default=20000, 
                                                 help='number of training steps')
    parser.add_argument('--penalty_cost', type=float, default=0, 
                                          help='penalize when misclassifying the minor class(fog)')
    
    parser.add_argument('--random_aug', action='store_true', help="randomly augment data")
    parser.add_argument('--num_feats', type=int, default=len(FEATURES_LIST) - 1, 
                                                 help='number of features in raw data')
    
    parser.add_argument('--block_size', type=int, default=15552)
    parser.add_argument('--block_stride', type=int, default=15552 // 16) # 972
    parser.add_argument('--patch_size', type=int, default=18)

    #! may need to change if embed annotation
    # parser.add_argument('--fog_model_input_dim', type=int, default=18*(len(FEATURES_LIST)-1))
    parser.add_argument('--fog_model_input_dim', type=int, default=18*3)

    parser.add_argument('--fog_model_dim', type=int, default=320)
    parser.add_argument('--fog_model_num_heads', type=int, default=8)
    parser.add_argument('--fog_model_num_encoder_layers', type=int, default=5)
    parser.add_argument('--fog_model_num_lstm_layers', type=int, default=2)
    parser.add_argument('--fog_model_first_dropout', type=float, default=0.1)
    parser.add_argument('--fog_model_encoder_dropout', type=float, default=0.1)
    parser.add_argument('--fog_model_mha_dropout', type=float, default=0.0)
    parser.add_argument('--feats_list', type=str, nargs='+', default=FEATURES_LIST)
    
    # file tracker =============================================================
    parser.add_argument('--save_dir', type=str, default='')
    parser.add_argument('--weights_dir', type=str, default='')

    opt = parser.parse_args()
        
    return opt

if __name__ == "__main__":
    
    assert torch.cuda.is_available(), "**** No available GPUs."
    
    opt = parse_opt()
    
    assert opt.version is not None, "pass the version parameter"
    
    cur_time = datetime.now()
    cur_time = '{:%Y_%m_%d_%H:%M:%S}.{:02.0f}'.format(cur_time, 
                                                      cur_time.microsecond / 10000.0)

    opt.save_dir = os.path.join(opt.project, opt.exp_name, cur_time)
    opt.cur_time = cur_time

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    # Create dir
    os.makedirs(opt.save_dir, exist_ok=True)

    # Save some important code
    source_files = [f'train/train_transformer_bilstm_v{opt.version}.py', 
                    f'utils/train_util_v{opt.version}.py', 'utils/config.py',
                    f'data/fog_dataset_v{opt.version}.py',
                    f'models/transformer_bilstm_v{opt.version}.py']
    codes_dest = os.path.join(opt.save_dir, 'codes')
    os.makedirs(codes_dest)
    for file_dir in source_files:
        shutil.copy2(file_dir, codes_dest)
        
    # Create weight dir to store model weights
    opt.weights_dir = os.path.join(opt.save_dir, 'weights')
    os.makedirs(opt.weights_dir, exist_ok=True)
    
    training_info_log_path = os.path.join(opt.save_dir,"training_info.log")
    # redirect all printing to a file
    if not opt.disable_wandb:
        import sys
        sys.stdout = open(training_info_log_path, "w")
        logging.basicConfig(filename=os.path.join(training_info_log_path),
                        filemode='a',
                        format='%(asctime)s.%(msecs)02d %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d-%H:%M:%S',
                        level=os.environ.get("LOGLEVEL", "INFO"))
    else:
        logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
    
    if opt.device != 'cpu':
        opt.device_info = torch.cuda.get_device_name(int(opt.device)) 
        opt.device = f"cuda:{opt.device}"
    else:
        MYLOGGER.warning("!!!!!!!! Running on CPU.")

    MYLOGGER.setLevel(logging.INFO)
    MYLOGGER.info(f"Running at {cur_time}")
    MYLOGGER.info(f"Using device: {opt.device}")
    
    if opt.description is not None:
        MYLOGGER.info(f"-------- Job Description: --------\n{opt.description}")
    
    save_group_args(opt)

    run_train(opt)


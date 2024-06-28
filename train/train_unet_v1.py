import argparse, glob, logging, os, random, shutil
from datetime import datetime

import torch, wandb
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
from torch.optim import Adam, AdamW
from torch.utils import data

from data.fog_dataset import FoGDataset
from models.unet_v1 import UNet
from tqdm import tqdm
from utils.config import ALL_DATASETS, FEATURES_LIST
from utils.train_util import cycle_dataloader, save_group_args


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
        self.max_grad_norm = opt.max_grad_norm
        self.preload_gpu = opt.preload_gpu
        self.grad_accum_step = opt.grad_accum_step
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
            pred: (B, window, 1) prob
            gt: (B, window, 3) one hot
        """
        max_indices = torch.argmax(gt, dim=2, keepdim=True) # (B, window, 1)
        tmp_gt_mask = (max_indices != 2).float() # (B, window, 1)
        tmp_gt = max_indices * tmp_gt_mask # (B, window, 1)
        
        loss = self.bce_loss(pred, tmp_gt) # (B, window, 1)

        mask = (gt[:,:,2] != 1).float() # (B, window)
        mask = mask.unsqueeze(-1) # (B, window, 1)
        
        # Additional cost for misclassifying the minority class
        minority_mask = (gt[:,:,1] == 1).float() # (B, window)
        minority_mask = minority_mask.unsqueeze(-1) # (B, window, 1)
        loss = loss * (mask + self.penalty_cost * minority_mask)
   
        return loss.sum() / mask.sum()

    def _evaluation_metrics(self, output, gt):
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

    def train(self):
        best_f1 = 0
        best_prec = 0
        best_recall = 0
        best_loss = float('inf')

        for step_idx in tqdm(range(0, self.train_num_steps), desc="Train"):
            self.model.train()
            
            #* training part -----------------------------------------------------------------------
            train_data = next(self.train_dl)
            train_gt = train_data['gt'] # (B, window, 3) one-hot
            train_input = train_data['model_input'] # (B, window, num_feats)
            train_input = torch.permute(train_input, (0,2,1)) # (B, C_in, window)
            
            if not self.preload_gpu:
                train_input = train_input.to(self.device)

            train_pred = self.model(train_input) # (B,1,window)
            train_pred = torch.permute(train_pred, (0,2,1)) # (B, window, 1)

            train_loss = self._loss_func(train_pred, train_gt.to(self.device))
            train_loss /= self.grad_accum_step
            train_loss.backward()
            
            # check gradients
            parameters = [p for p in self.model.parameters() if p.grad is not None]
            total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2.0).\
                                        to(self.device) for p in parameters]), 2.0)
            nan_exist = False
            if torch.isnan(total_norm):
                MYLOGGER.warning('NaN gradients. Skipping to next data...')
                torch.cuda.empty_cache()
                nan_exist = True
            
            if (step_idx + 1) % self.grad_accum_step == 0:
                if nan_exist:
                    continue
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            if self.use_wandb:
                log_dict = {
                    "Train/loss": train_loss.item(),
                }
                wandb.log(log_dict, step=step_idx+1)
            
            if not self.preload_gpu:
                torch.cuda.empty_cache()
                
            #* validation part ---------------------------------------------------------------------
            cur_epoch = (step_idx + 1) // self.train_n_batch
            if (step_idx + 1) % self.train_n_batch == 0: #* an epoch
            # if True:
                avg_val_f1 = 0.0
                avg_val_loss = 0.0
                avg_val_prec = 0.0
                avg_val_recall = 0.0
                
                self.model.eval()
                with torch.no_grad():
                    for _ in tqdm(range(self.val_n_batch), desc=f"Validation at epoch {cur_epoch}"):
                        val_data = next(self.val_dl) 
                        val_gt = val_data['gt'] # (B, window, 3)
                        
                        val_input = val_data['model_input'] # (B, window, num_feats)
                        val_input = torch.permute(val_input, (0,2,1)) # (B, num_feats, window)
                        
                        if not self.preload_gpu:
                            val_input = val_input.to(self.device)
                        val_pred = self.model(val_input) # (B, 1, window)
                        val_pred = torch.permute(val_pred, (0, 2, 1)) # (B, window, 1)
                        
                        prec, recall, f1 = self._evaluation_metrics(val_pred, 
                                                                val_gt.to(self.device))
                        val_loss = self._loss_func(val_pred, val_gt.to(self.device))
                        
                        avg_val_f1 += f1
                        avg_val_loss += val_loss
                        avg_val_prec += prec
                        avg_val_recall += recall
                        
                    avg_val_f1 /= self.val_n_batch
                    avg_val_loss /= self.val_n_batch
                    avg_val_prec /= self.val_n_batch
                    avg_val_recall /= self.val_n_batch
                    
                    if self.use_wandb:
                        log_dict = {
                            "Val/avg_val_loss": avg_val_loss.item(),
                            "Val/avg_val_f1": avg_val_f1.item(),
                            "Val/avg_val_prec": avg_val_prec.item(),
                            "Val/avg_val_recall": avg_val_recall.item(),
                            # "Val/pr_auc": pr_auc,
                        }
                        wandb.log(log_dict, step=step_idx+1)
        
                    MYLOGGER.info(f"avg_val_loss: {avg_val_loss.item():4f}")
                    MYLOGGER.info(f"avg_val_f1: {avg_val_f1.item():4f}")
                    MYLOGGER.info(f"avg_val_prec: {avg_val_prec.item():4f}")
                    MYLOGGER.info(f"avg_val_recall: {avg_val_recall.item():4f}")
                
                # Log learning rate
                if self.use_wandb:
                    wandb.log({'learning_rate': self.scheduler_optim.get_last_lr()[0]}, 
                              step=step_idx+1)
                
                if self.save_best_model and avg_val_f1 > best_f1:
                    best_f1 = avg_val_f1
                    if self.use_wandb:
                        wandb.run.summary['best_f1'] = avg_val_f1.item()
                    self._save_model(step_idx, base='f1', best=True)
                    
                if self.save_best_model and avg_val_prec > best_prec:
                    best_prec = avg_val_prec
                    if self.use_wandb:
                        wandb.run.summary['best_f1'] = avg_val_prec.item()
                    self._save_model(step_idx, base='prec', best=True)
                    
                if self.save_best_model and avg_val_recall > best_recall:
                    best_recall = avg_val_recall
                    if self.use_wandb:
                        wandb.run.summary['best_f1'] = avg_val_recall.item()
                    self._save_model(step_idx, base='recall', best=True)
                    
                if self.save_best_model and avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    if self.use_wandb:
                        wandb.run.summary['best_loss'] = avg_val_loss.item()
                    self._save_model(step_idx, base='loss', best=True)
                    
                self._save_model(step_idx, base='regular', best=False)
            
                #* learning rate scheduler ---------------------------------------------------------
                if cur_epoch > self.warmup_steps:    
                    self.scheduler_optim.step(avg_val_loss)
                    
                if not self.preload_gpu:
                    torch.cuda.empty_cache()

        if self.use_wandb:
            wandb.run.finish()


def run_train(opt):
    model = UNet(len(opt.feats))
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
    parser.add_argument('--feats', type=str, nargs='+', default=FEATURES_LIST, 
                                                 help='number of features in raw data')
    
    parser.add_argument('--window', type=int, default=15552, help="-1 means using full trial") 

    parser.add_argument('--max_grad_norm', type=float, default=None, 
                                           help="prevent gradient explosion")

    parser.add_argument('--grad_accum_step', type=int, default=1)
    
    parser.add_argument('--preload_gpu', action='store_true', help="preload all data to gpu")
    
    #! may need to change if embed annotation
    # parser.add_argument('--fog_model_input_dim', type=int, default=18*(len(FEATURES_LIST)-1))

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
    source_files = [f'train/train_unet_v{opt.version}.py', 
                    f'utils/train_util.py', 'utils/config.py',
                    f'data/fog_dataset.py',
                    f'models/unet_v{opt.version}.py']
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
    
    opt.mylogger = MYLOGGER

    run_train(opt)


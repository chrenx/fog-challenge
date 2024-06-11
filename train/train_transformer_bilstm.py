import argparse, logging, os, random, shutil
from datetime import datetime

import torch, wandb
import numpy as np
from torch.optim import Adam, AdamW
from torch.utils import data

from data.fog_dataset import FoGDataset
from models.transformer_bilstm import TransformerBiLSTM
from tqdm import tqdm
from utils.config import FEATURES_LIST
from utils.train_util import CustomLRScheduler, cycle_dataloader, save_group_args


MYLOGGER = logging.getLogger()


class Trainer(object):
    def __init__(self, model, opt):
        super().__init__()
        
        self.use_wandb = not opt.disable_wandb
        self.save_best_model = opt.save_best_model
        self.step = 0
        self.weights_dir = opt.weights_dir
        
        if self.use_wandb:
            MYLOGGER.info("Initialize W&B")
            wandb.init(config=opt, project=opt.wandb_pj_name, entity=opt.entity, 
                       name=opt.exp_name, dir=opt.save_dir)
        
        #* NEW: self.train_dl, self.train_ds_len, self.val_dl, self.val_ds_len
        self._prepare_dataloader(opt) 

        self.model = model
        self.optimizer = None
        self.cur_step = 0
        self.train_num_steps = opt.train_num_steps
        self.device = opt.device
        self.bce_loss = torch.nn.BCELoss(reduction='none')
        
        if opt.optimizer == 'adam':
            self.optimizer = Adam(self.model.parameters(), lr=opt.learning_rate)
        elif opt.optimizer == 'adamw':
            self.optimizer = AdamW(self.model.parameters(), lr=opt.learning_rate)
        assert self.optimizer is not None, "Error: optimizer is not given."
        
        self.scheduler_optim = CustomLRScheduler(self.optimizer,
                                                 initial_lr=opt.learning_rate, 
                                                 warmup_steps=64)
        
    def _prepare_dataloader(self, opt):
        MYLOGGER.info("Loading training data ...")
        
        self.train_ds = FoGDataset(opt, mode='train')
        self.train_ds_len = len(self.train_ds)
        
        self.val_ds = FoGDataset(opt, mode='val')
        self.val_ds_len = len(self.val_ds)
        
        self.train_dl = cycle_dataloader(data.DataLoader(self.train_ds, 
                                                        batch_size=opt.batch_size, 
                                                        shuffle=True, 
                                                        pin_memory=False, 
                                                        num_workers=0))
        
        self.val_dl = cycle_dataloader(data.DataLoader(self.val_ds, 
                                                       batch_size=opt.batch_size, 
                                                       shuffle=False, 
                                                       pin_memory=False, 
                                                       num_workers=0))
        
    def _loss_func(self, pred, gt):
        """Compute the Binary Cross-Entropy loss for each class and sum over the class dimension

        Args:
            pred: (B, BLKS//P, 2)
            gt: (B, BLKS//P, 3)
        """
        loss = self.bce_loss(pred, gt[:,:,:2]) # (B, BLKS//P, 2)
        mask = (gt[:,:,2] != 1).float() # (B, BLKS//P)
        mask = mask.unsqueeze(-1).expand(-1, -1, 2) # (B, BLKS//P, 2)
        loss *= mask
        return loss.sum() / mask.sum()

    def _evaluation_metrics(self, pred, gt):
        # Convert the model output probabilities to class predictions
        pred = torch.argmax(pred, dim=-1)  # (B, BLKS//P)

        # Extract the first two classes from the ground truth
        real = torch.argmax(gt[:, :, :2], dim=-1)  # (B, BLKS//P)

        # Create a mask to ignore the positions where the ground truth class is 2
        mask = (gt[:, :, 2] != 2).float()  # (B, BLKS//P)

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
        
        return precision, recall, f1
    
    def _save_model(self, step, best=False):
        data = {
            'step': step,
            'model': self.model.state_dict(),
        }
        filename = f"best_model_{step}.pt" if best else f"model_{step}.pt"
        torch.save(data, os.path.join(self.weights_dir, filename))

    def train(self):
        best_f1 = 0
        for step_idx in tqdm(range(0, self.train_num_steps), desc="Train"):
            self.model.train()
            self.optimizer.zero_grad()
            
            # training part ----------------------------------------------------
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
                wandb.log(log_dict)
                
            # validation part --------------------------------------------------
            if (step_idx + 1) % self.train_ds_len == 0: #* An Epoch
                avg_val_f1 = 0.0
                avg_val_loss = 0.0
                self.model.eval()
                with torch.no_grad():
                    for _ in tqdm(range(self.val_ds_len), desc="Validation"):
                        val_data = next(self.val_dl) 
                        val_input = val_data['model_input']
                        val_gt = val_data['gt']
                        val_pred = self.model(val_input) # (B, BLKS//P, 2)
                        prec, recall, f1 = self._evaluation_metrics(val_pred, 
                                                                    val_gt.to(self.device))
                        val_loss = self._loss_func(val_pred, val_gt.to(self.device))
                        avg_val_f1 += f1
                        avg_val_loss += val_loss
                        
                        if self.use_wandb:
                            log_dict = {
                                "Val/precision": prec.item(),
                                "Val/recall": recall.item(),
                                "Val/f1_score": f1.item(),
                                "Val/loss": val_loss.item(),
                            }
                            wandb.log(log_dict)

                    avg_val_f1 /= self.val_ds_len
                    avg_val_loss /= self.val_ds_len
                    
                MYLOGGER.info("Step: {0}".format(step_idx))
                MYLOGGER.info("F1-Score: %.4f" % (avg_val_f1))
                MYLOGGER.info("Avg Val Loss: %.4f" % (avg_val_loss))
                
                # Log learning rate
                if self.use_wandb:
                    for param_group in self.optimizer.param_groups:
                        wandb.log({'learning_rate': param_group['lr']})
                
                if self.save_best_model and avg_val_f1 > best_f1:
                    best_f1 = avg_val_f1
                    self._save_model(step_idx, best=True)
                else:
                    self._save_model(step_idx, best=False)
                    
                self.scheduler_optim.step()


def run_train(opt):
    model = TransformerBiLSTM(opt)
    model.to(opt.device)
    trainer = Trainer(model, opt)
    trainer.train()
    torch.cuda.empty_cache()

def parse_opt():
    parser = argparse.ArgumentParser()
    
    # project information: names ===============================================
    parser.add_argument('--project', default='runs/train', help='project/name')
    parser.add_argument('--exp_name', default='transfomer_bilstm', 
                                            help='save to project/name')

    # wandb setup ==============================================================
    parser.add_argument('--disable_wandb', action='store_true')
    parser.add_argument('--wandb_pj_name', type=str, default='fog-challenge', 
                                           help='wandb project name')
    parser.add_argument('--entity', default='', help='W&B entity or username')

    # data path
    parser.add_argument('--root_dpath', default='data/rectified_data', 
                                        help='directory that contains different processed datasets')
    
    # GPU ======================================================================
    parser.add_argument('--device', default='0', help='assign gpu')
    parser.add_argument('--device_info', type=str, default='')
    
    # training monitor =========================================================
    # parser.add_argument('--save_and_sample_every', type=int, default=50, 
    #                                                     help='save and sample')
    parser.add_argument('--save_best_model', action='store_true', 
                                                  help='save best model during training')

    # hyperparameters ==========================================================
    parser.add_argument('--seed', type=int, default=42, 
                                      help='set up seed for torch, numpy, random')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--optimizer', type=str, default="adam", 
                                       help="Choice includes [adam, adamw]")
    parser.add_argument('--learning_rate', type=float, default=2e-4, 
                                           help='generator_learning_rate')
    
    parser.add_argument('--train_num_steps', type=int, default=800000, 
                                                 help='number of training steps')
    
    parser.add_argument('--block_size', type=int, default=15552)
    parser.add_argument('--block_stride', type=int, default=15552 // 16) # 972
    parser.add_argument('--patch_size', type=int, default=18)

    parser.add_argument('--fog_model_input_dim', type=int, default=18*len(FEATURES_LIST))
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
    cur_time = datetime.now()
    cur_time = '{:%Y_%m_%d_%H:%M:%S}.{:02.0f}'.format(cur_time, cur_time.microsecond / 10000.0)
    opt.save_dir = os.path.join(opt.project, opt.exp_name, cur_time)
    if opt.device != 'cpu':
        opt.device_info = torch.cuda.get_device_name(int(opt.device)) 
        opt.device = f"cuda:{opt.device}"
    else:
        print("!!!!!!!! Running on CPU.")

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    # Create dir
    os.makedirs(opt.save_dir, exist_ok=True)

    # Save some important code
    source_files = ['train/train_transformer_bilstm.py']
    codes_dest = os.path.join(opt.save_dir, 'codes')
    os.makedirs(codes_dest)
    for file_dir in source_files:
        shutil.copy2(file_dir, codes_dest)
        
    # Create weight dir to store model weights
    opt.weights_dir = os.path.join(opt.save_dir, 'weights')
    os.makedirs(opt.weights_dir, exist_ok=True)
    
    logging.basicConfig(filename=os.path.join(opt.save_dir,"training_info.log"),
                        filemode='a',
                        format='%(asctime)s.%(msecs)02d %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d-%H:%M:%S',
                        level=os.environ.get("LOGLEVEL", "INFO"))
    MYLOGGER.setLevel(logging.INFO)
    MYLOGGER.info(f"Running at {cur_time}")
    MYLOGGER.info(f"Using device: {opt.device}")
    
    save_group_args(opt)

    run_train(opt)


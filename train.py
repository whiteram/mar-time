
import os
import math
import random 
import torch
from torch import nn 
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision import transforms
from models.MAR import MAR, PatchEmbed
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from typing import Any, Dict
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
seed_torch(7)

####################################################################
# MODEL
class LitModel(pl.LightningModule):

    def __init__(self):
        super().__init__()   
        self.model = MAR(
            img_size            =(1, 256),
            patch_size          =16,
            embed_dim           =64,
            embed_num           =1,
            depth               =8,
            num_heads           =4,
            mlp_ratio           =4.0,
            qkv_bias            =True,
            drop_rate           =0.0,
            attn_drop_rate      =0.0,
            drop_path_rate      =0.0,
            norm_layer          =nn.LayerNorm,
            patch_module        =PatchEmbed,# PatchNormEmbed
            init_std            =0.02,
            
            # -- masking params
            mNC_x = 5,
            mNC_y = 8,
            
            # -- diffusion loss params
            diffloss_w          = 256,
            diffloss_d          = 3,
            num_sampling_steps  = 1000,
            diffusion_batch_mul = 4,
        )
    
    def validation_step(self, batch, batch_idx):
        loss = self.model(batch)
        self.log('valid_loss' , loss , on_epoch=True, on_step=False, sync_dist=True)
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self.model(batch)
        #############################################################################
        if batch_idx==1:
            tensorboard = self.loggers[0].experiment
            for name, param in self.model.named_parameters():
                if isinstance(param, nn.Parameter) and len(param.shape)>0: 
                    if (param is None): continue
                    tensorboard.add_histogram(name + '_data', param, self.current_epoch)
                    if param.requires_grad and (not (param.grad is None)):
                        tensorboard.add_histogram(name + '_grad', param.grad, self.current_epoch)
        #############################################################################
        self.log('train_loss' , loss , on_epoch=True, on_step=False, sync_dist=True)
        return loss

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        res = super().on_load_checkpoint(checkpoint)
        self.configure_optimizers()
        return res
    
    def configure_optimizers(self):
        
        param_groups = [
            {
                'params': (p for n, p in self.model.named_parameters())
            }, 
        ]
        
        optimizer = torch.optim.AdamW(param_groups, lr=6e-5)        
        
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=ARGS.max_lr, steps_per_epoch=ARGS.steps_per_epoch, 
                                                           epochs=ARGS.max_epochs,
                                                           div_factor = 2,
                                                           final_div_factor=8,
                                                           pct_start = 0.2 ,
                                                           )
        lr_dict = {
            'scheduler': lr_scheduler, # The LR scheduler instance (required)
            # The unit of the scheduler's step size, could also be 'step'
            'interval': 'step',
            'frequency': 1, # The frequency of the scheduler
            'monitor': 'valid_loss', # Metric for `ReduceLROnPlateau` to monitor
            'strict': True, # Whether to crash the training if `monitor` is not found
            'name': None, # Custom name for `LearningRateMonitor` to use
        }
        return (
            {'optimizer': optimizer, 'lr_scheduler': lr_dict},
        )

####################################################################
# ARGS
class ARGS:
    max_epochs      = 200
    steps_per_epoch = None
    max_lr          = 1e-3
    batch_size      = 16*8
    save_path       = './logs/checkpoints'
    save_name       = 'mar-time-epoch={epoch}-valid_loss={valid_loss:.4f}'
    
    # accelerator = 'cpu'
    accelerator = 'auto'
    devices     = [0, 1, 2, 3, 4, 5]

if ARGS.accelerator == 'cpu':
    ARGS.devices = "auto"

####################################################################

data_info_path = os.path.join(r'/disks/SSD/data/transformed2/Data_processed/all_samples_info.csv')
data_info = pd.read_csv(data_info_path)

data_root = os.path.join(r'/disks/SSD/data/transformed2/')

train_dataset_mask = (data_info['Dataset']!='Zhou2016')&(data_info['Dataset']!='BNCI2015_001')&(data_info['Dataset']!='BNCI2014_001')&(data_info['Dataset']!='AlexMI')&(data_info['Dataset']!='BNCI2014_002')&(data_info['Dataset']!='BNCI2014_004')
valid_dataset_mask = (data_info['Dataset']=='Zhou2016')|(data_info['Dataset']=='BNCI2015_001')|(data_info['Dataset']=='BNCI2014_001')|(data_info['Dataset']=='AlexMI')
train_info = data_info[train_dataset_mask]
valid_info = data_info[valid_dataset_mask]

class eeg_dataset(Dataset):
    def __init__(self, data_info, root_dir, sub_id = False,  transform=None,):

        self.data_info = data_info
                
        #所有用到的导联的电极
        self.electrode_names   =  [k.upper() for k in [
            'Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7','TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz','Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2', 'Fp1h', 'Fp2h', 'AF1', 'AF2', 'AF5', 'AF6', 'F9', 'F10', 'FT9', 'FT10', 'TP9', 'TP10', 'P9h', 'P10h', 'F1h', 'F2h', 'F5h', 'F6h', 'F7h', 'F8h', 'FC1h', 'FC2h', 'FC5h', 'FC6h', 'FT1', 'FT2', 'C1h', 'C2h', 'C5h', 'C6h', 'T1', 'T2', 'TP1', 'TP2', 'CP1h', 'CP2h', 'CP5h','CP6h', 'TP3', 'TP4', 'P1h', 'P2h', 'P3h', 'P4h', 'PO1', 'PO2', 'PO5', 'PO6', 'O9', 'O10', 'FT7h','FT8h', 'TP7h', 'TP8h', 'PO9', 'PO10', 'Iz2', 'Oz2', 'Pz2', 'CPz2', 'TPP9h', 'TPP10h', 'AFF1', 'AFF2', 'FFC5h', 'FFC3h', 'FFC4h', 'FFC6h', 'FCC5h', 'FCC3h', 'FCC4h', 'FCC6h', 'CCP5h', 'CCP3h', 'CCP4h','CCP6h', 'CPP5h', 'CPP3h', 'CPP4h', 'CPP6h', 'PPO1', 'PPO2', 'I1', 'I2','AFp3h', 'AFp4h', 'AFF5h', 'AFF6h', 'FFT7h', 'FFC1h', 'FFC2h', 'FFT8h','FTT9h', 'FTT7h', 'FCC1h', 'FCC2h', 'FTT8h', 'FTT10h', 'TTP7h', 'CCP1h', 'CCP2h', 'TTP8h', 'TPP7h', 'CPP1h', 'CPP2h','TPP8h','PPO9h', 'PPO5h','PPO6h', 'PPO10h', 'POO9h', 'POO3h', 'POO4h', 'POO10h', 'OI1h', 'OI2h', 'T3', 'T4', 'T9', 'T10', 'AFp1', 'AFp2', 'AFF1h', 'AFF2h', 'PPO1h', 'POO1', 'POO2', 'PPO2h', 'NONE'
         ]]


        self.all_labels  = [
            'right_hand', 'left_hand', 'feet', 'tongue', 'rest','subtraction','word_ass', 'navigation', 'hands', 'left_hand_right_foot','right_hand_left_foot', 'right_pronation', 'right_hand_close','right_elbow_extension', 'right_hand_open', 'right_supination','right_elbow_flexion', 'both_hand']
        
        self.root_dir = root_dir

        self.sub_id = sub_id
  

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):

        sample_info =  self.data_info.iloc[idx]
        path = sample_info['data_path']
        raw_x = torch.FloatTensor(np.load(os.path.join(self.root_dir, path)))
        raw_x = torch.nn.functional.layer_norm(raw_x, normalized_shape=raw_x.shape)
        TIME_LEN = 1*256
        random_began = random.randint(0, raw_x.shape[1]-TIME_LEN)
        random_channel = random.randint(0, raw_x.shape[0]-1)
        x = raw_x[random_channel:random_channel+1, random_began:random_began+TIME_LEN]
        return x
    
if __name__ == '__main__':
    ##################################################################
    # Dataset
    train_dataset = eeg_dataset(train_info,data_root)
    valid_dataset = eeg_dataset(valid_info,data_root)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=ARGS.batch_size, num_workers=4, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=ARGS.batch_size, num_workers=4, shuffle=False)
    
    
    ARGS.steps_per_epoch = math.ceil(len(train_loader) / (len(ARGS.devices) if ARGS.devices != "auto" else 1))

    # -- init model
    torch.set_float32_matmul_precision("medium")
    model = LitModel()

    ckpt_callback = ModelCheckpoint(
        monitor='valid_loss',
        save_top_k=1,
        mode='min',
        dirpath=ARGS.save_path,
        filename='best-{}'.format(ARGS.save_name),
        auto_insert_metric_name=False
    )
    last_callback = ModelCheckpoint(
        every_n_epochs=ARGS.max_epochs,
        save_top_k=1,
        dirpath=ARGS.save_path,
        filename='last-{}'.format(ARGS.save_name),
        auto_insert_metric_name=False
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    
    callbacks = [lr_monitor, ckpt_callback, last_callback]

    trainer = pl.Trainer(   
                            accelerator=ARGS.accelerator,
                            devices    =ARGS.devices, 
                            gradient_clip_val = 1.0,
                            gradient_clip_algorithm="norm",
                            strategy='auto',
                            #strategy='ddp_find_unused_parameters_true', 
                            #precision="bf16-mixed",
                            precision=32, 
                            max_epochs=ARGS.max_epochs, 
                            callbacks=callbacks,
                            logger=[
                                pl_loggers.TensorBoardLogger('./logs/', name="mar_tb"), 
                                pl_loggers.CSVLogger('./logs/', name="mar_csv")])
    trainer.fit(model, train_loader, valid_loader, )
    


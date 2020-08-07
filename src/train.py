import os
import torch
import pandas as pd
from scipy import stats
import numpy as np

from tqdm import tqdm
from collections import OrderedDict, namedtuple
import torch.nn as nn
from torch.optim import lr_scheduler
import joblib

import logging
import transformers
from transformers import AdamW, get_constant_schedule_with_warmup, get_constant_schedule
import sys
from sklearn import metrics, model_selection

import warnings
import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.distributed.data_parallel as dp
import torch_xla.distributed.parallel_loader as pl
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.test.test_utils as test_utils

import config
import dataset
import model
import engine

warnings.filterwarnings('ignore')

class AverageMeter:
    '''
    Computes and stores the average and current value
    '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum / self.count

def run():
    df_train1 = pd.read_csv(config.TRAIN_PATH_1, usecols=['comment_text', 'toxic']).fillna('none')
    df_train2 = pd.read_csv(config.TRAIN_PATH_2, usecols=['comment_text', 'toxic']).fillna('none')
    
    df_train_full = pd.concat([df_train1, df_train2], axis=0).reset_index(drop=True)

    df_train_full = df_train_full.sample(frac=1).reset_index(drop=True).head(400000)
    
    df_valid = pd.read_csv(config.VALID_PATH)

    tokenizer = config.tokenizer
    
    train_targets = df_train.toxic.values
    valid_targets = df_valid.toxic.values

    train_dataset = dataset.BERTDatasetTraining(
        comment_text=df_train.comment_text.values, 
        targets=train_targets,
        tokenizer=tokenizer, 
        max_length=config.MAX_LEN
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, 
        num_replicas=xm.xrt_world_size(), 
        rank=xm.get_ordinal(),
        shuffle=True
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=config.TRAIN_BATCH_SIZE, 
        sampler=train_sampler, 
        drop_last=True, 
        num_workers=4
    )

    valid_dataset = dataset.BERTDatasetTraining(
        comment_text=df_valid.comment_text.values,
        targets=valid_targets,
        tokeizer=config.tokenizer, 
        max_length=config.MAX_LEN
    )
    
    valid_sampler = torch.utils.data.distributed.DistributedSampler(
        valid_dataset, 
        num_replicas=xm.xrt_world_size(), 
        rank=xm.get_ordinal(), 
        shuffle=False
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, 
        batch_size=config.VALID_BATCH_SIZE, 
        sampler=valid_sampler, 
        drop_last=True, 
        num_workers=4 
    )

    device = xm.xla_device()
    model = BERTBaseUncased(bert_path=config.MODEL_PATH).to(device)
    
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    
    optimizer_grouped_parameters = [
        {'params':[p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay':0.01}, 
        {'params':[p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay':0.0}
    ]

    lr = 3e-5 * xm.xrt_world_size()
    num_train_steps = int(len(train_dataset)/config.TRAIN_BATCH_SIZE/xm.xrt_world_size()*EPOCHS)
    xm.master_print(f'num_train_steps={num_train_steps}, world_size={xm.world_size()}')
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    scheduler = get_constant_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0, 
        num_training_steps=num_train_steps
    )

    for epoch in range(config.EPOCHS):
        para_loader = pl.ParallelLoader(train_data_loader, [device])
        engine.train_fn(para_loader.per_device_loader(device), model, 
        optimizer, device, scheduler=scheduler)
        
        para_loader = pl.ParallelLoader(valid_data_loader, [device])
        o, t = engine.eval_fn(para_loader.per_device_loader, model, device)
        xm.save(model.state_dict(), 'model.bin')
        auc = metrics.roc_auc_score(np.array(t) >= 0.5, o)
        xm.master_print(f'AUC ={auc}')
        
        
    
    
    
    
    

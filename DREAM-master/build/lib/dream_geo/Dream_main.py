# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 12:20:27 2022

@author: lenovo
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import _init_paths
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from enum import IntEnum

import albumentations as albu
import numpy as np
from PIL import Image as PILImage
import torch
from torch.utils.data import Dataset as TorchDataset
import torchvision.transforms as TVTransforms
import numpy as np
from ruamel.yaml import YAML
import torch.utils.data
from lib.opts import opts
from lib.model.model import create_model, load_model, save_model
from lib.model.data_parallel import DataParallel
from lib.logger import Logger
from utilities import find_ndds_seq_data_in_dir, set_random_seed, exists_or_mkdir
from datasets import CenterTrackSeqDataset
# import dream_geo as dream
# from lib.dataset.dataset_factory import get_dataset # 这里的dataset用我们自己的
from lib.Dream_trainer import Trainer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def get_optimizer(opt, model):
    if opt.optim == 'adam':
      optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    elif opt.optim == 'sgd':
      print('Using SGD')
      optimizer = torch.optim.SGD(
        model.parameters(), opt.lr, momentum=0.9, weight_decay=0.0001)
    else:
      assert 0, opt.optim
    return optimizer


def main(opt):
    set_random_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
    
    # 这些地方写tensorboard
    tb_path = os.path.join(opt.save_dir, 'tb')
    ckpt_path = os.path.join(opt.save_dir, 'ckpt')
    exists_or_mkdir(tb_path)
    exists_or_mkdir(ckpt_path)
    writer = SummaryWriter(tb_path)

    input_data_path = opt.dataset # 这里是dataset的路径
    found_data = find_ndds_seq_data_in_dir(input_data_path)
    keypoint_names = [
    "Link0",
    "Link1",
    "Link3",
    "Link4", 
    "Link6",
    "Link7",
    "Panda_hand",
    ]
    
    network_input_resolution = (480, 480) # 时刻需要注意这里是width x height
    network_output_resolution = (120, 120) # 时刻需要注意这里是width x height
    input_width, input_height = network_input_resolution
    network_input_resolution_transpose = (input_height, input_width) # centertrack的输入是HxW
    opt = opts().update_dataset_info_and_set_heads_dream(opt, 7, network_input_resolution_transpose)

    Dataset = CenterTrackSeqDataset(
    found_data, 
    "Franka_Emika_Panda", 
    keypoint_names, 
    opt, 
    [0.5, 0.5, 0.5],
    [0.5, 0.5, 0.5],
    include_ground_truth=True,
    include_belief_maps=True
    )
    
    print('length dataset', len(Dataset))
    
    n_data = len(Dataset)
    n_train_data = int(round(n_data * 0.67))
    n_valid_data = n_data - n_train_data
    Dataset, valid_dataset = torch.utils.data.random_split(
        Dataset, [n_train_data, n_valid_data]
    )
    
    print(opt)
#    if not opt.not_set_cuda_env:
#      print(opt.not_set_cuda_env)
#      print('gpus_str', opt.gpus_str)
#      os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
    
    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv, opt=opt)
    optimizer = get_optimizer(opt, model)
    start_epoch = 0
    
    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(
        model, opt.load_model, opt, optimizer)
    
    trainer = Trainer(opt, model, optimizer)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)
    train_loader = torch.utils.data.DataLoader(
        Dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True, drop_last=True
        )
    
    for epoch in tqdm(range(start_epoch + 1, opt.num_epochs + 1)):
        trainer.train(epoch, train_loader, opt.device, writer)
        this_path = os.path.join(ckpt_path, "model_{}.pth".format(epoch))
        save_model(this_path, epoch, model, optimizer)

if __name__ == '__main__':
  opt = opts().parse()
  main(opt)



















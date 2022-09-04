# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 10:42:02 2022

@author: lenovo
"""
# 设计我们自己的trainer, 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
import numpy as np
from progress.bar import Bar

from .model.data_parallel import DataParallel
from .utils.utils import AverageMeter

from .model.losses import FastFocalLoss, RegWeightedL1Loss
from .model.losses import BinRotLoss, WeightedBCELoss
from .model.decode import generic_decode
from .model.utils import _sigmoid, flip_tensor, flip_lr_off, flip_lr
from .utils.debugger import Debugger
from .utils.post_process import generic_post_process
import dream_geo as dream
from tqdm import tqdm

# 要定义loss_function

class RegL1Loss(torch.nn.Module):
    def __init__(self):
        super(RegL1Loss, self).__init__() 
        self.loss = torch.nn.SmoothL1Loss()
    
    def forward(self, output, kp_projs_dis, cord):
        # output为bs x 2 x H X W
        # kp_projs为bs x 7 x 2 (先x, 后y)
        # cord为 bs x 7 x 2 (先x, 后y)
        output = output.permute(0, 2, 3, 1).contiguous()
        bs, n_kp, _ = kp_projs_dis.shape
        loss_data = torch.zeros(bs,n_kp, 2)
        for batch_idx in range(bs):
            for kp in range(n_kp):
                cor_x, cor_y = cord[batch_idx][kp].type(torch.long)
                out_x, out_y = output[batch_idx][cor_y][cor_x]
                loss_data[batch_idx][kp][0] = out_x
                loss_data[batch_idx][kp][1] = out_y
        
        loss = self.loss(loss_data, kp_projs_dis)
        return loss
    
class Loss(torch.nn.Module):
    def __init__(self, opt):
        super(Loss, self).__init__()
        self.crit = torch.nn.MSELoss()
        self.crit_reg = RegL1Loss()
        self.opt = opt
    
    def _sigmoid_output(self, output):
        if 'hm' in output:
            output['hm'] = _sigmoid(output['hm'])
        if 'hm_hp' in output:
            output['hm_hp'] = _sigmoid(output['hm_hp'])
        if 'dep' in output:
            otput['dep'] = 1. / (output['dep'].sigmoid() + 1e-6) - 1
        return output
    
    def forward(self, outputs, batch):
        opt = self.opt
        losses = {head: 0 for head in opt.heads}
        weights = {head : 1 for head in opt.heads}
        weights['tracking'] = 0.01
        weights['reg'] = 0.01
        
        
        for s in range(opt.num_stacks):
            output = outputs[s]
#            for key in output:
#                print(key, output[key].shape)
#            for key in batch:
#                if key != "config":
#                    print(key, batch[key].shape)
            
            
            output = self._sigmoid_output(output)
            
            if 'hm' in output:
                losses['hm'] += self.crit(output['hm'], batch["next_belief_maps"].to(opt.device)) / opt.num_stacks
            
            regression_heads = [
            'reg', 'tracking'] 
            # reg是点对点的offset, 
            # wh和ltrb_amodal是size之间的回归，
            # track表示两帧之间的差距，
            # 所以我们这里不需要wh和ltrb_amodal
            for head in regression_heads:
                losses[head] += self.crit_reg(
                    output[head], batch[head], batch["next_keypoint_projections_output_int"]
                    ) / opt.num_stacks
        
        losses['tot'] = 0
        for head in opt.heads:
            losses['tot'] += losses[head] * weights[head]
            
        return losses['tot'], losses

class Trainer(object):
    def __init__(
        self, opt, model, optimizer=None
            ):
        self.opt = opt
        self.optimizer = optimizer
        self.model = model
        self.loss = Loss(self.opt)
    
    def set_device(self, gpus, chunk_sizes, device):
        if len(gpus) > 1:
            self.model = DataParallel(
                self.model, device_ids=gpus, chunk_sizes=chunk_sizes
                ).cuda()
        else:
            self.model = self.model.cuda()
        
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, non_blocking=True)
    
    def run_epoch(self, phase, epoch, data_loader, device, writer):
        model = self.model
        if len(self.opt.gpus) > 1:
            model = self.model.module
        opt = self.opt
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            # batch = batch.to(device)
            torch.cuda.empty_cache()
            pre_img = batch['prev_image_rgb_input'].to(device) # bs x 3 x H x W
            pre_hm = batch['prev_belief_maps_as_input_resolution'].to(device) # bs x 1 x H x W
            pre_hm = pre_hm.unsqueeze(1)
            next_img = batch['next_image_rgb_input'].to(device)
#            print('pre_img.size', pre_img.shape)
#            print('pre_hm.size', pre_hm.shape)
#            print('next_img.size', next_img.shape)
            outputs = model(next_img, pre_img, pre_hm)
            loss, loss_stats = self.loss(outputs, batch)
#            print('loss', loss)
            
            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            with torch.no_grad():
                loss_all = loss_stats["tot"].item()
                loss_hm = loss_stats["hm"].item()
                loss_reg = loss_stats["reg"].item()
                loss_tracking = loss_stats["tracking"].item()
                writer.add_scalar(f"training_loss", loss_all, batch_idx + (epoch-1) * len(data_loader))
                writer.add_scalar(f"heatmap_loss", loss_hm, batch_idx + (epoch-1) * len(data_loader))
                writer.add_scalar(f"reg_loss", loss_reg, batch_idx + (epoch-1) * len(data_loader))
                writer.add_scalar(f"tracking_loss", loss_tracking, batch_idx + (epoch-1) * len(data_loader))
                
                if batch_idx % 10 == 0:
                    print('loss_all', loss_all)
                    print('loss_hm', loss_hm)
                    print('loss_reg', loss_reg)
                    print('loss_tracking', loss_tracking)
                
                if batch_idx % 50 == 0:
                    # 我们一共可视化这些东西
                    # 上一帧的input图
                    # 上一帧的一张热力图
                    # 下一帧的input图
                    # 下一帧的预测热力图
                    # 下一帧的热力图gt
                    # 此处的R为4，即下采样倍数
                    
                    output = outputs[0]
                    prev_rgb_net_inputs = dream.image_proc.images_from_tensor(batch["prev_image_rgb_input"]) # bs x 3 x H x W
                    next_rgb_net_inputs = dream.image_proc.images_from_tensor(batch["next_image_rgb_input"]) # bs x 3 x H x W
                    next_gt_belief_maps_wholes = batch["prev_belief_maps_as_input_resolution"] # bs x H x W
                    next_belief_maps = output["hm"] # bs x num_kp x (H/R) x (W/R)
                    next_gt_belief_maps = batch["next_belief_maps"] # bs x num_kp x (H/R) x (W/R)
                    
                    for idx, sample in enumerate(zip(prev_rgb_net_inputs,next_rgb_net_inputs, next_gt_belief_maps_wholes, next_belief_maps, next_gt_belief_maps)):
                        prev_rgb_net_input_img, next_rgb_net_input_img, next_gt_belief_map_whole, next_belief_map, next_gt_belief_map = sample
                        # prev_rgb_net_input与next_rgb_net_input都已经是Image了
                        next_gt_belief_map_whole_img = dream.image_proc.image_from_belief_map(next_gt_belief_map_whole)
                        
                        next_belief_map_img = dream.image_proc.images_from_belief_maps(
                        next_belief_map, normalization_method=6
                        )
                        next_belief_maps_mosaic = dream.image_proc.mosaic_images(
                        next_belief_map_img, rows=2, cols=4, inner_padding_px=10
                        )
                        next_gt_belief_map_img = dream.image_proc.images_from_belief_maps(
                        next_gt_belief_map, normalization_method=6
                        )
                        next_gt_belief_maps_mosaic = dream.image_proc.mosaic_images(
                        next_gt_belief_map_img, rows=2, cols=4, inner_padding_px=10
                        )
                          
                        writer.add_image(f'{idx} prev_rgb_net_input_img', np.array(prev_rgb_net_input_img), batch_idx + (epoch-1) * len(data_loader), dataformats='HWC')
                        writer.add_image(f'{idx} next_rgb_net_input_img', np.array(next_rgb_net_input_img), batch_idx + (epoch-1) * len(data_loader), dataformats='HWC')
                        writer.add_image(f'{idx} prev_gt_belief_map_whole_img', np.array(next_gt_belief_map_whole_img), batch_idx + (epoch-1) * len(data_loader), dataformats='HWC')
                        writer.add_image(f'{idx} next_belief_maps_img', np.array(next_belief_maps_mosaic), batch_idx + (epoch-1) * len(data_loader), dataformats='HWC')
                        writer.add_image(f'{idx} next_gt_belief_map_img', np.array(next_gt_belief_maps_mosaic), batch_idx + (epoch-1) * len(data_loader), dataformats='HWC')
                        
                        
                        
    
    def train(self, epoch, train_loader,device, writer):
        return self.run_epoch('train', epoch, train_loader, device, writer)
        
    






















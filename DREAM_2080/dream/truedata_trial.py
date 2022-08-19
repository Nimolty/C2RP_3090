# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 09:26:56 2022

@author: lenovo
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"
import torchvision.utils
import torch.utils.data as Data
from PIL import Image as PILImage

import numpy as np
from ruamel.yaml import YAML
import torch
import torchvision.transforms as TVTransforms
import cv2
from torch.utils.tensorboard import SummaryWriter
from epropnp_all import *

from dream.monte_carlo_pose_loss import MonteCarloPoseLoss
from dream.geometric_vision import *
from dream.image_proc import *
from dream.spatial_softmax import SoftArgmaxPavlo
import dream
from make_data_raw import *

# 写一个dataloader,让它读进来吧

# 假设有了一个dataloader
# 需要有一个cam_mat
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_all = 1000
n_epochs = 10
bs = 32
x3d_all, x2d_all, r_gt, t_gt = np.zeros((num_all, 7, 3)), \
    np.zeros((num_all, 7, 2)), np.zeros((num_all, 4)), np.zeros((num_all, 3))

for i in range(num_all):
    x3d_all[i], x2d_all[i], r_gt[i], t_gt[i] = make_one_pose(7)
    
sample = torch.from_numpy(np.concatenate([x3d_all, x2d_all], axis = -1)).to(device)
gt = torch.from_numpy(np.concatenate([t_gt, r_gt], axis = -1)).to(device)
dataset = Data.TensorDataset(sample, gt)
loader = Data.DataLoader(
    dataset=dataset,
    batch_size=bs,
    shuffle=True)

camera = PerspectiveCamera()
cost_fun = AdaptiveHuberPnPCost(relative_delta=0.5)
log_weight_scale = torch.zeros(2)
epropnp = EProPnP6DoF(
            mc_samples=1024,
            num_iter=32,
            solver=LMSolver(
                dof=6,
                num_iter=20,
                init_solver=RSLMSolver(
                    dof=6,
                    num_points=6,
                    num_proposals=128,
                    num_iter=50)))


for epoch in range(n_epochs):
    for batch_idx, (sample, gt) in enumerate(tqdm(loader)):
        
        # x3d shape(bs, num_pt, 3)
        x3d = sample[:, :, :3]
        x2d = sample[:, :, 3:]
        gt_pose = gt
        batch_size, num_pt, _ = x3d.shape
        # x2d shape(bs, num_pt, 2)
        #x3d = sample['x3d']
        #x2d = sample['x2d']
        cam_mats = torch.tensor(cam_mat).expand(batch_size, -1, -1).to(device)
        w2d = torch.full([batch_size, num_pt, 2], 1 / num_pt).to(device) * log_weight_scale.to(device).exp()
        cost_fun.set_param(x2d.detach(), w2d)
        
        pose_opt, cost, pose_opt_plus, pose_samples, pose_sample_logweights, cost_tgt = epropnp.monte_carlo_forward(
                x3d,
                x2d,
                w2d,
                camera,
                cost_fun,
                pose_init=gt_pose,
                force_init_solve=True,
                with_pose_opt_plus=True)
        
        norm_factor = log_weight_scale.detach().exp().mean()
        
        pose_opt, _, _, _ = epropnp(x3d, x2d, w2d, camera, cost_fun)
                
                
        dr_pose = torch.zeros_like(gt_pose)
        tt_pose = torch.zeros_like(gt_pose)
        distCoeffs = np.asarray([0, 0, 0, 0, 0], dtype=np.float64)
        for p in range(len(pose_opt_plus)):
#                    ###dream自己的solve_pnp，即后面的dr
            pnp_retval, translation, quaternion, inliers = solve_pnp_ransac(canonical_points = np.asarray(x3d[p].detach().cpu()), projections = np.asarray(x2d[p].detach().cpu()), camera_K = cam_mat, dist_coeffs=distCoeffs)
            dream_pose = torch.cat((torch.tensor(translation), torch.tensor(quaternion))).reshape(1, 7).to(device)
                    
            dr_pose[p] = dream_pose
                    
            _, rrval, ttval = cv2.solvePnP(np.asarray(x3d[p].detach().cpu()), np.asarray(x2d[p].detach().cpu()), cam_mat, distCoeffs)
            rmat_val, _ = cv2.Rodrigues(rrval)
            rqval = matrix_to_quaternion(torch.tensor(rmat_val)).unsqueeze(1)
            t_pose = torch.cat((torch.tensor(ttval), torch.tensor(rqval))).reshape(1, 7).to(device)
            tt_pose[p] = t_pose
                    
            print('dr: ', dream_pose)
            print('cv: ', t_pose)
            print('gt: ', pose_opt[p])
        
        
        















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
from tqdm import tqdm

import numpy as np
from ruamel.yaml import YAML
import torch
import torchvision.transforms as TVTransforms
from torch.utils.tensorboard import SummaryWriter
from epropnp_all import *

from make_data_raw import make_one_pose

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_all = 1000
n_epochs = 10
bs = 32
x3d_all, x2d_all, r_gt, t_gt = np.zeros((num_all, 7, 3)), \
    np.zeros((num_all, 7, 2)), np.zeros((num_all, 4)), np.zeros((num_all, 3))

for i in range(num_all):
    x3d_all[i], x2d_all[i], r_gt[i], t_gt[i] = make_one_pose(7)
    
sample = torch.from_numpy(np.concatenate([x3d_all, x2d_all], axis = -1)).double().to(device)
gt = torch.from_numpy(np.concatenate([t_gt, r_gt], axis = -1)).double().to(device)
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

cam_mat = np.array([[320, 0, 320], [0, 320, 320], [0, 0, 1]], dtype=np.float64)

for epoch in range(n_epochs):
    for batch_idx, (sample, gt) in enumerate(tqdm(loader)):
        
        # x3d shape(bs, num_pt, 3)
        x3d = sample[:, :, :3].double()
        x2d = sample[:, :, 3:].double()
        #print(x3d.dtype, x2d.dtype)
        gt_pose = gt.double()
        batch_size, num_pt, _ = x3d.shape
        # x2d shape(bs, num_pt, 2)
        #x3d = sample['x3d']
        #x2d = sample['x2d']
        cam_mats = torch.tensor(cam_mat).expand(batch_size, -1, -1).double().to(device)
        w2d = torch.full([batch_size, num_pt, 2], 1 / num_pt).double().to(device) * log_weight_scale.to(device).exp()
        cost_fun.set_param(x2d.detach(), w2d)
        camera.set_param(cam_mats)
        
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
                
        #print(pose_opt)   
        dr_pose = torch.zeros_like(gt_pose)
        tt_pose = torch.zeros_like(gt_pose)
        distCoeffs = np.asarray([0, 0, 0, 0, 0], dtype=np.float64)
        for p in range(len(pose_opt_plus)):
#                    ###dream自己的solve_pnp，即后面的dr
#            pnp_retval, translation, quaternion, inliers = solve_pnp_ransac(canonical_points = np.asarray(x3d[p].detach().cpu()), projections = np.asarray(x2d[p].detach().cpu()), camera_K = cam_mat, dist_coeffs=distCoeffs)
#            print(pnp_retval, translation, quaternion, inliers)
#            dream_pose = torch.cat((torch.tensor(translation), torch.tensor(quaternion))).reshape(1, 7).to(device)
            
#            canonical_points = np.asarray(x3d[p].detach().cpu())
#            projections = np.asarray(x2d[p].detach().cpu())
#            n_canonial_points = len(canonical_points)
#            n_projections = len(projections)
#            canonical_points_proc = []
#            projections_proc = []
#            for canon_pt, proj in zip(canonical_points, projections):
#                if (
#                    canon_pt is None
#                    or len(canon_pt) == 0
#                    or canon_pt[0] is None
#                    or canon_pt[1] is None
#                    or proj is None
#                    or len(proj) == 0
#                    or proj[0] is None
#                    or proj[1] is None
#                  ):
#                    continue
#
#                canonical_points_proc.append(canon_pt)
#                projections_proc.append(proj)
#            
#            canonical_points_proc = np.array(canonical_points_proc)
#            projections_proc = np.array(projections_proc)
#            
#            pnp_retval, rvec, tvec, inliers = cv2.solvePnPRansac(
#            canonical_points_proc.reshape(canonical_points_proc.shape[0], 1, 3),
#            projections_proc.reshape(projections_proc.shape[0], 1, 2),
#            cam_mat,
#            distCoeffs=distCoeffs,
#            reprojectionError=5.0,
#            flags=cv2.SOLVEPNP_EPNP,
#            )
#            print(rvec, tvec)
#
#            translation = tvec[:, 0]
#            quaternion = convert_rvec_to_quaternion(rvec[:, 0])
#            dream_pose = torch.cat((torch.tensor(translation), torch.tensor(quaternion))).reshape(1, 7).to(device)
#            dr_pose[p] = dream_pose
                    
            _, rrval, ttval = cv2.solvePnP(np.asarray(x3d[p].detach().cpu()), np.asarray(x2d[p].detach().cpu()), cam_mat, distCoeffs)
            rmat_val, _ = cv2.Rodrigues(rrval)
            rqval = matrix_to_quaternion(torch.tensor(rmat_val)).unsqueeze(1)
            t_pose = torch.cat((torch.tensor(ttval), torch.tensor(rqval))).reshape(1, 7).to(device)
            tt_pose[p] = t_pose
                    
            print('cv: ', t_pose)
            print('gt: ', gt_pose[p])
            print('pnp: ', pose_opt[p])
        
        
        















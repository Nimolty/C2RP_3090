"""
Copyright (C) 2010-2022 Alibaba Group Holding Limited.
"""

import torch
import torch.nn as nn


class MonteCarloPoseLoss(nn.Module):

    def __init__(self, init_norm_factor=1.0, momentum=0.1):
        super(MonteCarloPoseLoss, self).__init__()
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.register_buffer('norm_factor', torch.tensor(init_norm_factor, dtype=torch.float))
        self.momentum = momentum

    def forward(self, pose_sample_logweights, cost_target, norm_factor):
        """
        Args:
            pose_sample_logweights: Shape (mc_samples, num_obj)
            cost_target: Shape (num_obj, )
            norm_factor: Shape ()
        """
        if self.training:
            #print('self.norm_factor.device', self.norm_factor.device)
            with torch.no_grad():
                self.norm_factor.mul_(
                    1 - self.momentum).add_(self.momentum * norm_factor)

        
        loss_tgt = cost_target
        loss_pred = torch.logsumexp(pose_sample_logweights, dim=0)  # (num_obj, )

        loss_pose = loss_tgt + loss_pred  # (num_obj, )
        loss_pose[torch.isnan(loss_pose)] = 0
        loss_pose = loss_pose.mean() / self.norm_factor

        return loss_pose.mean()

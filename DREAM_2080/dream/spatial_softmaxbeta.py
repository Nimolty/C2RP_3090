# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np

# This code is adapted from
# https://gitlab-master.nvidia.com/pmolchanov/lpr-3d-hand-pose-rgb-demo/blob/master/handpose/models/image_heatmaps_pose2dZrel_softargmax_slim.py


class SoftArgmaxPavlo(torch.nn.Module):
    def __init__(self, n_keypoints=5, learned_beta=False, initial_beta=25.0):
        super(SoftArgmaxPavlo, self).__init__()
        self.avgpool = torch.nn.AvgPool2d(7, stride=1, padding=3)
        if learned_beta:
            self.beta = torch.nn.Parameter(torch.ones(n_keypoints) * initial_beta)
        else:
            self.beta = (torch.ones(n_keypoints) * initial_beta).cuda()

    def forward(self, heatmaps, size_mult=1.0):

        epsilon = 1e-8
        bch, ch, n_row, n_col = heatmaps.size()
        n_kpts = ch

        beta = self.beta

        # input has the shape (#bch, n_kpts+1, img_sz[0], img_sz[1])
        # +1 is for the Zrel
        heatmaps2d = heatmaps[:, :n_kpts, :, :]
        heatmaps2d = self.avgpool(heatmaps2d)

        # heatmaps2d has the shape (#bch, n_kpts, img_sz[0]*img_sz[1])
        heatmaps2d = heatmaps2d.contiguous().view(bch, n_kpts, -1)

        # getting the max value of the maps across each 2D matrix
        map_max = torch.max(heatmaps2d, dim=2, keepdim=True)[0]

        # reducing the max from each map
        # this will make the max value zero and all other values
        # will be negative.
        # max_reduced_maps has the shape (#bch, n_kpts, img_sz[0]*img_sz[1])
        heatmaps2d = heatmaps2d - map_max

        beta_ = beta.view(1, n_kpts, 1).repeat(1, 1, n_row * n_col)
        # due to applying the beta value, the non-max values will be further
        # pushed towards zero after applying the exp function
        exp_maps = torch.exp(beta_ * heatmaps2d)
        # normalizing the exp_maps by diving it to the sum of elements
        # exp_maps_sum has the shape (#bch, n_kpts, 1)
        exp_maps_sum = torch.sum(exp_maps, dim=2, keepdim=True)
        exp_maps_sum = exp_maps_sum.view(bch, n_kpts, 1, 1)
        normalized_maps = exp_maps.view(bch, n_kpts, n_row, n_col) / (
            exp_maps_sum + epsilon
        )

        col_vals = torch.arange(0, n_col) * size_mult
        col_repeat = col_vals.repeat(n_row, 1)
        col_idx = col_repeat.view(1, 1, n_row, n_col).cuda()
        # col_mat gives a column measurement matrix to be used for getting
        # 'x'. It is a matrix where each row has the sequential values starting
        # from 0 up to n_col-1:
        # 0,1,2, ..., n_col-1
        # 0,1,2, ..., n_col-1
        # 0,1,2, ..., n_col-1

        row_vals = torch.arange(0, n_row).view(n_row, -1) * size_mult
        row_repeat = row_vals.repeat(1, n_col)
        row_idx = row_repeat.view(1, 1, n_row, n_col).cuda()
        # row_mat gives a row measurement matrix to be used for getting 'y'.
        # It is a matrix where each column has the sequential values starting
        # from 0 up to n_row-1:
        # 0,0,0, ..., 0
        # 1,1,1, ..., 1
        # 2,2,2, ..., 2
        # ...
        # n_row-1, ..., n_row-1

        col_idx = Variable(col_idx, requires_grad=False)
        weighted_x = normalized_maps * col_idx.float()
        weighted_x = weighted_x.view(bch, n_kpts, -1)
        x_vals = torch.sum(weighted_x, dim=2)

        row_idx = Variable(row_idx, requires_grad=False)
        weighted_y = normalized_maps * row_idx.float()
        weighted_y = weighted_y.view(bch, n_kpts, -1)
        y_vals = torch.sum(weighted_y, dim=2)

        out = torch.stack((x_vals, y_vals), dim=2)

        return out

class SpatialSoftmax(torch.nn.Module):
    def __init__(self, height, width, channel, temperature=None, data_format='NCHW'):
        super(SpatialSoftmax, self).__init__()
        self.data_format = data_format
        self.height = height
        self.width = width
        self.channel = channel

        if temperature:
            self.temperature = Parameter(torch.ones(1)*temperature)
        else:
            self.temperature = 1.

        pos_x, pos_y = np.meshgrid(
                np.linspace(-1., 1., self.height),
                np.linspace(-1., 1., self.width)
                )
        pos_x = torch.from_numpy(pos_x.reshape(self.height*self.width)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self.height*self.width)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

    def forward(self, feature):
        # Output:
        #   (N, C*2) x_0 y_0 ...
        print(feature.shape)
        if self.data_format == 'NHWC':
            feature = feature.transpose(1, 3).tranpose(2, 3).view(-1, self.height*self.width)
        else:
            feature = feature.view(-1, self.height*self.width)

        softmax_attention = F.softmax(feature/self.temperature, dim=-1)
        expected_x = torch.sum(self.pos_x*softmax_attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y*softmax_attention, dim=1, keepdim=True)
        expected_xy = torch.cat([expected_x, expected_y], 1)
        feature_keypoints = expected_xy.view(-1, self.channel*2)

        return expected_xy
import random




# def soft_argmax(x):
#
#     alpha = 10000.0
#     N,C,L = x.shape
#     soft_max = F.softmax(x*alpha,dim=2)
#     soft_max = soft_max.view(x.shape)
#     indices_kernel = torch.arange(start=0, end=L).unsqueeze(0)
#     conv = soft_max*indices_kernel
#     indices = conv.sum(2)
#     return indices
#
#
# # if __name__ == "__main__":
# #     from torch.autograd import Variable
# #
# #     data = Variable(torch.zeros([16,7,100,100]))
# #     for i in range(16):
# #         for j in range(7):
# #             x = random.randint(0,99)
# #             y = random.randint(0,99)
# #             data[i][j][x][y] = 1
# #             print(x,y)
# #     # layer = SpatialSoftmax(100,100,7, temperature=1)
# #     # maxx = layer(data[0])
# #     print(torch.argmax(data[0][0]))
# #     d1 = data.reshape(16,7,10000)
# #     maxx = soft_argmax(d1)
# #
# #     print(maxx)
# #
# # # if __name__ == '__main__':
# # #     data = torch.zeros([1,3, 3, 3])
# # #     data[0, 0, 0, 1] = 10
# # #     data[0, 1, 1, 1] = 10
# # #     data[0, 2, 1, 2] = 10
# # #     # layer = SpatialSoftmax(3, 3, 3, temperature=1)
# # #     data = data.reshape(1,3,9)
# # #     print(data,soft_argmax(data))
# #
# #
# import numpy as np
#
array = np.arange(25)
#
# np.random.shuffle(array)
#
array = np.reshape(array, (5, 5))
#
# print(array)
#
maxa = np.max(array)
#
# coorda = np.where(array == maxa)
#
# coorda = np.squeeze(coorda)
#
# print('numpy自带求最大值坐标： ', coorda)
#
fmwidth = 5

soft_argmax_x = np.zeros((fmwidth, fmwidth))

soft_argmax_y = np.zeros((fmwidth, fmwidth))

for i in range(1, fmwidth + 1, 1):
    soft_argmax_x[i - 1, :] = i / fmwidth

for j in range(1, fmwidth + 1, 1):
    soft_argmax_y[:, j - 1] = j / fmwidth

array_softmax = np.exp(array - maxa) / np.sum(np.exp(array - maxa))

xcoord = np.sum(np.multiply(array_softmax, soft_argmax_x))

ycoord = np.sum(np.multiply(array_softmax, soft_argmax_y))

print('softargmax求出的最大值坐标：', xcoord, ycoord)


class SoftArgmax(torch.nn.Module):
    def __init__(self, learned_beta=False, initial_beta=25.0):
        super(SoftArgmax, self).__init__()
        if learned_beta:
            self.beta = torch.nn.Parameter(initial_beta)
        else:
            self.beta = torch.tensor(initial_beta)

    def forward(self, heatmaps):
        B, C, H, W = heatmaps.shape
        beta = self.beta
        heat_indices = np.indices((H, W), dtype=np.float32).transpose(1, 2, 0)
        heat_indices_x = torch.tensor(heat_indices[..., 1]).reshape(1, -1)
        heat_indices_y = torch.tensor(heat_indices[..., 0]).reshape(1, -1)
        heatmaps = heatmaps.reshape(B, C, -1)
        heat_volume_x = F.softmax(heatmaps * beta, dim=2) * heat_indices_x
        heat_volume_y = F.softmax(heatmaps * beta, dim=2) * heat_indices_y

        heat_x = torch.sum(heat_volume_x, 2).unsqueeze(-1)
        heat_y = torch.sum(heat_volume_y, 2).unsqueeze(-1)
        heat_2d = torch.cat((heat_x, heat_y), -1)
        return heat_2d

def softargg(x):
    h,w = x.shape
    max_e = torch.max(x.reshape(h*w),dim = -1).values
    soft_max_x = torch.zeros_like(x)
    soft_max_y = torch.zeros_like(x)
    for i in range(h):
        soft_max_x[i, :] = i / h
    for j in range(w):
        soft_max_y[:, j] = j / w
    array_softmax = torch.exp(x - max_e) / torch.sum(torch.exp(x - max_e))
    xcoord = torch.sum(array_softmax* soft_max_x)
    ycoord = torch.sum(array_softmax* soft_max_y)
    return xcoord,ycoord

if __name__ == "__main__":
    from torch.autograd import Variable

    data = Variable(torch.zeros([16,7,100,100]))
    for i in range(16):
        for j in range(7):
            x = random.randint(0,99)
            y = random.randint(0,99)
            data[i][j][x][y] = 1
            print(x,y)
            print(softargg(data[i][j]))
    layer = SpatialSoftmax(100,100,7, temperature=1)
    # data = Variable(torch.zeros([1, 4, 4, 4]))
    # data[0, 0, 0, 1] = 10000
    # data[0, 1, 1, 1] = 10000
    # data[0, 2, 1, 2] = 1000
    # layer = SpatialSoftmax(4,4,4, temperature=1)
    # maxx = layer(data)
    # # print(torch.argmax(data[0][0]))
    # print(data, maxx)
    # print(softargshit(data[0][0]))
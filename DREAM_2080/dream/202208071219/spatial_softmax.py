import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np


class SoftArgmax(torch.nn.Module):
    def __init__(self, learned_beta=False, initial_beta=25.0):
        super(SoftArgmax, self).__init__()
        if learned_beta:
            self.beeta = torch.nn.Parameter(initial_beta)
        else:
            self.beeta = torch.tensor(initial_beta)

    def forward(self, heatmaps):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        B, C, H, W = heatmaps.shape
        beeta = self.beeta
        heat_indices = np.indices((H, W), dtype=np.float32).transpose(1, 2, 0)
        heat_indices_x = torch.tensor(heat_indices[..., 1]).reshape(1, -1).to(device)
        heat_indices_y = torch.tensor(heat_indices[..., 0]).reshape(1, -1).to(device)
        heatmaps = heatmaps.reshape(B, C, -1).to(device)
        heat_volume_x = F.softmax(heatmaps * beeta, dim=2) * heat_indices_x
        heat_volume_y = F.softmax(heatmaps * beeta, dim=2) * heat_indices_y

        heat_x = torch.sum(heat_volume_x, 2).unsqueeze(-1).to(device)
        heat_y = torch.sum(heat_volume_y, 2).unsqueeze(-1).to(device)
        heat_2d = torch.cat((heat_y, heat_x), -1).to(device)
        return heat_2d

import random
if __name__ == "__main__":
    from torch.autograd import Variable

    data = Variable(torch.zeros([16,7,100,100]))
    for i in range(16):
        for j in range(7):
            x = random.randint(0,99)
            y = random.randint(0,99)
            data[i][j][x][y] = 1
            print(x,y)
    layer = SoftArgmax()
    print(layer(data))
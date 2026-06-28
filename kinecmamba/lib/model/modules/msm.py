import torch
import torch.nn as nn

class PerJointDelta(nn.Module):
    def __init__(self, num_joints, channels, num_groups=4):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, num_groups, channels, 1, num_joints))

    def forward(self, dts, B, K, D, H, W):
        dts = dts.view(B, K, D, H, W)
        dts = dts + self.bias
        dts = dts.view(B, K*D, H*W)
        return dts

import torch
from torch import nn

class BinaryFeature(nn.Module):

    def __init__(self):
        super(BinaryFeature, self).__init__()

    def forward(self, treatment):
        return torch.cat([treatment, 1.0-treatment], dim=1)

def get_ihdp_feature():
    backdoor_feature = nn.Sequential(nn.Linear(25, 200),
                                     nn.ReLU(),
                                     nn.Linear(200, 200),
                                     nn.ReLU())
    treatment_feature = BinaryFeature()
    return backdoor_feature, treatment_feature


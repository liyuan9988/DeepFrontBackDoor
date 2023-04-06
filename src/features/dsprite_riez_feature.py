from torch import nn
from torch.nn.utils import spectral_norm
import torch

class Learner(nn.Module):

    def __init__(self):
        super().__init__()
        n_common = 200
        self.common = nn.Sequential(spectral_norm(nn.Linear(64 * 64 + 2, 1024)),
                                  nn.ReLU(),
                                  spectral_norm(nn.Linear(1024, 512)),
                                  nn.ReLU(),
                                  nn.BatchNorm1d(512),
                                  spectral_norm(nn.Linear(512, 128)),
                                  nn.ReLU(),
                                  spectral_norm(nn.Linear(128, 32)),
                                  nn.BatchNorm1d(32),
                                  nn.Tanh())
        self.riesz_nn = nn.Sequential(nn.Linear(32, 1))
        self.reg_nn = nn.Sequential(nn.Linear(32, 32), nn.ReLU(),
                                    nn.Linear(32, 32), nn.ReLU(),
                                    nn.Linear(32, 1))


    def forward(self, x):
        feats = self.common(x)
        riesz = self.riesz_nn(feats)
        reg = self.reg_nn(feats)
        return torch.cat([reg, riesz], dim=1)


def get_dsprite_feature_riez(**kwargs):
    return Learner()


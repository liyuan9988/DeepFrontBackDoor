from torch import nn
from torch.nn.utils import spectral_norm

from src.utils.pytorch_linear_reg_utils import add_const_col


class AddConstCol(nn.Module):

    def __init__(self):
        super(AddConstCol, self).__init__()

    def forward(self, X):
        return add_const_col(X)


def get_dsprite_feature():
    backdoor_feature = nn.Sequential(nn.Linear(2, 36),
                                     nn.ReLU(),
                                     nn.Linear(36, 5),
                                     nn.ReLU(),
                                     AddConstCol())

    treatment_feature = nn.Sequential(spectral_norm(nn.Linear(64 * 64, 1024)),
                                      nn.ReLU(),
                                      spectral_norm(nn.Linear(1024, 512)),
                                      nn.ReLU(),
                                      nn.BatchNorm1d(512),
                                      spectral_norm(nn.Linear(512, 128)),
                                      nn.ReLU(),
                                      spectral_norm(nn.Linear(128, 32)),
                                      nn.BatchNorm1d(32),
                                      nn.Tanh(),
                                      AddConstCol())

    return backdoor_feature, treatment_feature


def get_dsprite_feature_att():
    treatment_feature_1st = nn.Sequential(spectral_norm(nn.Linear(64 * 64, 1024)),
                                          nn.ReLU(),
                                          spectral_norm(nn.Linear(1024, 512)),
                                          nn.ReLU(),
                                          nn.BatchNorm1d(512),
                                          spectral_norm(nn.Linear(512, 128)),
                                          nn.ReLU(),
                                          spectral_norm(nn.Linear(128, 32)),
                                          nn.BatchNorm1d(32),
                                          nn.Tanh(),
                                          AddConstCol())

    treatment_feature_2nd = nn.Sequential(spectral_norm(nn.Linear(64 * 64, 1024)),
                                          nn.ReLU(),
                                          spectral_norm(nn.Linear(1024, 512)),
                                          nn.ReLU(),
                                          nn.BatchNorm1d(512),
                                          spectral_norm(nn.Linear(512, 128)),
                                          nn.ReLU(),
                                          spectral_norm(nn.Linear(128, 32)),
                                          nn.BatchNorm1d(32),
                                          nn.Tanh(),
                                          AddConstCol())

    frontdoor_feature = nn.Sequential(nn.Linear(1, 5),
                                      nn.ReLU(),
                                      nn.Linear(5, 3),
                                      AddConstCol())

    return frontdoor_feature, treatment_feature_1st, treatment_feature_2nd

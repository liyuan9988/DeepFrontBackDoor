import torch
from torch import nn

from itertools import chain, combinations
from itertools import combinations_with_replacement as combinations_w_r

def _combinations(n_features, degree, interaction_only):
        comb = (combinations if interaction_only else combinations_w_r)
        return chain.from_iterable(comb(range(n_features), i)
                                   for i in range(0, degree + 1))


class Learner(nn.Module):

    def __init__(self, n_t=178, n_hidden=100, p=0, degree=0, interaction_only=True):
        super().__init__()
        n_common = 200
        self.monomials = list(_combinations(n_t, degree, interaction_only))
        self.common = nn.Sequential(nn.Dropout(p=p), nn.Linear(n_t, n_common), nn.ELU(),
                                    nn.Dropout(p=p), nn.Linear(n_common, n_common), nn.ELU(),
                                    nn.Dropout(p=p), nn.Linear(n_common, n_common), nn.ELU())
        self.riesz_nn = nn.Sequential(nn.Dropout(p=p), nn.Linear(n_common, 1))
        self.riesz_poly = nn.Sequential(nn.Linear(len(self.monomials), 1))
        self.reg_nn0 = nn.Sequential(nn.Dropout(p=p), nn.Linear(n_common, n_hidden), nn.ELU(),
                                     nn.Dropout(p=p), nn.Linear(n_hidden, n_hidden), nn.ELU(),
                                     nn.Dropout(p=p), nn.Linear(n_hidden, 1))
        self.reg_nn1 = nn.Sequential(nn.Dropout(p=p), nn.Linear(n_common, n_hidden), nn.ELU(),
                                     nn.Dropout(p=p), nn.Linear(n_hidden, n_hidden), nn.ELU(),
                                     nn.Dropout(p=p), nn.Linear(n_hidden, 1))
        self.reg_poly = nn.Sequential(nn.Linear(len(self.monomials), 1))

    def forward(self, x):
        poly = torch.cat([torch.prod(x[:, t], dim=1, keepdim=True)
                          for t in self.monomials], dim=1)
        feats = self.common(x)
        riesz = self.riesz_nn(feats) + self.riesz_poly(poly)
        reg = self.reg_nn0(feats) * (1 - x[:, [0]]) + self.reg_nn1(feats) * x[:, [0]] + self.reg_poly(poly)
        return torch.cat([reg, riesz], dim=1)


def get_acic_feature_riez(**kwargs):
    return Learner()
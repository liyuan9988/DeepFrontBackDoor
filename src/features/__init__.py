from .ihdp_feature import get_ihdp_feature
from .dsprite_feature import get_dsprite_feature, get_dsprite_feature_att
from .acic_feature import get_acic_feature


def get_nn_feature_ate(name):
    if name == "ihdp":
        return get_ihdp_feature()
    elif name == "dsprite":
        return get_dsprite_feature()
    elif name == "acic":
        return get_acic_feature()
    else:
        raise ValueError(f"{name} not known")


def get_nn_feature_att(name):
    if name == "dsprite":
        return get_dsprite_feature_att()
    else:
        raise ValueError(f"{name} not known")
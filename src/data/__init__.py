from typing import Dict, Any

from src.data.ihdp_data import generate_ihdp_ate
from src.data.dsprite import generate_ate_dsprite, generate_dsprite_att
from src.data.acic_data import generate_acic_ate
from src.data.data_class import BackDoorTrainDataSet, ATETestDataSet


def generate_ate_data(data_config: Dict[str, Any], rand_seed: int) -> (BackDoorTrainDataSet, ATETestDataSet):
    data_name = data_config["name"]
    if data_name == "ihdp":
        return generate_ihdp_ate(rand_seed)
    elif data_name == "dsprite":
        return generate_ate_dsprite(rand_seed=rand_seed, **data_config)
    elif data_name == "acic":
        return generate_acic_ate(rand_seed=rand_seed, **data_config)
    else:
        raise ValueError(f"{data_name} not known")


def generate_att_data(data_config: Dict[str, Any], rand_seed: int) -> (BackDoorTrainDataSet, ATETestDataSet):
    data_name = data_config["name"]
    if data_name == "dsprite":
        return generate_dsprite_att(rand_seed=rand_seed, **data_config)
    else:
        raise ValueError(f"{data_name} not known")

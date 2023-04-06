from typing import Dict, Any
import numpy as np
from pathlib import Path
import logging

from src.data import generate_ate_data
from src.features import get_nn_feature_ate
from src.utils.pytorch_linear_reg_utils import *

logger = logging.getLogger()

def evaluate_backdoor_ate_nn_pure_reg(data_config: Dict[str, Any], model_param: Dict[str, Any], dump_folder: Path,
                                    random_seed: int = 42, verbose: int = 0):
    train_data, test_data = generate_ate_data(data_config, random_seed)
    data_name = data_config["name"]
    backdoor_feature_mdl, treatment_feature_mdl = get_nn_feature_ate(data_name)


    try:
        treatment_opt = torch.optim.Adam(treatment_feature_mdl.parameters())
    except:
        treatment_opt = None

    treatment_t = torch.tensor(train_data.treatment, dtype=torch.float32)
    outcome_t = torch.tensor(train_data.outcome, dtype=torch.float32)
    reg = model_param.get("reg", 0.001)
    niter = model_param.get("niter", 100)

    treatment_feature_mdl.train(True)
    for i in range(niter):
        if treatment_opt is not None:
            treatment_opt.zero_grad()
        treatment_feature = treatment_feature_mdl(treatment_t)

        loss = linear_reg_loss(outcome_t, treatment_feature, reg)
        loss.backward()
        if treatment_opt is not None:
            treatment_opt.step()
        if verbose > 0:
            logger.info(loss)


    treatment_feature_mdl.train(False)
    treatment_feature = treatment_feature_mdl(treatment_t)
    weight = fit_linear(outcome_t, treatment_feature, reg)

    # get treatmented
    n_test = test_data.treatment.shape[0]
    test_treatment_t = torch.tensor(test_data.treatment, dtype=torch.float32)
    test_treatment_feature = treatment_feature_mdl(test_treatment_t)
    pred = linear_reg_pred(test_treatment_feature, weight).detach().numpy()
    np.savetxt(dump_folder.joinpath(f"pred.{random_seed}.csv"), pred)
    loss = np.sum((test_data.structural - pred) ** 2) / test_data.structural.shape[0]
    return loss

from typing import Dict, Any
import numpy as np
from pathlib import Path
import logging

from src.data import generate_ate_data, generate_att_data
from src.features import get_nn_feature_ate, get_nn_feature_att
from src.utils.pytorch_linear_reg_utils import *

logger = logging.getLogger()


def evaluate_backdoor_ate_nn(data_config: Dict[str, Any], model_param: Dict[str, Any], dump_folder: Path,
                             random_seed: int = 42, verbose: int = 0):
    train_data, test_data = generate_ate_data(data_config, random_seed)
    data_name = data_config["name"]
    backdoor_feature_mdl, treatment_feature_mdl = get_nn_feature_ate(data_name)

    backdoor_t = torch.tensor(train_data.backdoor, dtype=torch.float32)
    treatment_t = torch.tensor(train_data.treatment, dtype=torch.float32)
    outcome_t = torch.tensor(train_data.outcome, dtype=torch.float32)
    reg = model_param.get("reg", 0.001)
    niter = model_param.get("niter", 100)

    backdoor_weight_decay = model_param.get("backdoor_weight_decay", 0.0)
    backdoor_opt = torch.optim.Adam(backdoor_feature_mdl.parameters(), weight_decay=backdoor_weight_decay)
    try:
        treatment_weight_decay = model_param.get("treatment_weight_decay", 0.0)
        treatment_opt = torch.optim.Adam(treatment_feature_mdl.parameters(), weight_decay=treatment_weight_decay)
    except:
        treatment_opt = None

    backdoor_feature_mdl.train(True)
    treatment_feature_mdl.train(True)
    for i in range(niter):
        backdoor_opt.zero_grad()
        if treatment_opt is not None:
            treatment_opt.zero_grad()
        backdoor_feature = backdoor_feature_mdl(backdoor_t)
        treatment_feature = treatment_feature_mdl(treatment_t)

        feature = outer_prod(treatment_feature, backdoor_feature).reshape((backdoor_feature.shape[0], -1))
        loss = linear_reg_loss(outcome_t, feature, reg)
        loss.backward()
        backdoor_opt.step()
        if treatment_opt is not None:
            treatment_opt.step()
        if verbose > 0:
            logger.info(loss)

    backdoor_feature_mdl.train(False)
    treatment_feature_mdl.train(False)
    backdoor_feature = backdoor_feature_mdl(backdoor_t)
    treatment_feature = treatment_feature_mdl(treatment_t)
    feature = outer_prod(treatment_feature, backdoor_feature).reshape((backdoor_feature.shape[0], -1))
    weight = fit_linear(outcome_t, feature, reg)

    # get treatmented
    n_test = test_data.treatment.shape[0]
    mean_bakcdoor_feature = torch.mean(backdoor_feature, dim=0).repeat(n_test, 1)
    test_treatment_t = torch.tensor(test_data.treatment, dtype=torch.float32)
    test_treatment_feature = treatment_feature_mdl(test_treatment_t)
    feature = outer_prod(test_treatment_feature, mean_bakcdoor_feature).reshape((n_test, -1))
    pred = linear_reg_pred(feature, weight).detach().numpy()
    np.savetxt(dump_folder.joinpath(f"pred.{random_seed}.csv"), pred)
    loss = np.sum((test_data.structural - pred) ** 2) / test_data.structural.shape[0]
    return loss


def evaluate_frontdoor_att_nn(data_config: Dict[str, Any], model_param: Dict[str, Any], dump_folder: Path,
                              random_seed: int = 42, verbose: int = 0):
    train_data, test_data = generate_att_data(data_config, random_seed)
    data_name = data_config["name"]
    frontdoor_feature_mdl, treatment_feature_1st_mdl, treatment_feature_2nd_mdl = get_nn_feature_att(data_name)

    frontdoor_feature_opt = torch.optim.Adam(frontdoor_feature_mdl.parameters())
    treatment_1st_opt = torch.optim.Adam(treatment_feature_1st_mdl.parameters())
    treatment_2nd_opt = torch.optim.Adam(treatment_feature_2nd_mdl.parameters())

    frontdoor_t = torch.tensor(train_data.frontdoor, dtype=torch.float32)
    treatment_t = torch.tensor(train_data.treatment, dtype=torch.float32)
    outcome_t = torch.tensor(train_data.outcome, dtype=torch.float32)
    reg = model_param.get("reg", 0.001)
    niter_stage1 = model_param.get("niter_stage1", 100)

    frontdoor_feature_mdl.train(True)
    treatment_feature_1st_mdl.train(True)
    for i in range(niter_stage1):
        frontdoor_feature_opt.zero_grad()
        treatment_1st_opt.zero_grad()
        frontdoor_feature = frontdoor_feature_mdl(frontdoor_t)
        treatment_feature = treatment_feature_1st_mdl(treatment_t)

        feature = outer_prod(treatment_feature, frontdoor_feature).reshape((frontdoor_feature.shape[0], -1))
        loss = linear_reg_loss(outcome_t, feature, reg)
        loss.backward()
        frontdoor_feature_opt.step()
        treatment_1st_opt.step()
        if verbose > 0:
            logger.info(loss)

    frontdoor_feature_mdl.train(False)
    treatment_feature_1st_mdl.train(False)
    frontdoor_feature = frontdoor_feature_mdl(frontdoor_t)
    treatment_feature = treatment_feature_1st_mdl(treatment_t)
    feature = outer_prod(treatment_feature, frontdoor_feature).reshape((frontdoor_feature.shape[0], -1))
    stage1_weight = fit_linear(outcome_t, feature, reg)


    frontdoor_feature_no_bias = frontdoor_feature_mdl(frontdoor_t).detach()
    treatment_feature_2nd_mdl.train(True)
    niter_stage2 = model_param.get("niter_stage2", 100)
    for i in range(niter_stage2):
        treatment_2nd_opt.zero_grad()
        treatment_feature = treatment_feature_2nd_mdl(treatment_t)
        loss = linear_reg_loss(frontdoor_feature_no_bias, treatment_feature, reg)
        loss.backward()
        treatment_2nd_opt.step()
        if verbose > 0:
            logger.info(loss)

    treatment_feature_2nd_mdl.train(False)
    treatment_feature = treatment_feature_2nd_mdl(treatment_t)
    stage2_weight = fit_linear(frontdoor_feature_no_bias, treatment_feature, reg)


    # get treatmented
    n_test = test_data.target.shape[0]
    treatment_cf_t = torch.tensor(test_data.treatment_cf, dtype=torch.float32)
    treatment_ac_t = torch.tensor(test_data.treatment_ac, dtype=torch.float32)
    pred_front_feature = linear_reg_pred(treatment_feature_2nd_mdl(treatment_cf_t),
                                         stage2_weight)
    feature = outer_prod(treatment_feature_1st_mdl(treatment_ac_t),
                         pred_front_feature).reshape((n_test, -1))

    pred = linear_reg_pred(feature, stage1_weight).detach().numpy()
    np.savetxt(dump_folder.joinpath(f"pred.{random_seed}.csv"), pred)
    loss = np.sum((test_data.target - pred) ** 2) / n_test
    return loss

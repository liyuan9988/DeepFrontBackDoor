from typing import Dict, Any
from pathlib import Path
import os
import numpy as np
from joblib import Parallel, delayed
import logging

from src.utils import grid_search_dict
from src.models.kernel_feature import evaluate_backdoor_ate_rkhs, evaluate_frontdoor_att_rkhs
from src.models.nn_feature import evaluate_backdoor_ate_nn, evaluate_frontdoor_att_nn
from src.models.nn_feature_pure_reg import evaluate_backdoor_ate_nn_pure_reg
from src.models.riesz_net import evaluate_backdoor_ate_nn_rieznet

logger = logging.getLogger()


def get_run_func_ate(mdl_name: str):
    if mdl_name == "rkhs":
        return evaluate_backdoor_ate_rkhs
    elif mdl_name == "nn":
        return evaluate_backdoor_ate_nn
    elif mdl_name == "nn_reg":
        return evaluate_backdoor_ate_nn_pure_reg
    elif mdl_name == "riez":
        return evaluate_backdoor_ate_nn_rieznet
    else:
        raise ValueError(f"name {mdl_name} is not known")


def get_run_func_att(mdl_name: str):
    if mdl_name == "rkhs":
        return evaluate_frontdoor_att_rkhs
    elif mdl_name == "nn":
        return evaluate_frontdoor_att_nn
    else:
        raise ValueError(f"name {mdl_name} is not known")

def experiments(configs: Dict[str, Any],
                dump_dir: Path,
                num_cpus: int,
                prob: str):

    data_config = configs["data"]
    model_config = configs["model"]
    n_repeat: int = configs["n_repeat"]

    if num_cpus <= 1 and n_repeat <= 1:
        verbose: int = 2
    else:
        verbose: int = 0

    if prob == "ate":
        run_func = get_run_func_ate(model_config["name"])
    elif prob == "att":
        run_func = get_run_func_att(model_config["name"])

    for dump_name, env_param in grid_search_dict(data_config):
        one_dump_dir = dump_dir.joinpath(dump_name)
        os.mkdir(one_dump_dir)
        for mdl_dump_name, mdl_param in grid_search_dict(model_config):
            if mdl_dump_name != "one":
                one_mdl_dump_dir = one_dump_dir.joinpath(mdl_dump_name)
                os.mkdir(one_mdl_dump_dir)
            else:
                one_mdl_dump_dir = one_dump_dir
            tasks = [delayed(run_func)(env_param, mdl_param, one_mdl_dump_dir, idx, verbose) for idx in range(n_repeat)]
            res = Parallel(n_jobs=num_cpus)(tasks)
            np.savetxt(one_mdl_dump_dir.joinpath("result.csv"), np.array(res))
        logger.critical(f"{dump_name} ended")

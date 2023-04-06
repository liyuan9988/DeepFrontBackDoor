from typing import Dict, Any, Tuple
import numpy as np
from pathlib import Path

from src.data import generate_ate_data, generate_att_data
from src.utils.kernel_func import AbsKernel, GaussianKernel, BinaryKernel
from src.utils import cal_loocv, cal_loocv_emb


def get_kernel_func_backdoor(data_name: str) -> Tuple[AbsKernel, AbsKernel]:
    if data_name in ["ihdp", "acic"]:
        return GaussianKernel(), BinaryKernel()
    elif data_name == "dsprite":
        return GaussianKernel(), GaussianKernel()
    else:
        raise ValueError(f"data_name {data_name} unknown")


def get_kernel_func_frontdoor(data_name: str) -> Tuple[AbsKernel, AbsKernel]:
    if data_name == "dsprite":
        return GaussianKernel(), GaussianKernel()
    else:
        raise ValueError(f"data_name {data_name} unknown")


def evaluate_backdoor_ate_rkhs(data_config: Dict[str, Any], model_param: Dict[str, Any], dump_folder: Path,
                               random_seed: int = 42, verbose: int = 0):
    train_data, test_data = generate_ate_data(data_config, random_seed)
    data_name = data_config["name"]
    backdoor_kernel_func, treatment_kernel_func = get_kernel_func_backdoor(data_name)
    backdoor_kernel_func.fit(train_data.backdoor, )
    treatment_kernel_func.fit(train_data.treatment, )

    treatment_kernel = treatment_kernel_func.cal_kernel_mat(train_data.treatment, train_data.treatment)
    backdoor_kernel = backdoor_kernel_func.cal_kernel_mat(train_data.backdoor, train_data.backdoor)
    n_data = treatment_kernel.shape[0]
    lam = model_param.get("lam", 0.001)
    if isinstance(lam, list):
        lam_score = [cal_loocv(treatment_kernel * backdoor_kernel, train_data.outcome, lam_one) for lam_one in lam]
        lam = lam[np.argmin(lam_score)]

    kernel_mat = treatment_kernel * backdoor_kernel + n_data * lam * np.eye(n_data)
    mean_backdoor_kernel = np.mean(backdoor_kernel, axis=0)
    test_kernel = treatment_kernel_func.cal_kernel_mat(train_data.treatment, test_data.treatment)
    test_kernel = test_kernel * mean_backdoor_kernel[:, np.newaxis]
    mat = np.linalg.solve(kernel_mat, test_kernel)
    predict = np.dot(mat.T, train_data.outcome)
    np.savetxt(dump_folder.joinpath(f"pred.{random_seed}.csv"), predict)
    loss = np.sum((test_data.structural - predict) ** 2) / test_data.structural.shape[0]
    return loss


def evaluate_frontdoor_att_rkhs(data_config: Dict[str, Any], model_param: Dict[str, Any], dump_folder: Path,
                                random_seed: int = 42, verbose: int = 0):
    train_data, test_data = generate_att_data(data_config, random_seed)
    data_name = data_config["name"]
    frontdoor_kernel_func, treatment_kernel_func = get_kernel_func_frontdoor(data_name)
    frontdoor_kernel_func.fit(train_data.frontdoor, )
    treatment_kernel_func.fit(train_data.treatment, )

    treatment_kernel = treatment_kernel_func.cal_kernel_mat(train_data.treatment, train_data.treatment)
    frontdoor_kernel = frontdoor_kernel_func.cal_kernel_mat(train_data.frontdoor, train_data.frontdoor)
    n_data = treatment_kernel.shape[0]
    lam1 = model_param.get("lam1", 0.001)
    if isinstance(lam1, list):
        lam_score = [cal_loocv(treatment_kernel * frontdoor_kernel, train_data.outcome, lam_one) for lam_one in lam1]
        lam1 = lam1[np.argmin(lam_score)]

    kernel_mat = treatment_kernel * frontdoor_kernel + n_data * lam1 * np.eye(n_data)
    weight = np.linalg.solve(kernel_mat, train_data.outcome)

    # conditonal embedding
    lam2 = model_param.get("lam2", 0.001)
    if isinstance(lam2, list):
        lam_score = [cal_loocv_emb(treatment_kernel, frontdoor_kernel, lam_one) for lam_one in lam2]
        lam2 = lam2[np.argmin(lam_score)]

    tmp = treatment_kernel_func.cal_kernel_mat(train_data.treatment, test_data.treatment_cf)
    test_kernel = frontdoor_kernel @ np.linalg.solve(treatment_kernel + n_data * lam2 * np.eye(n_data), tmp)
    test_kernel = test_kernel * treatment_kernel_func.cal_kernel_mat(train_data.treatment, test_data.treatment_ac)
    predict = np.dot(test_kernel.T, weight)
    np.savetxt(dump_folder.joinpath(f"pred.{random_seed}.csv"), predict)
    loss = np.sum((test_data.target - predict) ** 2) / test_data.target.shape[0]
    return loss

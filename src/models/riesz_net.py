from typing import Dict, Any
import numpy as np
from pathlib import Path
import logging
import torch
from src.data import generate_ate_data
from src.utils.rieznet import RieszNet
from src.features.dsprite_riez_feature import get_dsprite_feature_riez
from src.features.acic_riez_feature import get_acic_feature_riez
from sklearn.model_selection import train_test_split
logger = logging.getLogger()


def get_learner(name):
    if name == "dsprite":
        return get_dsprite_feature_riez()
    elif name == "acic":
        return get_acic_feature_riez()

def evaluate_backdoor_ate_nn_rieznet(data_config: Dict[str, Any], model_param: Dict[str, Any], dump_folder: Path,
                                    random_seed: int = 42, verbose: int = 0):
    methods = ['dr', 'direct', 'ips']
    srr = {'dr': True, 'direct': False, 'ips': True}
    dr_pred = []
    direct_pred = []
    ips_pred =[]
    train_data, test_data = generate_ate_data(data_config, random_seed)
    # Training params
    learner_lr = 1e-5
    learner_l2 = 1e-3
    learner_l1 = 0.0
    earlystop_rounds = 40  # how many epochs to wait for an out-of-sample improvement
    earlystop_delta = 1e-4
    target_reg = 1.0
    riesz_weight = 0.1

    bs = 64
    device = torch.cuda.current_device() if torch.cuda.is_available() else None
    print("GPU:", torch.cuda.is_available())

    drop_prob = 0.0  # dropout prob of dropout layers throughout notebook
    n_hidden = 100  # width of hidden layers throughout notebook

    X = np.concatenate([train_data.treatment, train_data.backdoor], axis=1)
    y = train_data.outcome
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    n_test = test_data.treatment.shape[0]
    if n_test == 2:
        n_test = 1 # for ATE calculation in binary treatment

    for i in range(n_test):
        learner = get_learner(data_config["name"])
        if n_test > 2:
            test_treament = torch.tensor(test_data.treatment[i, :], dtype=torch.float32)
            n_treatment = test_treament.shape[0]

            def potential_outcome_moment_fn(x, test_fn, device):
                n_data = x.shape[0]
                if torch.is_tensor(x):
                    with torch.no_grad():
                        t = torch.cat([test_treament.repeat(n_data, 1).to(device), x[:, n_treatment:]], dim=1)
                else:
                    t = np.concatenate([np.tile(test_data.treatment[i], (n_data, 1)), x[:, n_treatment:]], axis=1)
                return test_fn(t)

            agmm = RieszNet(learner, potential_outcome_moment_fn)
        else:
            def ate_moment_fn(x, test_fn, device):
                if torch.is_tensor(x):
                    with torch.no_grad():
                        t1 = torch.cat([torch.ones((x.shape[0], 1)).to(device), x[:, 1:]], dim=1)
                        t0 = torch.cat([torch.zeros((x.shape[0], 1)).to(device), x[:, 1:]], dim=1)
                else:
                    t1 = np.hstack([np.ones((x.shape[0], 1)), x[:, 1:]])
                    t0 = np.hstack([np.zeros((x.shape[0], 1)), x[:, 1:]])
                return test_fn(t1) - test_fn(t0)

            agmm = RieszNet(learner, ate_moment_fn)

        # Fast training
        agmm.fit(X_train, y_train, Xval=X_test, yval=y_test,
                 earlystop_rounds=2, earlystop_delta=earlystop_delta,
                 learner_lr=1e-4, learner_l2=learner_l2, learner_l1=learner_l1,
                 n_epochs=100, bs=bs, target_reg=target_reg,
                 riesz_weight=riesz_weight, optimizer='adam',
                 model_dir=str(Path.home()), device=device, verbose=0)
        # Fine tune
        agmm.fit(X_train, y_train, Xval=X_test, yval=y_test,
                 earlystop_rounds=earlystop_rounds, earlystop_delta=earlystop_delta,
                 learner_lr=learner_lr, learner_l2=learner_l2, learner_l1=learner_l1,
                 n_epochs=600, bs=bs, target_reg=target_reg,
                 riesz_weight=riesz_weight, optimizer='adam', warm_start=True,
                 model_dir=str(Path.home()), device=device, verbose=0)

        params = tuple(x for method in methods
                       for x in agmm.predict_avg_moment(X, y, model='earlystop', method=method, srr=srr[method]))

        dr_pred.append(params[0])
        direct_pred.append(params[3])
        ips_pred.append(params[6])

    np.savetxt(dump_folder.joinpath(f"pred.{random_seed}.dr.csv"), np.array(dr_pred))
    np.savetxt(dump_folder.joinpath(f"pred.{random_seed}.direct.csv"), np.array(direct_pred))
    np.savetxt(dump_folder.joinpath(f"pred.{random_seed}.ips.csv"), np.array(ips_pred))
    loss_dr = np.sum((test_data.structural - np.array(dr_pred)[:, np.newaxis]) ** 2) / test_data.structural.shape[0]
    loss_direct = np.sum((test_data.structural - np.array(direct_pred)[:, np.newaxis]) ** 2) / test_data.structural.shape[0]
    loss_ips = np.sum((test_data.structural - np.array(ips_pred)[:, np.newaxis]) ** 2) / test_data.structural.shape[0]
    return [loss_dr, loss_direct, loss_ips]


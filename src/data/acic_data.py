from filelock import FileLock
import numpy as np
import pathlib
import pickle
import pandas as pd
from sklearn import preprocessing

from src.data.data_class import BackDoorTrainDataSet, ATETestDataSet

DATA_PATH = pathlib.Path(__file__).resolve().parent.parent.parent.joinpath("data/")

def generate_acic_ate(rand_seed: int = 42,
                      standardize: bool = False,
                      outlier_alpha = 1.5,
                      **kwargs) -> (BackDoorTrainDataSet, ATETestDataSet):
    idx = ((rand_seed) % 101)
    with FileLock(DATA_PATH.joinpath("./data.lock")):
        ACIC_PATH = DATA_PATH.joinpath("ACIC")
        x_raw = pd.read_csv(ACIC_PATH.joinpath("x.csv"), index_col='sample_id', header=0, sep=',')
        folder, uid = pickle.load(open(ACIC_PATH.joinpath("simulation_file_list.pickle"), "rb"))[idx]
        output = pd.read_csv(ACIC_PATH.joinpath(f"{folder}/{uid}.csv"), index_col='sample_id', header=0, sep=',')

    dataset = x_raw.join(output, how='inner')
    t = dataset['z'].values
    y = dataset['y'].values
    x = dataset.values[:, :-2]

    #subsamplings
    Q1 = np.quantile(y, 0.25)
    Q3 = np.quantile(y, 0.75)
    IQR = Q3 - Q1
    valid_sample_flg = np.logical_and(y > Q1 - outlier_alpha * IQR, y < Q3 + outlier_alpha * IQR)
    t = t[valid_sample_flg]
    y = y[valid_sample_flg]
    x = x[valid_sample_flg]
    y = y - np.mean(y)


    if standardize:
        backdoor_normal_scalar = preprocessing.StandardScaler()
        x = backdoor_normal_scalar.fit_transform(x)

    train_data = BackDoorTrainDataSet(backdoor=x,
                                      outcome=y[:, np.newaxis],
                                      treatment=t[:, np.newaxis])

    test_data = ATETestDataSet(treatment=np.array([[0], [1]]),
                               structural=np.array([[0.0], [0.0]]))
    return train_data, test_data



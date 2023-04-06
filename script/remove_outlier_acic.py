import pandas as pd
import pathlib
import pickle
import numpy as np
import shutil

DATA_PATH = pathlib.Path(__file__).resolve().parent.parent.joinpath("data/")
ORG_ACIC_PATH = DATA_PATH.joinpath("ACIC")
NEW_ACIC_PATH = DATA_PATH.joinpath("ACIC_new")
outlier_alpha = 5.0

NEW_ACIC_PATH.mkdir(parents=True, exist_ok=True)
file_list = pickle.load(open(ORG_ACIC_PATH.joinpath("simulation_file_list.pickle"), "rb"))

for folder, uid in file_list:
    output = pd.read_csv(ORG_ACIC_PATH.joinpath(f"{folder}/{uid}.csv"), index_col='sample_id', header=0, sep=',')
    y = output['y'].values
    Q1 = np.quantile(y, 0.25)
    Q3 = np.quantile(y, 0.75)
    IQR = Q3 - Q1
    valid_sample_flg = np.logical_and(y > Q1 - outlier_alpha * IQR, y < Q3 + outlier_alpha * IQR)
    new_output = output.loc[valid_sample_flg, :]
    new_folder = NEW_ACIC_PATH.joinpath(f"{folder}/")
    if not new_folder.exists():
        new_folder.mkdir()
    new_output.to_csv(new_folder.joinpath(f"{uid}.csv"))


shutil.copy(ORG_ACIC_PATH.joinpath("x.csv"), NEW_ACIC_PATH.joinpath("x.csv"))
shutil.copy(ORG_ACIC_PATH.joinpath("params.csv"), NEW_ACIC_PATH.joinpath("params.csv"))
shutil.copy(ORG_ACIC_PATH.joinpath("simulation_file_list.pickle"),
            NEW_ACIC_PATH.joinpath("simulation_file_list.pickle"))



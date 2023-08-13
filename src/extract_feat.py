"""
This code presents a general way of feature extraction using the package tsfresh.
We encourage tailored feature definations.
Our feature defination is presented in the paper.
"""
import os
import pandas as pd
import numpy as np
import yaml

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--csv_name", type=str, required=True)
parser.add_argument("--data_type", type=str, required=True)
parser.add_argument("--dataset", type=str, default="aiops18")
parser.add_argument("--mini", action='store_false')

params = vars(parser.parse_args())

import gc
from tsfresh.utilities.dataframe_functions import roll_time_series
def df2rolled(file_path, window_size, mini=True):
    if not file_path.endswith(".csv"):
        raise ValueError(f"Cannot handle the {file_path} file")
    df = pd.read_csv(file_path).set_index("timestamp").sort_index()
    if mini: df = df.iloc[-300:, :]
    return roll_time_series(df, column_id="id", min_timeshift=window_size-1, max_timeshift=window_size-1)


from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute

def rolled2feat(rolled_path, metric_num, transformation = None):
    """
        transformation should be a customized function and returns extracted features in the format of DataFrame
    """
    if not rolled_path.endswith(".csv"): 
        raise ValueError(f"Cannot handle the {rolled_path} file")
    df_rolled = pd.read_csv(rolled_path)

    if "timestamp" in df_rolled: del df_rolled["timestamp"]
    if "Unnamed: 0" in df_rolled: del df_rolled["Unnamed: 0"]
    if "sort" in df_rolled: del df_rolled["sort"]
    
    target = pd.Series(df_rolled.label.values, index=pd.Index(df_rolled.id.values, name="id")).groupby(['id']).sum()
    target = pd.Series(np.where(target.values>0, 1, 0), index=target.index)
    del df_rolled["label"]
    assert len(list(df_rolled.columns)) == metric_num + 1

    if transformation is None:
        df_feat = extract_features(df_rolled, column_id="id", n_jobs=12)
        impute(df_feat)
    else:
        df_feat = transformation(df_rolled)
        assert isinstance(df_feat, pd.DataFrame)
        assert len(target) == len(df_feat)

    del df_rolled; gc.collect()
    
    df_feat["label"] = target.values

    return df_feat

def feat2Xy(feat_path, columns = None, filter_normal = False):
    if not feat_path.endswith(".csv"): 
        raise ValueError(f"Cannot handle the {feat_path} file")
    
    df_feat = pd.read_csv(feat_path)
    
    if filter_normal:
        df_feat = df_feat[df_feat.label == 0]
    
    y = np.array(df_feat.label.values)
    if columns is not None and isinstance(columns, (list, tuple)):
        df_feat = df_feat[columns]
    
    X = df_feat.to_numpy()
    print(X.shape, y.shape)
    return X, y

import json
import pickle
def main(dataset, csv_name, data_type, mini = False):

    with open(os.path.join("../data", dataset, "metadata.json")) as f:
        meta = json.load(f)
    metric_num = meta["metric_num"]
    with open(os.path.join("../data_ready", dataset, "config.yaml"), 'r') as f:
        configs = yaml.safe_load(f.read())
    window_size = configs["window_size"]

    rolled_df = df2rolled(f"../data/{dataset}/{csv_name}.csv", window_size, mini)
    rolled_df.to_csv(f"../data/{dataset}/rolled_{csv_name}.csv")
    
    df_feat = rolled2feat(f"../data/{dataset}/rolled_{csv_name}.csv", metric_num)
    df_feat.to_csv(f"../data/{dataset}/feat_{csv_name}.csv")

    X, y = feat2Xy(f"../data/{dataset}/feat_{csv_name}.csv")
    
    with open(f"../data/{dataset}/{data_type}/{csv_name}_X.pkl",'wb') as wf:
        pickle.dump(X, wf)
    with open(f"../data/{dataset}/{data_type}/{csv_name}_y.pkl",'wb') as wf:
        pickle.dump(y, wf)

if __name__ == "__main__":
    main(**params)
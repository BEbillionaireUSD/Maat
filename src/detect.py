import os
import yaml
from util import dump_configs, dump_scores
import logging

from dataload import DataHandler
from model.detection import Detector

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="aiops18")
parser.add_argument("--csv_name", type=str, required=True)
parser.add_argument("--pre_trained_detector", type=str, default=None)
params = vars(parser.parse_args())

import pickle
def get_Xy(dataset, data_type, csv_name):
    with open(f"../data/{dataset}/{data_type}/{csv_name}_X.pkl",'rb') as rf:
        X = pickle.load(rf)
    with open(f"../data/{dataset}/{data_type}/{csv_name}_y.pkl",'rb') as rf:
        y = pickle.load(rf)
    return X, y

def main(dataset, csv_name, pre_trained_detector=None):
    train_X, _ = get_Xy(dataset, "train", csv_name)
    valid_X, valid_y = get_Xy(dataset, "valid", csv_name)
    test_X, test_y = get_Xy(dataset, "test", csv_name)
   

    with open(os.path.join("../data_ready", dataset, "config.yaml"), 'r') as f:
        configs = yaml.safe_load(f.read())
    
    hash_id, _, result_dir = dump_configs(configs, dataset)
    logging.info("Start Experiment {}...".format(hash_id))

    detector = Detector(**configs)
    train_X = detector.scale(train_X, "train-test")
    test_X = detector.scale(test_X, "test")
    valid_X = detector.scale(valid_X, "test")

    if pre_trained_detector is None:
        top_k = configs["top_k"]
        indices = detector.feat_filter(valid_X, valid_y)
        cur_indices = indices[: top_k]
        detector.fit(train_X[:, cur_indices])
        eval_res = detector.evaluate(test_X[:, cur_indices], test_y)
    else:
        NotImplementedError
    
    dump_scores(result_dir, eval_res)


if __name__ == "__main__":
    if not os.path.exists("./result/detect"):
        os.makedirs("./result/detect", exist_ok=True)
    if not os.path.exists("./model_save/detect"):
        os.makedirs("./model_save/detect", exist_ok=True)
    main(**params)
import os
import yaml
import torch
import time
import logging
import shutil

# this repo
from model.prediction import SibylPredictor
from dataload import DataHandler
from util import dump_configs, seed_everything, dump_scores
from func_timeout import FunctionTimedOut

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="aiops18")
parser.add_argument("--mini", action='store_false')
parser.add_argument("--random_seed", type=int, default=42)
parser.add_argument("--pre_trained_predictor", type=str, default=None)


params = vars(parser.parse_args())


def main(dataset, random_seed, mini=True, pre_trained_predictor=None):
    if "yahoo" in dataset:
        with open("../data_ready/yahoo/config.yaml", 'r') as f:
            configs = yaml.safe_load(f.read())
    else:
        with open(os.path.join("../data_ready", dataset, "config.yaml"), 'r') as f:
            configs = yaml.safe_load(f.read())
    if mini:
        configs["epochs"] = 1
        configs["batch_size"] = 1
    
    hash_id, model_dir, result_dir = dump_configs(configs, dataset, "predict")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using {}...".format(device))
    
    logging.info("Start Experiment {}...".format(hash_id))
    seed_everything(random_seed)

    handler = DataHandler(dataset=dataset, mini=mini, **configs)
    handler.handle_data_dir(mini)

    ds, gdth, to_be_concat = handler.for_pred()

    sibyl_predictor = SibylPredictor(target_dim=handler.metric_num, freq=handler.freq, device=device, **configs)
    if pre_trained_predictor is None: #training
        t = time.time()
        sibyl_predictor.train(ds["train"], **configs)
        logging.info("Predictor Training Done: {:.4f}".format(time.time()-t))
        torch.save(sibyl_predictor.pred_net.state_dict(), os.path.join(model_dir, "net.pt"))
    else:
        sibyl_predictor.pred_net.load_state_dict(torch.load(pre_trained_predictor))
        sibyl_predictor.create_predictor()
        
    # Evaluation
    preds = {"val": None, "test": None}
    pred_res, preds["test"] = sibyl_predictor.evaluation(ds["test"], gdth["test"])
    dump_scores(result_dir, pred_res, sibyl_predictor.losses)


    logging.info("Data processing for detection...")
    aim_dir = os.path.join("../data_ready/", dataset, "preded")
    if os.path.exists(aim_dir): shutil.rmtree(aim_dir)
    os.makedirs(aim_dir)
    
    preds["val"] = sibyl_predictor.predict(ds["val"])
    for mode in ["val", "test"]:
        if preds[mode] is None: continue
        try:
            rolled_dfs, targets = handler.for_anticipate(preds=preds[mode], to_be_concat=to_be_concat[mode], mode=mode)
        except FunctionTimedOut:
            print("Too long for predicting", mode)
            continue
        os.mkdir(os.path.join(aim_dir, mode))
        for i, df in enumerate(rolled_dfs):
            print(df.head())
            df.to_csv(os.path.join(aim_dir, mode, str(i)+".csv"))
            with open(os.path.join(aim_dir, mode, str(i)+".txt"), 'w') as wf:
                wf.write(",".join([str(label) for label in targets[i]]))
    logging.info("Prediction Done...")
        

if __name__ == "__main__":
    if not os.path.exists("./result/predict"):
        os.makedirs("./result/predict")
    if not os.path.exists("./model_save/predict"):
        os.makedirs("./model_save/predict")
    main(**params)
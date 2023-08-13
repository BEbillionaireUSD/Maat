import pandas as pd
from gluonts.dataset.common import ListDataset
import numpy as np
import json
import os
import math
from tqdm import tqdm
import logging
from collections import defaultdict
from func_timeout import func_set_timeout

class DataHandler:
    def __init__(self, dataset, window_size, prediction_length, mini=True, **kwargs):
        self.dataset = dataset
        self.window_size = window_size
        self.prediction_length = prediction_length
        self.mini = mini
        self.sizes = {"train":0, "test":0, "valid":0}
        self.dfs = defaultdict(list)
        
        if dataset in ["aiops18", "hades"] or "yahoo" in dataset:
            self.__load_metadata()
        else:
            raise NotImplementedError
    
    def __load_metadata(self):
        if "yahoo" in self.dataset:
            with open("../data/yahoo/metadata.json") as f:
                meta = json.load(f)
        else:
            with open(os.path.join("../data", self.dataset, "metadata.json")) as f:
                meta = json.load(f)
        self.metric_num = meta["metric_num"]
        self.freq = meta["frequency"]
        self.step = 1 if "step" not in meta else meta["step"]
        self.history_length = self.window_size if "history_length" not in meta else meta["history_length"]
        logging.info(f"metric num {self.metric_num}, frequency {self.freq}, step {self.step}, hist {self.history_length}")
    
    def for_pred(self):
        logging.info("Data processing for prediction...")
        dataset_train = ListDataset([
                {"target": train_df.iloc[:, :-1].to_numpy().T, "start": train_df.index[0]} 
                for train_df in self.dfs["train"]
            ], freq=self.freq, one_dim_target=False)
        
        val_pred_inputs, val_pred_gdth, val_to_be_concat_pair = [], [], []
        for df in self.dfs["train"]:
            a, b, c = self.__generate_test_windows(df.iloc[:, :-1], labels=df["label"].values, history_length=0)
            val_pred_inputs.extend(a); val_pred_gdth.extend(b); val_to_be_concat_pair.extend(c)

        self.sizes["val"] = len(val_pred_inputs)
        dataset_val = ListDataset([
            {"target": pred_input.values.T, "start": pred_input.index[0]} 
            for pred_input in val_pred_inputs
        ], freq=self.freq, one_dim_target=False)

        test_pred_inputs, test_pred_gdth, test_to_be_concat_pair = [], [], []
        for df in self.dfs["test"]:
            a, b, c = self.__generate_test_windows(df.iloc[:, :-1], labels=df["label"].values, history_length=0)
            test_pred_inputs.extend(a); test_pred_gdth.extend(b); test_to_be_concat_pair.extend(c)
        
        self.sizes["test"] = len(test_pred_inputs)
        dataset_test = ListDataset([
            {"target": pred_input.values.T, "start": pred_input.index[0]} 
            for pred_input in test_pred_inputs
        ], freq=self.freq, one_dim_target=False)
        logging.info("Train size {} and Test size {}".format(self.sizes["val"], self.sizes["test"]))

        return {"train": dataset_train, "test": dataset_test, "val": dataset_val},  {"test":test_pred_gdth, "val": val_pred_gdth}, {"test": test_to_be_concat_pair, "val": val_to_be_concat_pair}

    def __generate_test_windows(self, df, labels, history_length):
        pred_inputs, pred_gdth, to_be_concat, chunk_labels = [], [], [], []

        i = 0
        while i + self.window_size + self.prediction_length  < len(labels):
            pred_inputs.append(df.iloc[i : i + history_length + self.window_size, :])
            pred_gdth.append(df.iloc[i + history_length + self.window_size: i + history_length + self.window_size + self.prediction_length, :])
            to_be_concat.append(df.iloc[i + history_length + self.prediction_length: i + history_length + self.window_size, :])
            chunk_labels.append(int(sum(labels[i + history_length + self.prediction_length: i + history_length + self.window_size + self.prediction_length]) > 0))
            i += self.step
        return pred_inputs, pred_gdth, zip(to_be_concat, chunk_labels)

    @func_set_timeout(48*3600)  
    def for_anticipate(self, *, preds, to_be_concat, mode="val"):
        print(mode, type(preds))
        df_mode = "train" if mode == "val" else mode
        #anticipate
        window_num_per_df = math.ceil(self.sizes[mode] *1.0/ len(self.dfs[df_mode]))
        
        rolled_dfs, targets = [], []

        with tqdm(total=self.sizes[mode]) as pbar:
            pbar.set_description('Predictions processing')
            X, y, ids = [], [], []
            for i, pred in enumerate(preds):
                #print(i / self.sizes[mode])
                (w_to, l) = to_be_concat[i]
                w = np.concatenate((w_to.values, pred.mean), axis=0)
                if i == 0: assert len(w) == self.window_size, f"{len(w)}, {self.window_size}"
                X.append(w); y.append(l); ids.extend(len(w)*[i])

                if len(X) == window_num_per_df or i+1 == self.sizes[mode]:
                    X = np.concatenate(X, axis=0) if len(X) > 1 else np.array(X)
                    df_rolled = pd.DataFrame(X, columns = [str(i) for i in range(self.metric_num)])
                    df_rolled["id"] = ids
                    rolled_dfs.append(df_rolled)
                    targets.append(y)
                    X, y, ids = [], [], []
                
                pbar.update(1)
            
        return rolled_dfs, targets
    
    def handle_data_dir(self, mini=True):

        shown = False
        for data_type in ["train", "test"]:
            for file in sorted(os.listdir(os.path.join("../data", self.dataset, data_type))):
                if not file.endswith(".csv"): continue
                df = pd.read_csv(os.path.join("../data", self.dataset, data_type, file)).set_index('timestamp').sort_index()
                if mini: df = df.iloc[:100, :]
                print(data_type, file, len(df))
                if "Unnamed 0" in df.columns: del df["Unnamed 0"]
                assert df.columns[-1] == "label"
                assert "id" not in df.columns
                assert len(df.columns) - 1 == self.metric_num
                if self.dataset in ["hades"] or "yahoo" in self.dataset:
                    df.index = pd.to_datetime(df.index.values, unit='s')
                new_index = pd.period_range(df.index[0], periods=len(df.index), freq=self.freq)
                df2 = df.set_index(new_index)
                if not shown and data_type == "test":
                    print(df2.head())
                    shown = True
                self.dfs[data_type].append(df2)
        print("train / test",
                sum([len(df) for df in self.dfs["train"]]), 
                sum([len(df) for df in self.dfs["test"]]))
        print("Test anomaly ratio:", 
                sum([sum(df["label"].values) for df in self.dfs["test"]]) / sum([len(df["label"].values) for df in self.dfs["test"]])
        )

                
                





import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import (
    RandomForestClassifier, 
    HistGradientBoostingClassifier,
) 
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import time

class Detector(object):
    def __init__(self, n_jobs=12, scale_method="standard", approach="IsolationForest", feat_filter="randomforest", c=0.02, **kwargs) -> None:

        if scale_method == "standard":
            self.scaler = StandardScaler()
        else:
            NotImplementedError

        if approach == "IsolationForest":
            self.clf = IsolationForest(random_state=42, contamination=c, verbose=1, n_jobs=n_jobs)
            self.approach = approach
        else:
            NotImplementedError
        
        if feat_filter == "xgboost":
            self.feat_rf = XGBClassifier(random_state=42, n_jobs=n_jobs)
        elif feat_filter == "randomforest":
            self.feat_rf = RandomForestClassifier(verbose=1, n_jobs=n_jobs)
        elif feat_filter == "lightgbt":
            self.feat_rf == HistGradientBoostingClassifier(verbose=1, n_jobs=n_jobs)
        else:
            NotImplementedError

    def scale(self, X, mode="train"):
        return {"train": self.scaler.fit, "test": self.scaler.transform, "train-test":self.scaler.fit_transform}[mode](X)
    
    def fit(self, X):
        X = self.scale(X, mode="train-test")
        t = time.time()
        self.clf.fit(X)
        logging.info("Taking {:.4f}s for training".format(time.time()-t))
            
    
    def predict(self, X):
        X = self.scale(X, mode="test")
        if self.approach == "IsolationForest":
            return np.where(self.clf.predict(X) == -1, 1, 0)
        return np.where(self.clf.predict(X) > 0, 1, 0)
        #return np.where(np.sum(preds, axis=0) > self.model_num//2, 1, 0) #voting
    
    def evaluate(self, X, y):
        pred = self.predict(X)
        return {"f1": f1_score(y, pred), "rc": recall_score(y, pred),
            "pc": precision_score(y, pred), "acc": accuracy_score(y, pred)}
    
    def feat_filter(self, X, y):
        self.feat_rf.fit(X, y)
        return np.argsort(self.feat_rf.feature_importances_)[::-1]
        

### Basic
from typing import List, Optional, NamedTuple
import numpy as np

### Pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR

### Gluonts
from gluonts.env import env
from gluonts.itertools import Cyclic, PseudoShuffled, Cached, maybe_len
from gluonts.torch.util import copy_parameters
from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.time_feature import TimeFeature
from gluonts.model.forecast_generator import SampleForecastGenerator
from gluonts.model.estimator import Estimator
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.transform import (
    Transformation, SelectFields, TransformedDataset,
    ExpectedNumInstanceSampler,
    ValidationSplitSampler,
    InstanceSplitter,
    TestSplitSampler,
    RenameFields,
)
### This repo
from .util import (
    fourier_time_features_from_frequency,
    lags_for_fourier_time_features_from_frequency,
    create_transformation
)

from .network import TrainingNetwork, PredictionNetwork
import logging
import inspect
def get_module_forward_input_names(module: nn.Module):
    params = inspect.signature(module.forward).parameters
    param_names = [k for k, v in params.items() if not str(v).startswith("*")]
    return param_names

from torch.utils.data import IterableDataset
class TransformedIterableDataset(IterableDataset):
    def __init__(
        self,
        dataset: Dataset,
        transform: Transformation,
        is_train: bool = True,
        shuffle_buffer_length: Optional[int] = None,
        cache_data: bool = False,
    ):
        super().__init__()
        self.shuffle_buffer_length = shuffle_buffer_length

        self.transformed_dataset = TransformedDataset(
            Cyclic(dataset) if not cache_data else Cached(Cyclic(dataset)),
            transform,
            is_train=is_train,
        )

    def __iter__(self):
        if self.shuffle_buffer_length is None:
            return iter(self.transformed_dataset)
        else:
            shuffled = PseudoShuffled(
                self.transformed_dataset,
                shuffle_buffer_length=self.shuffle_buffer_length,
            )
            return iter(shuffled)

class TrainOutput(NamedTuple):
    transformation: Transformation
    trained_net: nn.Module
    predictor: PyTorchPredictor

from tqdm.auto import tqdm
import logging
l1_loss = lambda A, B: abs(A - B).mean()
mse_loss = lambda A, B: ((A - B)**2).mean()
smooth_l1_loss = lambda A, B, beta=1.0: np.where(abs(A - B) < beta, 0.5*((A - B)**2)/beta, abs(A - B)-0.5*beta).mean()
        
class SibylPredictor(Estimator):
    def __init__(self, target_dim, freq, cell_input_size, window_size, prediction_length, batch_size=64,
        num_parallel_samples=100, device = torch.device("cpu"), loss_type="l2", pick_incomplete=False, 
        lags_seq: Optional[List[int]] = None, time_features: Optional[List[TimeFeature]] = None, **kwargs):
        
        super().__init__(lead_time=0)

        #self.dtype = np.float32
        self.device = device
        self.batch_size = batch_size
        self.loss_type = loss_type

        self.context_length = window_size
        self.prediction_length = prediction_length
        
        self.lags_seq = lags_seq if lags_seq is not None else lags_for_fourier_time_features_from_frequency(freq_str=freq)

        self.time_features = time_features if time_features is not None else fourier_time_features_from_frequency(freq)

        self.history_length = self.context_length + max(self.lags_seq)
        self.train_sampler = ExpectedNumInstanceSampler(num_instances=1.0,
            min_past=0 if pick_incomplete else self.history_length, min_future=prediction_length,)

        self.validation_sampler = ValidationSplitSampler(min_past=0 if pick_incomplete else self.history_length,
            min_future=prediction_length,)
        
        self.losses = []
        self.trained_net = TrainingNetwork(
            input_size=cell_input_size,
            target_dim=target_dim,
            history_length=self.history_length,
            context_length=window_size,
            prediction_length=prediction_length,
            lags_seq=self.lags_seq,
            loss_type=self.loss_type,
            **kwargs
        ).to(device)

        self.pred_net = PredictionNetwork(
            input_size=cell_input_size,
            target_dim=target_dim,
            history_length=self.history_length,
            context_length=window_size,
            prediction_length=prediction_length,
            lags_seq=self.lags_seq,
            num_parallel_samples=num_parallel_samples,
            loss_type=self.loss_type,
            **kwargs,
        ).to(device)

        self.transformation = create_transformation(self.time_features, self.prediction_length)
        
    def train(self, train_data, shuffle_buffer_length=None, cache_data=False, 
                epochs: int = 100, num_batches_per_epoch: int = 50, lr: float = 1e-3,
                weight_decay: float = 1e-6, max_lr: float = 1e-2, clip_gradient: Optional[float] = None, **kwargs) -> PyTorchPredictor:
        
        self.input_names = get_module_forward_input_names(self.trained_net)

        with env._let(max_idle_transforms=maybe_len(train_data) or 0):
            training_instance_splitter = self.create_instance_splitter("training")
        
        train_ds = TransformedIterableDataset(dataset=train_data,
            transform=self.transformation + training_instance_splitter + SelectFields(self.input_names),
            is_train=True, shuffle_buffer_length=shuffle_buffer_length, cache_data=cache_data)
        
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, prefetch_factor=2, pin_memory=True, worker_init_fn=self._worker_init_fn)
        
        optimizer = Adam(self.trained_net.parameters(), lr=lr, weight_decay=weight_decay)

        lr_scheduler = OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=num_batches_per_epoch, epochs=epochs)

        logging.info("Predictor Training...")
        for epoch_no in range(epochs):
            self.trained_net.train()
            cumm_epoch_loss = 0.0
            total = num_batches_per_epoch - 1

            with tqdm(train_dl, total=total) as it:
                for batch_no, data_entry in enumerate(it, start=1):
                    optimizer.zero_grad()

                    inputs = [v.to(self.device) for v in data_entry.values()]
                    output = self.trained_net(*inputs)
                    loss = output[0] if isinstance(output, (list, tuple)) else output

                    cumm_epoch_loss += loss.item()
                    avg_epoch_loss = cumm_epoch_loss / batch_no
                    it.set_postfix({"epoch": f"{epoch_no + 1}/{epochs}", "avg_loss": avg_epoch_loss,}, refresh=False)

                    loss.backward()
                    if clip_gradient is not None:
                        nn.utils.clip_grad_norm_(self.trained_net.parameters(), clip_gradient)

                    optimizer.step()
                    lr_scheduler.step()

                    if num_batches_per_epoch == batch_no: break
                it.close()
            if (epoch_no + 1) % 10 == 0:
                logging.info("Predictor training epoch {}/{} with cumm loss {:.6f}".format(epoch_no+1, epochs, cumm_epoch_loss))
            self.losses.append(cumm_epoch_loss)

        return self.create_predictor(self.trained_net)

    def create_predictor(self, train_net=None):    
        if train_net is not None:
            copy_parameters(self.trained_net, self.pred_net)

        my_predictor = PyTorchPredictor(
            input_transform=self.transformation + self.create_instance_splitter("test"),
            input_names=get_module_forward_input_names(self.pred_net),
            prediction_net=self.pred_net,
            batch_size=self.batch_size,
            forecast_generator=SampleForecastGenerator(),
            prediction_length=self.prediction_length,
            device=self.device,
        )

        train_out_put = TrainOutput(transformation=self.transformation, trained_net=self.trained_net,predictor=my_predictor)
        self.predictor = train_out_put.predictor
        return self.predictor


    def predict(self, test_data): #PytorchPredictor: minibatch inside
        return self.predictor.predict(test_data.iterable)
    
    def evaluation(self, test_data, test_gdth):
        logging.info("Predictor Evaluating...")
        pred_values = self.predict(test_data) #'as_json_dict', 'copy_aggregate', 'copy_dim', 'dim', 'freq', 'index', 'info', 'item_id', 'mean', 'mean_ts', 'median', 'num_samples', 'plot', 'prediction_length', 'quantile', 'quantile_ts', 'samples', 'start_date', 'to_quantile_forecast'
        test_size = len(test_gdth)
        err = 0.0
        err_fn = {"l1": l1_loss, "l2": mse_loss, "huber": smooth_l1_loss}[self.loss_type]
        
        pred_lst = []
        with tqdm(pred_values, total=test_size) as it:
            for i, pred in enumerate(it, start=0):
                err += err_fn(pred.mean, test_gdth[i]).mean() / self.batch_size
                pred_lst.append(pred)
            it.close()
        
        return {self.loss_type: err / test_size}, pred_lst
            
    @staticmethod
    def _worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)
    
    def create_instance_splitter(self, mode: str):
        assert mode in ["training", "validation", "test"]

        instance_sampler = {
        "training": self.train_sampler,
        "validation": self.validation_sampler,
        "test": TestSplitSampler(),
        }[mode]

        return InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=instance_sampler,
            past_length=self.history_length,
            future_length=self.prediction_length,
            time_series_fields=[
                FieldName.FEAT_TIME,
                FieldName.OBSERVED_VALUES,],
            ) + (RenameFields({
                    f"past_{FieldName.TARGET}": f"past_{FieldName.TARGET}_cdf",
                    f"future_{FieldName.TARGET}": f"future_{FieldName.TARGET}_cdf",})
            )



  
        
       




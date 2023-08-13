from gluonts.dataset.field_names import FieldName
from gluonts.transform import (
    Transformation,
    Chain,
    AsNumpyArray,
    ExpandDimArray,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    VstackFeatures,
    SetFieldIfNotPresent,
    TargetDimIndicator,
)

def create_transformation(time_feat, prediction_length) -> Transformation:
    return Chain([
            AsNumpyArray(field=FieldName.TARGET, expected_ndim=2,),
            ExpandDimArray(field=FieldName.TARGET, axis=None,),
            AddObservedValuesIndicator(target_field=FieldName.TARGET, output_field=FieldName.OBSERVED_VALUES,),
            AddTimeFeatures(
                    start_field=FieldName.START,
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_TIME,
                    time_features=time_feat,
                    pred_length=prediction_length,
                ),
            VstackFeatures(output_field=FieldName.FEAT_TIME, input_fields=[FieldName.FEAT_TIME],),
            SetFieldIfNotPresent(field=FieldName.FEAT_STATIC_CAT, value=[0]),
            TargetDimIndicator(field_name="target_dimension_indicator", target_field=FieldName.TARGET,),
            AsNumpyArray(field=FieldName.FEAT_STATIC_CAT, expected_ndim=1),
            ]
        )

import torch
def weighted_average(
        x, #x and weights are both tensors, sharing the same shape
        weights = None, # dim: the dim along which to average `x`
        dim=None):

    if weights is not None:
        weighted_tensor = torch.where(weights != 0, x * weights, torch.zeros_like(x))
        sum_weights = torch.clamp(
            weights.sum(dim=dim) if dim else weights.sum(), min=1.0
        )
        return (
            weighted_tensor.sum(dim=dim) if dim else weighted_tensor.sum()
        ) / sum_weights
    else:
        return x.mean(dim=dim)

from abc import ABC, abstractmethod
from typing import Tuple, List, Optional
from gluonts.core.component import validated
import torch.nn as nn

class Scaler(ABC, nn.Module):
    def __init__(self, keepdim: bool = False, time_first: bool = True):
        super().__init__()
        self.keepdim = keepdim
        self.time_first = time_first

    @abstractmethod
    def compute_scale(
        self, data: torch.Tensor, observed_indicator: torch.Tensor
    ) -> torch.Tensor:
        pass

    def forward(
        self, data: torch.Tensor, observed_indicator: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scale = self.compute_scale(data, observed_indicator)

        if self.time_first:
            dim = 1
        else:
            dim = 2
        if self.keepdim:
            scale = scale.unsqueeze(dim=dim)
            return data / scale, scale
        else:
            return data / scale.unsqueeze(dim=dim), scale


class MeanScaler(Scaler):
    @validated()
    def __init__(self, minimum_scale: float = 1e-10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("minimum_scale", torch.tensor(minimum_scale))

    def compute_scale(
        self, data: torch.Tensor, observed_indicator: torch.Tensor
    ) -> torch.Tensor:

        if self.time_first:
            dim = 1
        else:
            dim = 2

        # these will have shape (N, C)
        num_observed = observed_indicator.sum(dim=dim)
        sum_observed = (data.abs() * observed_indicator).sum(dim=dim)

        # first compute a global scale per-dimension
        total_observed = num_observed.sum(dim=0)
        denominator = torch.max(total_observed, torch.ones_like(total_observed))
        default_scale = sum_observed.sum(dim=0) / denominator

        # then compute a per-item, per-dimension scale
        denominator = torch.max(num_observed, torch.ones_like(num_observed))
        scale = sum_observed / denominator

        # use per-batch scale when no element is observed
        # or when the sequence contains only zeros
        scale = torch.where(
            sum_observed > torch.zeros_like(sum_observed),
            scale,
            default_scale * torch.ones_like(num_observed),
        )

        return torch.max(scale, self.minimum_scale).detach()


class NOPScaler(Scaler):
    @validated()
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_scale(
        self, data: torch.Tensor, observed_indicator: torch.Tensor
    ) -> torch.Tensor:
        if self.time_first:
            dim = 1
        else:
            dim = 2
        return torch.ones_like(data).mean(dim=dim)

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from gluonts.time_feature import TimeFeature, norm_freq_str

def lags_for_fourier_time_features_from_frequency(freq_str: str, num_lags: Optional[int] = None):
    offset = to_offset(freq_str)
    _, granularity = offset.n, offset.name

    if granularity == "M":
        lags = [[1, 12]]
    elif granularity == "D":
        lags = [[1, 7, 14]]
    elif granularity == "B":
        lags = [[1, 2]]
    elif granularity == "H":
        lags = [[1, 24, 168]]
    elif granularity in ("T", "min", "S", "sec"):
        lags = [[1, 4, 12, 24, 48]]
    else:
        lags = [[1]]

    # use less lags
    output_lags = list([int(lag) for sub_list in lags for lag in sub_list])
    output_lags = sorted(list(set(output_lags)))
    return output_lags[:num_lags]


class FourierDateFeatures(TimeFeature):
    @validated()
    def __init__(self, freq: str) -> None:
        super().__init__()
        # reocurring freq
        freqs = [
            "second",
            "month",
            "day",
            "hour",
            "minute",
            "weekofyear",
            "weekday",
            "dayofweek",
            "dayofyear",
            "daysinmonth",
        ]

        assert freq in freqs
        self.freq = freq

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        values = getattr(index, self.freq)
        num_values = max(values) + 1
        steps = [x * 2.0 * np.pi / num_values for x in values]
        return np.vstack([np.cos(steps), np.sin(steps)])


def fourier_time_features_from_frequency(freq_str: str) -> List[TimeFeature]:
    offset = to_offset(freq_str)
    granularity = norm_freq_str(offset.name)

    features = {
        "M": ["weekofyear"],
        "W": ["daysinmonth", "weekofyear"],
        "D": ["dayofweek"],
        "B": ["dayofweek", "dayofyear"],
        "H": ["hour", "dayofweek"],
        "min": ["minute", "hour", "dayofweek"],
        "T": ["minute", "hour", "dayofweek"],
        "S": ["second", "minute", "hour"]
    }

    assert granularity in features, f"freq {granularity} not supported"

    feature_classes: List[TimeFeature] = [
        FourierDateFeatures(freq=freq) for freq in features[granularity]
    ]
    return feature_classes
from typing import List, Union

import torch
import torch.nn as nn

#from gluonts.core.component import validated
from .util import (MeanScaler, NOPScaler, weighted_average)
from .diffusion import DiffusionOutput, GaussianDiffusion

from .PixelCNN import EpsilonTheta

class TrainingNetwork(nn.Module):
    #@validated()
    def __init__(self, target_dim, input_size, history_length, context_length, prediction_length, lags_seq: List[int],
                cell_num_layers=2, cell_hidden_size=40, cell_type="GRU", dropout_rate=0.1, conditioning_hidden=100,
                diff_steps=100, loss_type="l2", beta_end=0.1, beta_schedule="linear", resnet_block_groups=8, residual_channels=8,
                dim_mults_len=2, embedding_dimension=5, scaling=True, **kwargs):
        
        super().__init__()

        self.target_dim = target_dim
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.history_length = history_length
        self.scaling = scaling

        assert len(set(lags_seq)) == len(lags_seq), "no duplicated lags allowed!"
        lags_seq.sort()
        self.lags_seq = lags_seq

        self.cell_type = cell_type
        rnn_cls = {"LSTM": nn.LSTM, "GRU": nn.GRU}[cell_type]

        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=cell_hidden_size,
            num_layers=cell_num_layers,
            dropout=dropout_rate,
            batch_first=True,
        )

        et_target_dim = target_dim if target_dim > 1 else 2 #for UTS
        self.denoise_model = EpsilonTheta(
            target_dim=et_target_dim,
            cond_hidden=conditioning_hidden,
            resnet_block_groups=resnet_block_groups,
            residual_channels=residual_channels,
            dilation_cycle_length=dim_mults_len,
        )

        self.diffusion = GaussianDiffusion(
            self.denoise_model,
            input_size=target_dim,
            diff_steps=diff_steps,
            loss_type=loss_type,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
        )

        self.distr_output = DiffusionOutput(self.diffusion, input_size=target_dim, cond_size=conditioning_hidden)

        self.proj_dist_args = self.distr_output.get_args_proj(cell_hidden_size)

        self.embed_dim = embedding_dimension
        self.embed = nn.Embedding(num_embeddings=self.target_dim, embedding_dim=self.embed_dim)

        if self.scaling:
            self.scaler = MeanScaler(keepdim=True)
        else:
            self.scaler = NOPScaler(keepdim=True)

    @staticmethod
    def get_lagged_subsequences(sequence, sequence_length, indices: List[int], subsequences_length: int = 1) -> torch.Tensor: #(N, T, C) --> #(N, S, C, I)

        assert max(indices) + subsequences_length <= sequence_length, (
            f"lags cannot go further than history length, found lag "
            f"{max(indices)} while history length is only {sequence_length}"
        )
        assert all(lag_index >= 0 for lag_index in indices)

        lagged_values = []
        for lag_index in indices:
            begin_index = -lag_index - subsequences_length
            end_index = -lag_index if lag_index > 0 else None
            lagged_values.append(sequence[:, begin_index:end_index, ...].unsqueeze(1))
        return torch.cat(lagged_values, dim=1).permute(0, 2, 3, 1) #(N, S, C, I)

    def unroll(self, lags, scale, time_feat, target_dimension_indicator, unroll_length, begin_state= None):
        lags_scaled = lags / scale.unsqueeze(-1) #(B, S, C, I)
        input_lags = lags_scaled.reshape((-1, unroll_length, len(self.lags_seq) * self.target_dim))#(B, T, C*E)
        index_embeddings = self.embed(target_dimension_indicator) #(B, C, E)
        repeated_index_embeddings = (index_embeddings.unsqueeze(1).expand(-1, unroll_length, -1, -1).reshape((-1, unroll_length, self.target_dim * self.embed_dim))) #(B, T, C*E)

        inputs = torch.cat((input_lags, repeated_index_embeddings, time_feat), dim=-1) #(B, S, input_dim)

        outputs, state = self.rnn(inputs, begin_state)

        return outputs, state, lags_scaled, inputs #State: Union[List[torch.Tensor], torch.Tensor],

    def unroll_encoder(
        self, past_time_feat, past_target_cdf, past_observed_values, past_is_pad, target_dimension_indicator, future_time_feat = None, future_target_cdf = None):

        past_observed_values = torch.min(past_observed_values, 1 - past_is_pad.unsqueeze(-1))

        if future_time_feat is None or future_target_cdf is None:
            time_feat = past_time_feat[:, -self.context_length :, ...]
            sequence = past_target_cdf
            sequence_length = self.history_length
            subsequences_length = self.context_length
        else:
            time_feat = torch.cat((past_time_feat[:, -self.context_length :, ...], future_time_feat), dim=1,)
            sequence = torch.cat((past_target_cdf, future_target_cdf), dim=1)
            sequence_length = self.history_length + self.prediction_length
            subsequences_length = self.context_length + self.prediction_length

        lags = self.get_lagged_subsequences(
            sequence=sequence,
            sequence_length=sequence_length,
            indices=self.lags_seq,
            subsequences_length=subsequences_length,
        ) #(B, S, C, I)

        _, scale = self.scaler(
            past_target_cdf[:, -self.context_length :, ...],
            past_observed_values[:, -self.context_length :, ...],
        ) #(B, 1, C)

        #(B, T, H), Nested list with (B, H), (B, S, C, I), inputs to rnn
        outputs, states, lags_scaled, inputs = self.unroll(
            lags=lags,
            scale=scale,
            time_feat=time_feat,
            target_dimension_indicator=target_dimension_indicator,
            unroll_length=subsequences_length,
            begin_state=None,
        )

        return outputs, states, scale, lags_scaled, inputs

    def proj_rnn_outputs(self, rnn_outputs: torch.Tensor):
        
        (rnn_output_projections,) = self.proj_dist_args(rnn_outputs)
        return rnn_output_projections

    def forward(self, target_dimension_indicator, past_time_feat, past_target_cdf, past_observed_values, past_is_pad, 
                        future_time_feat, future_target_cdf, future_observed_values):

        seq_len = self.context_length + self.prediction_length

        # unroll the decoder in "training mode", i.e. by providing future data as well
        rnn_outputs, _, scale, _, _ = self.unroll_encoder(
            past_time_feat=past_time_feat,
            past_target_cdf=past_target_cdf,
            past_observed_values=past_observed_values,
            past_is_pad=past_is_pad,
            future_time_feat=future_time_feat,
            future_target_cdf=future_target_cdf,
            target_dimension_indicator=target_dimension_indicator,
        )

        # put together target sequence
        target = torch.cat((past_target_cdf[:, -self.context_length :, ...], future_target_cdf), dim=1,) #(B, T, C)

        rnn_output_projections = self.proj_rnn_outputs(rnn_outputs=rnn_outputs)
        if self.scaling: self.diffusion.scale = scale

        likelihoods = self.diffusion.log_prob(target, rnn_output_projections).unsqueeze(-1)

        past_observed_values = torch.min(past_observed_values, 1 - past_is_pad.unsqueeze(-1))

        observed_values = torch.cat((past_observed_values[:, -self.context_length :, ...],future_observed_values,), dim=1,) #(B, S, C)

        # mask the loss at one time step if one or more observations is missing in the target dimensions 
        loss_weights, _ = observed_values.min(dim=-1, keepdim=True) #(B, S, 1)
        loss = weighted_average(likelihoods, weights=loss_weights, dim=1)

        return (loss.mean(), likelihoods, rnn_output_projections)


class PredictionNetwork(TrainingNetwork):
    def __init__(self, num_parallel_samples, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_parallel_samples = num_parallel_samples
        self.shifted_lags = [l - 1 for l in self.lags_seq]

    def sampling_decoder(
        self,
        past_target_cdf: torch.Tensor,
        target_dimension_indicator: torch.Tensor,
        time_feat: torch.Tensor,
        scale: torch.Tensor,
        begin_states: Union[List[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:

        def repeat(tensor, dim=0):
            return tensor.repeat_interleave(repeats=self.num_parallel_samples, dim=dim)

        repeated_past_target_cdf = repeat(past_target_cdf)
        repeated_time_feat = repeat(time_feat)
        repeated_scale = repeat(scale)
        if self.scaling:
            self.diffusion.scale = repeated_scale
        repeated_target_dimension_indicator = repeat(target_dimension_indicator)

        if self.cell_type == "LSTM":
            repeated_states = [repeat(s, dim=1) for s in begin_states]
        else:
            repeated_states = repeat(begin_states, dim=1)

        future_samples = []

        for k in range(self.prediction_length):
            lags = self.get_lagged_subsequences(
                sequence=repeated_past_target_cdf,
                sequence_length=self.history_length + k,
                indices=self.shifted_lags,
                subsequences_length=1,
            )

            rnn_outputs, repeated_states, _, _ = self.unroll(
                begin_state=repeated_states,
                lags=lags,
                scale=repeated_scale,
                time_feat=repeated_time_feat[:, k : k + 1, ...],
                target_dimension_indicator=repeated_target_dimension_indicator,
                unroll_length=1,
            )
            rnn_output_projections = self.proj_rnn_outputs(rnn_outputs=rnn_outputs)
            new_samples = self.diffusion.sample(cond=rnn_output_projections) #(B * N, 1, C)
            future_samples.append(new_samples) #(B, T, C)
            repeated_past_target_cdf = torch.cat((repeated_past_target_cdf, new_samples), dim=1)

        samples = torch.cat(future_samples, dim=1) #(B * N, prediction_length, C), N = num_parallel_samples

        return samples.reshape((-1, self.num_parallel_samples, self.prediction_length, self.target_dim,)) #(B, N, prediction_length, C)

    def forward(self,
        target_dimension_indicator: torch.Tensor, #(B, C)
        past_time_feat: torch.Tensor, #(B, H, num_feat)
        past_target_cdf: torch.Tensor, #(B, H, C)
        past_observed_values: torch.Tensor, #(B, H, C)
        past_is_pad: torch.Tensor, #(B, H)
        future_time_feat: torch.Tensor, #(B, prediction_length, num_feat)
    ) -> torch.Tensor:

        # mark padded data as unobserved
        past_observed_values = torch.min(past_observed_values, 1 - past_is_pad.unsqueeze(-1)) #(B, C, T)

        _, begin_states, scale, _, _ = self.unroll_encoder(
            past_time_feat=past_time_feat,
            past_target_cdf=past_target_cdf,
            past_observed_values=past_observed_values,
            past_is_pad=past_is_pad,
            future_time_feat=None,
            future_target_cdf=None,
            target_dimension_indicator=target_dimension_indicator,
        )

        return self.sampling_decoder(
            past_target_cdf=past_target_cdf,
            target_dimension_indicator=target_dimension_indicator,
            time_feat=future_time_feat,
            scale=scale,
            begin_states=begin_states,
        )

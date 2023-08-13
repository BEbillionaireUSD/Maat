import math
import torch
from torch import nn

class DiffusionEmbedding(nn.Module):
    def __init__(self, dim, out_dim, max_steps=500):
        super().__init__()
        self.register_buffer("embedding", self._embedder(dim, max_steps), persistent=False)
        self.projection = nn.Sequential(
            nn.Linear(dim * 2, out_dim), 
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
            nn.SiLU()
        )

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        return self.projection(x)

    def _embedder(self, dim, max_steps):
        steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
        dims = torch.arange(dim).unsqueeze(0)  # [1,dim]
        table = steps * 10.0 ** (dims * 4.0 / dim)  # [T,dim]
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table

class ResidualBlock(nn.Module):
    def __init__(self, hidden_size, residual_channels, dilation, leaky_ratio = 0.4):
        super().__init__()
        self.dilated_conv = nn.Conv1d(
            residual_channels,
            2 * residual_channels,
            kernel_size = 3,
            padding=dilation,
            dilation=dilation,
            padding_mode="circular",
        )
        self.diffusion_projection = nn.Linear(hidden_size, residual_channels)
        self.conditioner_projection = nn.Conv1d(
            1, 2 * residual_channels, 1, padding=2, padding_mode="circular"
        )
        self.output_projection = nn.Sequential(
            nn.Conv1d(residual_channels, 2 * residual_channels, 1),
            nn.LeakyReLU(leaky_ratio)
        )

        nn.init.kaiming_normal_(self.conditioner_projection.weight)
        nn.init.kaiming_normal_(self.output_projection[0].weight)

    def forward(self, x, condition_emb, diffusion_emb):
        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)
        condition_emb = self.conditioner_projection(condition_emb)

        y = x + diffusion_emb
        y = self.dilated_conv(y) 
        y = y + condition_emb

        left, right = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(left) * torch.tanh(right)

        y = self.output_projection(y)
        left, right = torch.chunk(y, 2, dim=1)
        return (x + left) / math.sqrt(2.0), right # to the next layer; for skip-connection outputs


class EpsilonTheta(nn.Module):
    def __init__(
        self,
        target_dim,
        cond_hidden,
        time_emb_dim=16,
        resnet_block_groups=8,
        residual_channels=8,
        dilation_cycle_length=2,
        residual_hidden=64,
        kernel_size = 3,
        leaky_ratio = 0.4,
    ):
        super().__init__()
        ### Process the three inputs
        self.input_projection = nn.Sequential(
            nn.Conv1d(1, residual_channels, 1, padding=2, padding_mode="circular"),
            nn.LeakyReLU(leaky_ratio)
        )
        self.diff_embedder = DiffusionEmbedding(dim=time_emb_dim, out_dim=residual_hidden)
        
        self.condition_projection = nn.Sequential(
            nn.Linear(cond_hidden, target_dim // 2),
            nn.LeakyReLU(leaky_ratio),
            nn.Linear(target_dim // 2, target_dim),
            nn.LeakyReLU(leaky_ratio),
        )
        
        self.resnet_block_groups = nn.ModuleList([
                ResidualBlock(
                    residual_channels=residual_channels,
                    dilation=2 ** (i % dilation_cycle_length),
                    hidden_size=residual_hidden,
                    leaky_ratio = leaky_ratio
                )
                for i in range(resnet_block_groups)
            ])
        
        #### Process the outputs
        self.skip_projection = nn.Sequential(
            nn.Conv1d(residual_channels, residual_channels, kernel_size=kernel_size),
            nn.LeakyReLU(leaky_ratio)
        )
        self.output_projection = nn.Conv1d(residual_channels, 1, kernel_size=kernel_size)

        nn.init.kaiming_normal_(self.input_projection[0].weight)
        nn.init.kaiming_normal_(self.skip_projection[0].weight)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, inputs, time, cond):
        target_dim = inputs.shape[-1]
        if target_dim == 1:
            inputs = inputs.repeat(1, 1, 2)

        x = self.input_projection(inputs)
        diffusion_emb = self.diff_embedder(time)
        condition_emb = self.condition_projection(cond)
        
        skip = []
        for layer in self.resnet_block_groups:
            x, skip_connection = layer(x, condition_emb, diffusion_emb)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.resnet_block_groups))
        x = self.skip_projection(x)
        x = self.output_projection(x)
        
        if target_dim == 1:
            return x.mean(dim=-1).unsqueeze(dim=-1)
        
        return x

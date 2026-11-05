"""Networks mapping an agent's local transition ``(o_t, a_t[, o_{t+1}])`` to an output.

The same network is used as a group classifier (``out_dim = num_groups``) and as a
reward regressor (``out_dim = 1``). With the default :class:`TaskSpec` presets the
classifier reproduces the per-task architectures used in the paper.
"""

import math
from typing import Optional, Sequence

import torch
from torch import nn


class TransitionNet(nn.Module):
    """``f(o, a[, o']) -> R^out_dim``.

    Args:
        obs_shape: per-agent observation shape, ``(H, W, C)`` for ``encoder="cnn"`` or
            any shape for ``encoder="mlp"`` (it is flattened).
        act_dim: size of the encoded action (one-hot size or continuous dim).
        out_dim: output size.
        use_next_obs: also feed ``o_{t+1}`` (stacked on channels for the CNN,
            concatenated for the MLP). Rewards are ``R(s, a, s')`` so the next
            observation is often very informative (e.g. a captured evader vanishes).
        encoder: ``"cnn"`` or ``"mlp"``.
        cnn_channels / cnn_paddings: conv stack (3x3 kernels, ReLU).
        hidden: hidden width of the head.
    """

    def __init__(
        self,
        obs_shape: Sequence[int],
        act_dim: int,
        out_dim: int,
        use_next_obs: bool = False,
        encoder: str = "mlp",
        cnn_channels: Sequence[int] = (16, 32),
        cnn_paddings: Sequence[int] = (1, 1),
        hidden: int = 128,
    ):
        super().__init__()
        self.obs_shape = tuple(obs_shape)
        self.use_next_obs = use_next_obs
        self.encoder_type = encoder
        n_obs = 2 if use_next_obs else 1

        if encoder == "cnn":
            if len(self.obs_shape) != 3:
                raise ValueError(f"CNN encoder expects (H, W, C) obs, got {obs_shape}")
            h, w, c = self.obs_shape
            layers, in_ch = [], c * n_obs
            for out_ch, pad in zip(cnn_channels, cnn_paddings):
                layers += [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=pad), nn.ReLU()]
                in_ch = out_ch
            layers.append(nn.Flatten())
            self.encoder = nn.Sequential(*layers)
            with torch.no_grad():
                feat_dim = self.encoder(torch.zeros(1, c * n_obs, h, w)).shape[-1]
            self.head = nn.Sequential(
                nn.Linear(feat_dim + act_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim)
            )
        elif encoder == "mlp":
            in_dim = n_obs * math.prod(self.obs_shape) + act_dim
            self.encoder = nn.Identity()
            self.head = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, out_dim),
            )
        else:
            raise ValueError(f"Unknown encoder '{encoder}'")

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor, next_obs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """``obs``/``next_obs``: ``[N, *obs_shape]``, ``action``: ``[N, act_dim]``."""
        if self.use_next_obs:
            if next_obs is None:
                raise ValueError("use_next_obs=True but next_obs was not provided")
            obs = torch.cat([obs, next_obs], dim=-1)
        if self.encoder_type == "cnn":
            feats = self.encoder(obs.permute(0, 3, 1, 2))  # [N, H, W, C] -> [N, C, H, W]
        else:
            feats = obs.flatten(1)
        return self.head(torch.cat([feats, action], dim=-1))

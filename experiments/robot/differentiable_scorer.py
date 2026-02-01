from typing import Optional

import torch
from torch import Tensor, nn


class ActionOnlyCritic(nn.Module):
    def __init__(self, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: Optional[object], action: Tensor) -> Tensor:
        return self.net(action).squeeze(-1)


def check_differentiable_scorer(
    scorer: nn.Module, state: Optional[object], action: Tensor
) -> Tensor:
    action = action.detach().clone().requires_grad_(True)
    score = scorer(state, action)
    if not isinstance(score, torch.Tensor) or not score.requires_grad:
        raise ValueError("scorer 必須回傳可微分的 torch.Tensor。")
    return score


from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from experiments.robot.token_action_converter import TokenActionConverter

Tensor = torch.Tensor


@dataclass
class StGsConfig:
    inner_steps: int = 5
    tau: float = 1.0
    step_size: float = 1e-1
    prior_weight: float = 0.0
    anchor_weight: float = 0.0
    init_logit_scale: float = 10.0
    detach_each_step: bool = True


def st_gumbel_softmax(logits: Tensor, tau: float, dim: int = -1) -> Tuple[Tensor, Tensor, Tensor]:
    gumbel = -torch.empty_like(logits).exponential_().log()
    y_soft = F.softmax((logits + gumbel) / tau, dim=dim)
    y_hard = F.one_hot(y_soft.argmax(dim=dim), num_classes=logits.size(dim)).type_as(y_soft)
    y_st = y_hard - y_soft.detach() + y_soft
    return y_soft, y_hard, y_st


def init_logits_from_token_ids(
    token_ids: np.ndarray,
    converter: TokenActionConverter,
    init_logit_scale: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    bin_indices = converter.token_ids_to_bin_indices(token_ids)
    if not torch.is_tensor(bin_indices):
        bin_indices = torch.as_tensor(bin_indices, device=device, dtype=torch.long)
    action_dim = bin_indices.numel()
    z0 = torch.full(
        (action_dim, converter.num_bins),
        fill_value=-init_logit_scale,
        device=device,
        dtype=dtype,
    )
    z0[torch.arange(action_dim, device=device), bin_indices] = init_logit_scale
    return z0


def _as_scalar(x: Tensor) -> Tensor:
    return x if x.ndim == 0 else x.mean()


def _ensure_differentiable_reward(reward: Tensor) -> None:
    if not isinstance(reward, torch.Tensor):
        raise TypeError("reward_fn must return a torch.Tensor.")
    if not reward.requires_grad:
        raise ValueError("reward_fn must be differentiable w.r.t. a_soft.")


def _log_prob_under_base(y_st: Tensor, base_logits: Tensor) -> Tensor:
    log_probs = F.log_softmax(base_logits, dim=-1)
    token_logp = (y_st * log_probs).sum(dim=-1)
    return token_logp.sum()


def refine_logits_with_stgs(
    z0: Tensor,
    converter: TokenActionConverter,
    reward_fn: Callable[[Tensor], Tensor],
    cfg: StGsConfig,
) -> Tensor:
    z = z0.detach().clone().requires_grad_(True)
    for _ in range(cfg.inner_steps):
        y_soft, _, y_st = st_gumbel_softmax(z, cfg.tau)
        a_soft = converter.soft_token_probs_to_action(y_soft)

        reward = _as_scalar(reward_fn(a_soft))
        _ensure_differentiable_reward(reward)

        prior = (
            _as_scalar(_log_prob_under_base(y_st, z0))
            if cfg.prior_weight != 0.0
            else torch.zeros_like(reward)
        )
        anchor = (z - z0).pow(2).mean() if cfg.anchor_weight != 0.0 else torch.zeros_like(reward)

        objective = reward + cfg.prior_weight * prior - cfg.anchor_weight * anchor
        grad = torch.autograd.grad(objective, z, retain_graph=False, create_graph=False)[0]
        z = z + cfg.step_size * grad
        if cfg.detach_each_step:
            z = z.detach().requires_grad_(True)
    return z


def refine_tokens_with_stgs(
    token_ids: np.ndarray,
    converter: TokenActionConverter,
    reward_fn: Callable[[Tensor], Tensor],
    cfg: StGsConfig,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tuple[np.ndarray, np.ndarray]:
    refined_token_ids = []
    refined_actions = []

    for single_tokens in token_ids:
        z0 = init_logits_from_token_ids(
            single_tokens, converter, cfg.init_logit_scale, device=device, dtype=dtype
        )
        z = refine_logits_with_stgs(z0, converter, reward_fn, cfg)
        bin_indices = z.argmax(dim=-1)
        final_token_ids = converter.bin_indices_to_token_ids(bin_indices).detach().cpu().numpy()
        refined_token_ids.append(final_token_ids)
        refined_actions.append(converter.token_to_action(final_token_ids))

    return np.array(refined_token_ids), np.array(refined_actions)


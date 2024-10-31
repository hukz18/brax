# Copyright 2024 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Proximal policy optimization training.

See: https://arxiv.org/pdf/1707.06347.pdf
"""

from typing import Any, Tuple

from brax.training import types
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.types import Params
import flax
import jax
import jax.numpy as jnp


@flax.struct.dataclass
class PPONetworkParams:
  """Contains training state for the learner."""
  policy: Params
  value: Params


def compute_gae(truncation: jnp.ndarray,
                termination: jnp.ndarray,
                rewards: jnp.ndarray,
                values: jnp.ndarray,
                bootstrap_value: jnp.ndarray,
                lambda_: float = 1.0,
                discount: float = 0.99):
  """Calculates the Generalized Advantage Estimation (GAE).

  Args:
    truncation: A float32 tensor of shape [T, B] with truncation signal.
    termination: A float32 tensor of shape [T, B] with termination signal.
    rewards: A float32 tensor of shape [T, B] containing rewards generated by
      following the behaviour policy.
    values: A float32 tensor of shape [T, B] with the value function estimates
      wrt. the target policy.
    bootstrap_value: A float32 of shape [B] with the value function estimate at
      time T.
    lambda_: Mix between 1-step (lambda_=0) and n-step (lambda_=1). Defaults to
      lambda_=1.
    discount: TD discount.

  Returns:
    A float32 tensor of shape [T, B]. Can be used as target to
      train a baseline (V(x_t) - vs_t)^2.
    A float32 tensor of shape [T, B] of advantages.
  """

  truncation_mask = 1 - truncation
  # Append bootstrapped value to get [v1, ..., v_t+1]
  values_t_plus_1 = jnp.concatenate(
      [values[1:], jnp.expand_dims(bootstrap_value, 0)], axis=0)
  deltas = rewards + discount * (1 - termination) * values_t_plus_1 - values
  deltas *= truncation_mask

  acc = jnp.zeros_like(bootstrap_value)
  vs_minus_v_xs = []

  def compute_vs_minus_v_xs(carry, target_t):
    lambda_, acc = carry
    truncation_mask, delta, termination = target_t
    acc = delta + discount * (1 - termination) * truncation_mask * lambda_ * acc
    return (lambda_, acc), (acc)

  (_, _), (vs_minus_v_xs) = jax.lax.scan(
      compute_vs_minus_v_xs, (lambda_, acc),
      (truncation_mask, deltas, termination),
      length=int(truncation_mask.shape[0]),
      reverse=True)
  # Add V(x_s) to get v_s.
  vs = jnp.add(vs_minus_v_xs, values)

  vs_t_plus_1 = jnp.concatenate(
      [vs[1:], jnp.expand_dims(bootstrap_value, 0)], axis=0)
  advantages = (rewards + discount *
                (1 - termination) * vs_t_plus_1 - values) * truncation_mask
  return jax.lax.stop_gradient(vs), jax.lax.stop_gradient(advantages)


def compute_ppo_loss(
    params: PPONetworkParams,
    normalizer_params: Any,
    data: types.Transition,
    rng: jnp.ndarray,
    ppo_network: ppo_networks.PPONetworks,
    entropy_cost: float = 1e-4,
    discounting: float = 0.9,
    reward_scaling: float = 1.0,
    gae_lambda: float = 0.95,
    clipping_epsilon: float = 0.3,
    normalize_advantage: bool = True) -> Tuple[jnp.ndarray, types.Metrics]:
  """Computes PPO loss.

  Args:
    params: Network parameters,
    normalizer_params: Parameters of the normalizer.
    data: Transition that with leading dimension [B, T]. extra fields required
      are ['state_extras']['truncation'] ['policy_extras']['raw_action']
        ['policy_extras']['log_prob']
    rng: Random key
    ppo_network: PPO networks.
    entropy_cost: entropy cost.
    discounting: discounting,
    reward_scaling: reward multiplier.
    gae_lambda: General advantage estimation lambda.
    clipping_epsilon: Policy loss clipping epsilon
    normalize_advantage: whether to normalize advantage estimate

  Returns:
    A tuple (loss, metrics)
  """
  parametric_action_distribution = ppo_network.parametric_action_distribution
  policy_apply = ppo_network.policy_network.apply
  value_apply = ppo_network.value_network.apply

  # Put the time dimension first.
  data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)
  policy_logits = policy_apply(normalizer_params, params.policy,
                               data.observation)

  baseline = value_apply(normalizer_params, params.value, data.privileged_observation)

  bootstrap_value = value_apply(normalizer_params, params.value,
                                data.next_privileged_observation[-1])

  rewards = data.reward * reward_scaling
  truncation = data.extras['state_extras']['truncation']
  termination = (1 - data.discount) * (1 - truncation)

  target_action_log_probs = parametric_action_distribution.log_prob(
      policy_logits, data.extras['policy_extras']['raw_action'])
  behaviour_action_log_probs = data.extras['policy_extras']['log_prob']

  vs, advantages = compute_gae(
      truncation=truncation,
      termination=termination,
      rewards=rewards,
      values=baseline,
      bootstrap_value=bootstrap_value,
      lambda_=gae_lambda,
      discount=discounting)
  if normalize_advantage:
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
  rho_s = jnp.exp(target_action_log_probs - behaviour_action_log_probs)

  surrogate_loss1 = rho_s * advantages
  surrogate_loss2 = jnp.clip(rho_s, 1 - clipping_epsilon,
                             1 + clipping_epsilon) * advantages

  policy_loss = -jnp.mean(jnp.minimum(surrogate_loss1, surrogate_loss2))

  # Value function loss
  v_error = vs - baseline
  v_loss = jnp.mean(v_error * v_error) * 0.5 * 0.5

  # Entropy reward
  entropy = jnp.mean(parametric_action_distribution.entropy(policy_logits, rng))
  entropy_loss = entropy_cost * -entropy

  total_loss = policy_loss + v_loss + entropy_loss
  return total_loss, {
      'total_loss': total_loss,
      'policy_loss': policy_loss,
      'v_loss': v_loss,
      'entropy_loss': entropy_loss
  }

def compute_ppo_value_loss(
    params: PPONetworkParams,
    normalizer_params: Any,
    data: types.Transition,
    rng: jnp.ndarray,
    ppo_network: ppo_networks.PPONetworks,
    discounting: float = 0.9,
    reward_scaling: float = 1.0,
    gae_lambda: float = 0.95,
    normalize_advantage: bool = True) -> Tuple[jnp.ndarray, types.Metrics]:
  """Computes PPO value loss.

  Args:
    params: Network parameters,
    normalizer_params: Parameters of the normalizer.
    data: Transition that with leading dimension [B, T]. extra fields required
      are ['state_extras']['truncation'] ['policy_extras']['raw_action']
        ['policy_extras']['log_prob']
    rng: Random key
    ppo_network: PPO networks.
    discounting: discounting,
    reward_scaling: reward multiplier.
    gae_lambda: General advantage estimation lambda.
    normalize_advantage: whether to normalize advantage estimate

  Returns:
    A tuple (loss, metrics)
  """

  value_apply = ppo_network.value_network.apply

  # Put the time dimension first.
  data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)

  baseline = value_apply(normalizer_params, params.value, data.privileged_observation)

  bootstrap_value = value_apply(normalizer_params, params.value,
                                data.next_privileged_observation[-1])

  rewards = data.reward * reward_scaling
  truncation = data.extras['state_extras']['truncation']
  termination = (1 - data.discount) * (1 - truncation)

  vs, advantages = compute_gae(
      truncation=truncation,
      termination=termination,
      rewards=rewards,
      values=baseline,
      bootstrap_value=bootstrap_value,
      lambda_=gae_lambda,
      discount=discounting)
  if normalize_advantage:
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

  # Value function loss
  v_error = vs - baseline
  v_loss = jnp.mean(v_error * v_error) * 0.5 * 0.5

  return v_loss, {
      'v_loss': v_loss,
  }
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

"""Brax training acting functions."""

import os
import time
from typing import Any, Callable, List, Sequence, Tuple, Union

import mediapy as media
from moviepy.editor import VideoFileClip, clips_array

from flax import struct
from brax import envs
from brax.training.types import Metrics
from brax.training.types import Policy
from brax.training.types import PolicyParams
from brax.training.types import PRNGKey
from brax.training.types import Transition
from brax.v1 import envs as envs_v1
import jax
import numpy as np

State = Union[envs.State, envs_v1.State]
Env = Union[envs.Env, envs_v1.Env, envs_v1.Wrapper]

@struct.dataclass
class BraxState:
  """Dynamic state that changes after every pipeline step.

  Attributes:
    q: (q_size,) joint position vector
    qd: (qd_size,) joint velocity vector
  """

  q: np.ndarray
  qd: np.ndarray

# Function to convert dict-of-list to list-of-dict
def dict_of_list_to_list_of_dict(data):
    seq_length = next(iter(data.values())).shape[0]  # Get seq_length from any value

    # For each index in the sequence, create a dictionary of that slice
    return [BraxState(**jax.tree_map(lambda arr: np.array(arr[i]), data)) for i in range(seq_length)]


def render_video(
    env: Env,
    rollout: List[Any],
    run_name: str,
    current_step: int,
    render_every: int = 4,
    height: int = 480,
    width: int = 640,
    render_eval: bool = True,
):
    # Define paths for each camera's video
    video_paths: List[str] = []
    rollout = dict_of_list_to_list_of_dict(rollout)[:500]

    # Render and save videos for each camera
    # for camera in ["perspective", "side", "top", "front"]:
    video_path = os.path.join("results", run_name, "videos", f"train_{current_step}.mp4")
    if render_eval:
        video_path = video_path.replace("train", "eval")
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    media.write_video(
        video_path,
        env.render(
            rollout[::render_every], height=height, width=width
        ),
        fps=1.0 / env.dt / render_every,
    )
    video_paths.append(video_path)
    print(f"Saved video to {video_path}")

    # Load the video clips using moviepy
    # clips = [VideoFileClip(path) for path in video_paths]
    # Arrange the clips in a 2x2 grid
    # final_video = clips_array([[clips[0], clips[1]], [clips[2], clips[3]]])
    # Save the final concatenated video
    # final_video.write_videofile(os.path.join("results", run_name, "eval.mp4"))


def actor_step(
    env: Env,
    env_state: State,
    policy: Policy,
    key: PRNGKey,
    extra_fields: Sequence[str] = ()
) -> Tuple[State, Transition]:
  """Collect data."""
  actions, policy_extras = policy(env_state.obs, key)
  nstate = env.step(env_state, actions)
  assert isinstance(nstate, envs.State)
  state_extras = {x: nstate.info[x] for x in extra_fields}
  pipeline_state_keys = ['q', 'qd']
  pipeline_state = {x: getattr(nstate.pipeline_state, x) for x in pipeline_state_keys}
  # jax.debug.print('pipeline state shape {pipeline_state}', pipeline_state_shape=nstate.info.keys())
  return nstate, Transition(  # pytype: disable=wrong-arg-types  # jax-ndarray
      observation=env_state.obs,
      privileged_observation=env_state.privileged_obs,
      action=actions,
      reward=nstate.reward,
      discount=1 - nstate.done,
      next_observation=nstate.obs,
      next_privileged_observation=nstate.privileged_obs,
      extras={
          'policy_extras': policy_extras,
          'state_extras': state_extras,
          'pipeline_states': pipeline_state,
      })


def generate_unroll(
    env: Env,
    env_state: State,
    policy: Policy,
    key: PRNGKey,
    unroll_length: int,
    extra_fields: Sequence[str] = ()
) -> Tuple[State, Transition]:
  """Collect trajectories of given unroll_length."""

  @jax.jit
  def f(carry, unused_t):
    state, current_key = carry
    current_key, next_key = jax.random.split(current_key)
    nstate, transition = actor_step(
        env, state, policy, current_key, extra_fields=extra_fields)
    return (nstate, next_key), transition

  (final_state, _), data = jax.lax.scan(
      f, (env_state, key), (), length=unroll_length)
  return final_state, data


# TODO: Consider moving this to its own file.
class Evaluator:
  """Class to run evaluations."""

  def __init__(self, eval_env: envs.Env,
               eval_policy_fn: Callable[[PolicyParams], Policy], 
               num_eval_envs: int,
               episode_length: int, action_repeat: int, key: PRNGKey,
               render_interval: int):
    """Init.

    Args:
      eval_env: Batched environment to run evals on.
      eval_policy_fn: Function returning the policy from the policy parameters.
      num_eval_envs: Each env will run 1 episode in parallel for each eval.
      episode_length: Maximum length of an episode.
      action_repeat: Number of physics steps per env step.
      key: RNG key.
    """
    self._key = key
    self._eval_walltime = 0.

    eval_env = envs.training.EvalWrapper(eval_env)
    self.eval_env = eval_env
    self.render_interval = render_interval
    self.render_counter = 0

    def generate_eval_unroll(policy_params: PolicyParams,
                             key: PRNGKey) -> State:
      reset_keys = jax.random.split(key, num_eval_envs)
      eval_first_state = eval_env.reset(reset_keys)
      return generate_unroll(
          eval_env,
          eval_first_state,
          eval_policy_fn(policy_params),
          key,
          unroll_length=episode_length // action_repeat)

    self._generate_eval_unroll = jax.jit(generate_eval_unroll)
    self._steps_per_unroll = episode_length * num_eval_envs

  def run_evaluation(self,
                     policy_params: PolicyParams,
                     training_metrics: Metrics,
                     aggregate_episodes: bool = True,
                     run_name: str = "",
                     current_step: int = 0) -> Metrics:
    """Run one epoch of evaluation."""
    self._key, unroll_key = jax.random.split(self._key)

    t = time.time()
    eval_state, eval_data = self._generate_eval_unroll(policy_params, unroll_key)
    if self.render_interval and self.render_counter % self.render_interval == 0:
        render_video(self.eval_env, eval_data.extras['pipeline_states'], run_name, current_step)
    self.render_counter += 1

    eval_metrics = eval_state.info['eval_metrics']
    eval_metrics.active_episodes.block_until_ready()
    epoch_eval_time = time.time() - t
    metrics = {}
    for fn in [np.mean, np.std]:
      suffix = '_std' if fn == np.std else ''
      metrics.update(
          {
              f'eval/episode_{name}{suffix}': (
                  fn(value) if aggregate_episodes else value
              )
              for name, value in eval_metrics.episode_metrics.items()
          }
      )
    metrics['eval/avg_episode_length'] = np.mean(eval_metrics.episode_steps)
    metrics['eval/epoch_eval_time'] = epoch_eval_time
    metrics['eval/sps'] = self._steps_per_unroll / epoch_eval_time
    self._eval_walltime = self._eval_walltime + epoch_eval_time
    metrics = {
        'eval/walltime': self._eval_walltime,
        **training_metrics,
        **metrics
    }

    return metrics  # pytype: disable=bad-return-type  # jax-ndarray

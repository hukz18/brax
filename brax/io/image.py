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

"""Exports a system config and state as an image."""

import io
from typing import List, Optional, Sequence, Union
from tqdm import tqdm
import math
import brax
from brax import base
import mujoco
import numpy as np
from PIL import Image
import multiprocessing as mp
from joblib import Parallel, delayed


def get_image(model: mujoco.MjModel, state: base.State, height: int, width: int, camera: Optional[str]=None, spacing: Optional[float]=1.0, num_vis: Optional[int]=25) -> np.ndarray:
  # model = mujoco.MjModel.from_xml_path(model_xml)
  renderer = mujoco.Renderer(model, height=height, width=width)
  data = mujoco.MjData(model)
  if len(state.q.shape) == 1:

    data.qpos, data.qvel = state.q, state.qd
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=camera)
  elif len(state.q.shape) == 2:
    assert state.q.shape[0] == state.qd.shape[0]
    batch_size = state.q.shape[0]
    num_vis = min(num_vis, batch_size)

    # Calculate grid dimensions
    grid_cols = int(math.ceil(math.sqrt(num_vis)))
    grid_rows = int(math.ceil(num_vis / grid_cols))

    scene = renderer.scene
    vopt = mujoco.MjvOption()
    pert = mujoco.MjvPerturb()
    catmask = mujoco.mjtCatBit.mjCAT_DYNAMIC

    # Initialize base data to capture static elements (e.g., floor)
    data = mujoco.MjData(model)
    data.qpos, data.qvel = state.q[0], state.qd[0]
    mujoco.mj_forward(model, data)

    # Set up the camera to encompass the entire grid
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0, (grid_rows - 1) * spacing / 2, 0]  # Adjust Z as needed

    camera.distance = max(grid_cols, grid_rows) * spacing # Zoom out to fit all models
    camera.azimuth = 0
    camera.elevation = -30 # Look down at the models

    # Update the scene with static elements
    renderer.update_scene(data, camera=camera)

    # Loop over each model in the batch and add its geometry to the scene
    for idx in range(1, num_vis):
      data.qpos, data.qvel = state.q[idx], state.qd[idx]

      # Compute grid position
      row = idx // grid_cols
      col = idx % grid_cols
      x_shift = col * spacing
      y_shift = row * spacing

      # Shift the model's position
      data.qpos[0] += x_shift
      data.qpos[1] += y_shift
      # Add the model's geoms to the scene
      mujoco.mj_forward(model, data)
      mujoco.mjv_addGeoms(model, data, vopt, pert, catmask, scene)
  return renderer.render()


def render_array(
    sys: brax.System,
    trajectory: Union[List[base.State], base.State],
    height: int = 240,
    width: int = 320,
    camera: Optional[str] = None,
) -> Union[Sequence[np.ndarray], np.ndarray]:
  """Returns a sequence of np.ndarray images using the MuJoCo renderer."""
  
  camera = camera or -1
  # if isinstance(trajectory, list):
  #   return [get_image(s) for s in trajectory]

  if isinstance(trajectory, list):
      # Prepare arguments for multiprocessing
      args = [(sys.mj_model, state, height, width) for state in trajectory]
      # with mp.Pool(processes=32) as pool:
          # Use imap for progress tracking with tqdm
          # frames = list(tqdm(pool.imap(get_image, args), total=len(trajectory), desc="Rendering frames"))
      frames = Parallel(n_jobs=-1)(
            delayed(get_image)(*arg) for arg in tqdm(args, desc="Rendering frames")
        )
      # frames = [get_image(*arg) for arg in tqdm(args, desc="Rendering frames")]
      return frames


  return get_image(sys, trajectory, height, width, camera)


def render(
    sys: brax.System,
    trajectory: List[base.State],
    height: int = 240,
    width: int = 320,
    camera: Optional[str] = None,
    fmt: str = 'png',
) -> bytes:
  """Returns an image of a brax System."""
  if not trajectory:
    raise RuntimeError('must have at least one state')

  frames = render_array(sys, trajectory, height, width, camera)
  frames = [Image.fromarray(image) for image in frames]

  f = io.BytesIO()
  if len(frames) == 1:
    frames[0].save(f, format=fmt)
  else:
    frames[0].save(
        f,
        format=fmt,
        append_images=frames[1:],
        save_all=True,
        duration=sys.opt.timestep * 1000,
        loop=0,
    )

  return f.getvalue()

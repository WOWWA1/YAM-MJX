Tanjeed: I cloned the `mujoco_playground` folder that you see. Basic usage can
be found in the MuJoCo Playground repo/docs.

To see that the YAM arm is loaded and functional, run these commands from this
clone:

```bash
cd /Users/tanjeedalam/projects/yamproject/YAM-MJX/mujoco_playground
UV_CACHE_DIR=.uv-cache uv venv --python 3.12
source .venv/bin/activate
UV_CACHE_DIR=.uv-cache uv pip install -e .
python scripts/view_yam.py
```

The YAM scripts load the robot model from the local MuJoCo Menagerie assets.
If those assets are missing, they are downloaded into
`mujoco_playground/mujoco_playground/external_deps/`. The viewer expects this
file to exist afterward:

```text
mujoco_playground/mujoco_playground/external_deps/mujoco_menagerie/i2rt_yam/scene.xml
```

The same YAM + cube scene is also registered as a MuJoCo Playground environment:

```bash
python - <<'PY'
import jax
from mujoco_playground import manipulation

env = manipulation.load("YamTiptopCube")
state = env.reset(jax.random.PRNGKey(0))
state = env.step(state, state.data.ctrl)

print("action_size:", env.action_size)
print("observation_size:", env.observation_size)
print("gripper_cube_distance:", float(state.metrics["gripper_cube_distance"]))
PY
```

This environment is meant for scripted control, diagnostics, rendering, and
eventually replaying TiPToP-style plans. It uses MuJoCo Playground's env API,
but it does not require training a policy.

To create a simulator observation for TiPToP:

```bash
python scripts/save_tiptop_h5_from_yam.py \
  --output /tmp/yam_tiptop_obs.h5 \
  --preview-png /tmp/yam_tiptop_obs.png
```

To run TiPToP planning from that observation, `torch`, `tiptop`, and `cutamp`
must be installed or importable from this environment. If TiPToP is in a local
checkout, set `TIPTOP_PACKAGE_DIR` to that checkout before running:

```bash
TIPTOP_PACKAGE_DIR=/path/to/tiptop \
python scripts/run_tiptop_h5_yam_debug.py \
  --h5-path /tmp/yam_tiptop_obs.h5 \
  --task-instruction "pick up the red cube" \
  --output-dir /tmp/tiptop_yam_debug \
  --yam-sim-bootstrap \
  --joint-space-fallback
```

To replay a generated plan in the MuJoCo YAM simulator:

```bash
python scripts/replay_tiptop_plan_yam.py --plan /tmp/tiptop_yam_debug/<run>/tiptop_plan.json
```

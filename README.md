# YAM MuJoCo Playground + TiPToP

This checkout connects the YAM MuJoCo model to TiPToP's offline simulator
workflow. It is simulator-only: the scripts here do not command the real robot.

## Pipeline

The current workflow follows TiPToP's documented offline H5 simulation mode:

```text
MuJoCo YAM + cube scene
  -> render RGB-D observation and save H5
  -> run TiPToP perception + planning on the H5
  -> save tiptop_plan.json
  -> replay the generated plan in a MuJoCo YAM scene
```

The main scripts are:

- `mujoco_playground/scripts/view_yam.py`: opens the raw YAM model.
- `mujoco_playground/scripts/save_tiptop_h5_from_yam.py`: renders RGB-D from the YAM + cube scene and writes the H5 observation TiPToP expects.
- `mujoco_playground/scripts/run_tiptop_h5_yam_debug.py`: simulator-only wrapper around TiPToP H5 planning with YAM-specific debug flags.
- `mujoco_playground/scripts/replay_tiptop_plan_yam.py`: replays a saved `tiptop_plan.json` in MuJoCo.
- `mujoco_playground/scripts/diagnose_yam_tool_frame.py`: compares MuJoCo YAM frames against cuRobo/cuTAMP frames.

The same YAM + cube setup is also registered as a MuJoCo Playground environment:

```python
from mujoco_playground import manipulation

env = manipulation.load("YamTiptopCube")
```

That environment is useful for scripted control, diagnostics, rendering, and
eventually a more integrated observe-plan-execute loop. It does not require
training a policy.

## Fresh Linux/NVIDIA Setup

For a fresh Linux/NVIDIA machine or pod, use the top-level bootstrap scripts.
This is the preferred setup path. It follows TiPToP's documented install flow
for TiPToP/cuRobo/cuTAMP and TiPToP's M2T2 fork, then applies the minimal YAM
bridge patches from this repo.

### 1. Clone and bootstrap

```bash
git clone https://github.com/WOWWA1/YAM-MJX.git
cd YAM-MJX

# Recommended on RunPod/root containers so large envs avoid slower or
# quota-limited network volumes. If /opt is not writable, skip this line and
# the scripts will use ~/.cache/yam-mjx.
export YAM_LOCAL=/opt/yam-local

scripts/bootstrap_yam_stack.sh
source "$YAM_LOCAL/yam_env.sh"
scripts/check_yam_stack.sh
```

The bootstrap uses:

- TiPToP: `https://github.com/tiptop-robot/tiptop.git`
- M2T2: `https://github.com/williamshen-nz/M2T2.git`

Heavy generated environments and caches are created under `$YAM_LOCAL` and
symlinked into the repo before installation. This keeps paths stable and avoids
the broken-environment problems that happen when pixi or Python environments
are moved after creation.

If `scripts/check_yam_stack.sh` passes, the machine is ready.

### 2. Start M2T2

Use a dedicated terminal and leave it running:

```bash
cd YAM-MJX
source "$YAM_LOCAL/yam_env.sh"
scripts/start_m2t2_server.sh
```

M2T2 serves grasp proposals over HTTP for TiPToP. If you want to check it from
another terminal:

```bash
curl http://localhost:8123/health
```

### 3. Run the YAM simulation flow

In a second terminal:

```bash
cd YAM-MJX
source "$YAM_LOCAL/yam_env.sh"

scripts/make_tiptop_h5.sh

export GOOGLE_API_KEY="your-key-here"
scripts/run_tiptop_yam.sh
scripts/render_tiptop_latest.sh
```

The default rendered video is written to:

```text
/tmp/yam_tiptop_replay.mp4
```

If M2T2 weights fail to download because of Hugging Face/Git LFS limits, copy
`m2t2.pth` into:

```text
$YAM_ROOT/external/M2T2/weights/m2t2.pth
```

If MuJoCo EGL fails with an `eglQueryString`/`NoneType` error, the machine is
usually missing the system EGL library. On Ubuntu:

```bash
apt-get update
apt-get install -y libegl1
```

## Manual Install Notes

The scripted Linux setup above is preferred. The notes below are older manual
commands kept for debugging individual pieces when the bootstrap is not enough.

You need two environments:

- MuJoCo Playground venv for rendering and replay:
  `~/yam-mujocoplayground/mujoco_playground/.venv`
- TiPToP pixi env for TiPToP, cuTAMP, cuRobo, Gemini, SAM2, and M2T2:
  `~/yam-tamp/tiptop/tiptop`

If the MuJoCo Playground venv is not installed yet:

```bash
cd ~/yam-mujocoplayground/mujoco_playground
UV_CACHE_DIR=.uv-cache uv venv --python 3.12
source .venv/bin/activate
UV_CACHE_DIR=.uv-cache uv pip install -e .
```

The YAM scripts use MuJoCo Menagerie assets. If they are missing, the helper
scripts download them into:

```text
mujoco_playground/mujoco_playground/external_deps/mujoco_menagerie/
```

TiPToP needs a Gemini API key in the terminal where you run TiPToP:

```bash
export GOOGLE_API_KEY="your-key-here"
```

Do not commit or paste API keys into files.

## Manual Terminal Layout

### Terminal 1: M2T2 Server

TiPToP uses M2T2 for grasp proposals. Keep this running:

```bash
cd ~/yam-tamp/tiptop/M2T2
pixi run server
```

Health check from another terminal:

```bash
curl http://localhost:8123/health
```

### Terminal 2: Create The H5 Observation

Render one RGB-D observation from the YAM + cube simulator scene:

```bash
cd ~/yam-mujocoplayground/mujoco_playground
source .venv/bin/activate

python scripts/save_tiptop_h5_from_yam.py \
  --output /tmp/yam_tiptop_obs.h5 \
  --preview-png /tmp/yam_tiptop_obs.png \
  --fovy 24 \
  --camera-pos 0.65,-0.30,0.42 \
  --camera-target 0.45,0.0,0.025 \
  --cube-pos 0.45,0.0,0.025
```

The preview image is useful for checking that the cube is visible and not hidden
inside the gripper.

### Terminal 3: Run TiPToP On The H5

Run from the TiPToP Python package directory. This avoids the repo-root
`cutamp/` folder shadowing the installed cuTAMP package:

```bash
cd ~/yam-tamp/tiptop/tiptop/tiptop
export GOOGLE_API_KEY="your-key-here"
```

Confirm the key is visible:

```bash
pixi run python -c "import os; print(bool(os.environ.get('GOOGLE_API_KEY')))"
```

For the current YAM simulator bootstrap, this produces a replayable JSON plan
even while the YAM/cuRobo motion-planning path is still being tuned:

```bash
pixi run python ~/yam-mujocoplayground/mujoco_playground/scripts/run_tiptop_h5_yam_debug.py \
  --h5-path /tmp/yam_tiptop_obs.h5 \
  --task-instruction "pick up the red cube" \
  --output-dir /tmp/tiptop_yam_sim_bootstrap \
  --num-particles 512 \
  --max-planning-time 60 \
  --disable-m2t2-grasps \
  --yam-sim-bootstrap \
  --tool-frame-mode measured-grasp-site \
  --ignore-robot-world-collision \
  --joint-space-fallback
```

The output will be under a timestamped directory, for example:

```text
/tmp/tiptop_yam_sim_bootstrap/2026-07-01_15-54-26/tiptop_plan.json
```

### Terminal 4: Replay The Plan In MuJoCo

Use the MuJoCo Playground venv:

```bash
cd ~/yam-mujocoplayground/mujoco_playground
source .venv/bin/activate

python scripts/replay_tiptop_plan_yam.py \
  --plan /tmp/tiptop_yam_sim_bootstrap/<timestamp>/tiptop_plan.json
```

Press `R` in the MuJoCo viewer to replay the plan.

## Useful Checks

View the raw YAM model:

```bash
cd ~/yam-mujocoplayground/mujoco_playground
source .venv/bin/activate
python scripts/view_yam.py
```

Check that the Playground environment loads:

```bash
cd ~/yam-mujocoplayground/mujoco_playground
source .venv/bin/activate

python - <<'CHECK_ENV'
import jax
from mujoco_playground import manipulation

env = manipulation.load("YamTiptopCube")
state = env.reset(jax.random.PRNGKey(0))
state = env.step(state, state.data.ctrl)

print("action_size:", env.action_size)
print("observation_size:", env.observation_size)
print("gripper_cube_distance:", float(state.metrics["gripper_cube_distance"]))
CHECK_ENV
```

Compare MuJoCo and cuRobo YAM frames:

```bash
cd ~/yam-mujocoplayground

PYTHONPATH=~/yam-tamp/tiptop/tiptop/.pixi/envs/default/lib/python3.12/site-packages/rerun_sdk:~/yam-tamp/tiptop/tiptop/cutamp:~/yam-tamp/tiptop/tiptop/curobo/src:~/yam-tamp/tiptop/tiptop/.pixi/envs/default/lib/python3.12/site-packages:~/yam-tamp/tiptop/tiptop/tiptop \
LD_LIBRARY_PATH=~/yam-tamp/tiptop/tiptop/.pixi/envs/default/lib \
./mujoco_playground/.venv/bin/python mujoco_playground/scripts/diagnose_yam_tool_frame.py
```

## Current Debugging Status

The H5/perception side works: TiPToP detects and segments the red cube from the
MuJoCo-rendered RGB-D observation.

The YAM planning path is still being tuned. The important debug flags are:

- `--tool-frame-mode measured-grasp-site`: uses a measured transform that aligns
  cuRobo's YAM `ee_link` with MuJoCo's `grasp_site` better than the placeholder.
- `--ignore-robot-world-collision`: relaxes only the cuTAMP
  `Collision/robot_to_world` constraint. This showed that perceived table/world
  collision is one blocker.
- `--drop-static-world`: replaces the perceived static table with a tiny
  far-away dummy obstacle so cuRobo can be tested without table collision.
- `--joint-space-fallback`: writes a direct joint-space fallback plan for MuJoCo
  replay when cuRobo refinement fails. Use this only for simulator bootstrap
  testing, not as a real robot plan.

To debug the real cuRobo path, remove `--joint-space-fallback` and inspect the
constraint logs. A healthy no-fallback run should first get nonzero satisfying
particles, then reach `Trying cuRobo planning...`.

## Common Problems

`could not find pixi.toml`

Run `pixi` commands from the TiPToP checkout, not from this repo:

```bash
cd ~/yam-tamp/tiptop/tiptop/tiptop
```

`cuTAMP version mismatch: required 0.0.5, found <0.0.2`

You are probably running from the TiPToP repo root and Python is importing the
local source checkout of `cutamp`. Change into the TiPToP package directory:

```bash
cd ~/yam-tamp/tiptop/tiptop/tiptop
```

`No API key was provided`

Set `GOOGLE_API_KEY` in the same terminal where you run `pixi run ...`.

`Cannot connect to host localhost:8123`

Start the M2T2 server in Terminal 1.

`FileNotFoundError: H5 file not found`

Regenerate `/tmp/yam_tiptop_obs.h5`; files in `/tmp` may disappear between
sessions.

`Segmentation fault (core dumped)` after outputs are saved

This has been happening during GPU/library cleanup after TiPToP saves outputs.
If the log says `Saved TiPToP plan` or `Saved outputs`, inspect the saved
directory before treating the segfault as the primary failure.

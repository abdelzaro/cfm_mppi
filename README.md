<div align="center">

## **Unified Generation-Refinement Planning: Bridging Guided Flow Matching and Sampling-Based MPC for Social Navigation**

[![Project Page](https://img.shields.io/badge/Project--Page-blue)](https://cfm-mppi.github.io/)
[![Paper](https://img.shields.io/badge/ArXiv-Paper-red)](https://arxiv.org/abs/2508.01192)

**[Kazuki Mizuta](https://m-kazuki.github.io/)**<sup>1</sup> · **[Karen Leung](https://faculty.washington.edu/kymleung/)**<sup>1,2</sup>

<sup>1</sup>University of Washington &nbsp;&nbsp;&nbsp; <sup>2</sup>NVIDIA

<img src="media/hero.png" alt="Hero Image" width="70%"/>

</div>

---

## Installation

To set up the environment, run the following commands:

```bash
conda env create -f conda_environment.yaml
conda activate cfm_mppi
pip install -e . 
```

## Usage

### Standalone pretrained inference check

Run one CFM + MPPI planning call without ROS, Arena, plotting, or training:

```bash
cd /home/azaro/cfm_mppi
conda activate cfm_mppi
python -m cfm_mppi.evaluation.smoke_inference --device cpu
```

This loads the pretrained checkpoint and UCY scenario 110, uses 20 candidates
and an 80-step horizon, and validates the returned control sequences. It prints
the first unicycle control `[v, omega]` and inference time. To use the evaluator's
200-candidate setting and save the result:

```bash
python -m cfm_mppi.evaluation.smoke_inference --device cpu --samples 200 \
  --output /tmp/cfm_mppi_reference_inference.json
```

Use `--device cuda` with a working CUDA environment, or `--device auto` to select
the available device. Model and dataset paths are resolved from the module's
location. The command does not publish robot commands.

Check consecutive planning calls and reset reproducibility:

```bash
python -m cfm_mppi.evaluation.smoke_inference --device cpu --steps 12 \
  --check-reset --output /tmp/cfm_mppi_repeated_inference.json
```

Repeat with no obstacle tracks:

```bash
python -m cfm_mppi.evaluation.smoke_inference --device cpu --steps 12 \
  --empty-scene --check-reset --output /tmp/cfm_mppi_empty_inference.json
```

These checks use the reference evaluator's finite, shrinking planning horizon
and a numerical unicycle rollout. They validate finite, bounded controls on every
call. The reset check recreates the MPPI solver, all four history buffers, noise,
and numerical state; with the same seed and observations, full control sequences
must match the initial episode. The pretrained model stays loaded. `--steps`
accepts 1 through 80; this is not yet an indefinitely running controller.

Both 12-call CPU checks passed with 20 candidates, including history-buffer
rollover at 10 entries and matching reset replays (48 planning calls total).
Warm calls were approximately 0.46–0.53 seconds on this machine. This validates
offline inference behavior, not 10 Hz operation or Arena navigation. An Arena
bridge still needs observation/frame conversion, stale-result handling, and a
deliberate policy for maintaining or restarting the planning horizon.

### Arena worker

The separate ROS integration uses `cfm_mppi.evaluation.arena_worker`: a persistent
pretrained process with newline-delimited JSON requests and responses on
stdin/stdout. It is started by Arena's `cfm_mppi_bridge.py`. It uses a fresh
80-step plan per request, accepts map-frame robot `[x,y,yaw]` and goal `[x,y]`,
and currently assumes an empty scene. Arena owns timing, reset-generation checks,
command bounds, and MBF routing. See
`/home/azaro/arena_ws/src/arena/arena-rosnav/arena_bringup/CFM_MPPI_BRIDGE.md`
for the tested launch and enable commands.

For Arena's separate `cfm_mppi_continuous` option, a worker request with
`"sequence": true` returns 80 controls, 81 predicted unicycle poses, and dt=0.1.
This mode applies [0, 0.1] m/s and [-0.3, 0.3] rad/s bounds inside MPPI, allowing
the ROS side to align delayed plans with measured motion. The original
single-action mode remains available. See Arena's `CFM_CONTINUOUS.md` for the
empty-scene goal-approach results and commands.

The obstacle-aware Arena option additionally sends `"obstacles"` as map-frame
`[x,y,vx,vy]` tracks and `"collision_radius"`. Omitted tracks retain the empty
baseline. `"reverse": true` permits sequence-mode speed down to -0.1 m/s;
sequence mode remains forward-only by default. See Arena's `CFM_OBSTACLES.md`
for the adapters, scenarios, and current validation status.

### Training

To train the CFM (Conditional Flow Matching) model, run:

```bash
python cfm_mppi/train.py
```

### Demos
We provide Jupyter notebook demos for different dynamics models:
- **Unicycle dynamics:** [`cfm_mppi/example/unicycle.ipynb`](cfm_mppi/example/unicycle.ipynb)
- **Double integrator dynamics:** [`cfm_mppi/example/doubleintegrator.ipynb`](cfm_mppi/example/doubleintegrator.ipynb)

## Pre-trained Models and Datasets

For quick evaluation, download the pre-trained weights and datasets from [Google Drive](https://drive.google.com/drive/folders/1gggsHGnLyh_rsaEw91EaqIO8I4Asqh64?usp=sharing) and organize them into the following directory structure:

- `dataset/` &rarr; `./dataset/`
- `checkpoint.pt` &rarr; `./output_dir/cfm_transformer/checkpoint.pth`
- `args.json` &rarr; `./output_dir/cfm_transformer/args.json`

## Citation

If you find our work useful in your research, please consider citing:

```bibtex
@inproceedings{MizutaLeung2026,
  title = {Unified Generation-Refinement Planning: Bridging Guided Flow Matching and Sampling-Based MPC for Social Navigation},
  author = {Kazuki Mizuta and Karen Leung},
  booktitle = {Proc.\ IEEE Conf.\ on Robotics and Automation},
  year = {2026},
}
```

## Acknowledgements

This repository is built upon several excellent open-source projects:
- [flow_matching](https://github.com/facebookresearch/flow_matching)
- [latent-diffusion](https://github.com/CompVis/latent-diffusion/tree/main)
- [diffuser](https://github.com/jannerm/diffuser/tree/maze2d)
- [trajdata](https://github.com/NVlabs/trajdata/tree/main)
- [mppi_playground](https://github.com/kohonda/mppi_playground)

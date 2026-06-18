# Imitation Learning Playground

Hands-on imitation learning on [robomimic](https://robomimic.github.io/) manipulation
tasks (robosuite + MuJoCo), built up from a plain behavior-cloning baseline to a
multimodal GMM policy and an Action-Chunking Transformer (ACT). Each notebook ties the
implementation back to the theory (BC as MLE, covariate shift / `O(εT²)` compounding,
mixture-density heads, the ACT CVAE/ELBO + temporal ensembling).

## Notebooks

Rollouts are saved as **GIFs under [`media/`](media/)** and shown via markdown image cells, so
the plots *and* the rollout videos render directly on GitHub. (GitHub renders neither Jupyter's
base64 `<video>` nor inline `image/gif` outputs, but it *does* render relative markdown images —
so the notebooks `save_gif(...)` to a file and embed it with `![](media/…gif)`.)

| Notebook | What it does |
|---|---|
| [`bc_playground.ipynb`](bc_playground.ipynb) | **BC baseline** (MLP, MSE) on robomimic `lift`. Train, roll out closed-loop, and watch covariate shift in the success-rate grid. |
| [`bc_playground-can.ipynb`](bc_playground-can.ipynb) | **GMM (mixture-density) head** on the harder `can` task, trained by NLL — models a *multimodal* `p(a\|s)` instead of mode-averaging. Includes a `ph`↔`mh` dataset toggle (the GMM's win shows up on multi-human data). |
| [`act_playground-can.ipynb`](act_playground-can.ipynb) | **ACT**: a CVAE that predicts a *chunk* of `k` actions (transformer encoder/decoder) with **temporal ensembling** at rollout. |

Shared plumbing lives in [`bc_harness.py`](bc_harness.py) (data loading, training loop,
closed-loop rollout/eval, GIF display) and [`act_harness.py`](act_harness.py) (action-chunk
dataset, ELBO training, temporal-ensembling rollout). The policies/models themselves live
in the notebooks — edit them freely.

## Setup

The environment is a Nix flake that bootstraps a Python venv pinned for Pascal GPUs
(torch 2.7 / CUDA 12.6 — newer wheels drop `sm_61`).

```bash
# 1. clone WITH the robomimic submodule
git clone --recursive <your-fork-url> imitation_learning
cd imitation_learning
#    (already cloned without --recursive? run: git submodule update --init --recursive)

# 2. enter the dev shell -> first run auto-bootstraps .venv (downloads torch cu126, robosuite,
#    robomimic editable, jupyterlab). CUDA is verified at the end.
nix develop

# 3. download the demo datasets (lift + can, proficient-human, low-dim state)
./download_data.sh
#    for the multi-human can set used by the GMM toggle:
#    python robomimic/robomimic/scripts/download_datasets.py --tasks can --dataset_types mh --hdf5_types low_dim

# 4. launch
nix develop --command jupyter lab
```

`robomimic` is a **git submodule** pinned to the upstream commit this project was built
against; the datasets it downloads live under `robomimic/datasets/` and are gitignored
(research-only, re-download after a fresh clone).

## Stack

robomimic (IL framework) → robosuite 1.5 (Panda arm, `PickPlaceCan`/`Lift`) → MuJoCo 3.9
(physics + offscreen EGL rendering). Actions are 7-D operational-space end-effector deltas
+ gripper; observations are low-dim state (eef pose, gripper, object).

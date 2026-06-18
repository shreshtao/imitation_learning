#!/usr/bin/env python
"""
Milestone A -- Behavior Cloning (the MLE baseline) on robomimic lift/can (state-based).

Builds a plain-BC config in code (NOT bc_rnn) and trains via robomimic's train() loop.
BC here is maximum likelihood: a feed-forward policy pi_theta(a|s) fit to the expert's
state-conditional actions. The default robomimic "bc" algo is an MLP with an MSE/log-prob
regression head -- exactly the Gaussian-NLL == MSE view from the theory.

Run a quick smoke test (no sim needed -- rollouts off):
    python train_bc.py --epochs 5

Train + closed-loop eval (success rate, headless, no video):
    python train_bc.py --epochs 50 --rollout --rollout-n 20

Usage notes: run inside `nix develop` (so the venv + CUDA libs are active).
"""
import argparse
import os

import robomimic.utils.torch_utils as TorchUtils
from robomimic.config import config_factory
from robomimic.scripts.train import train

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET = os.path.join(HERE, "robomimic", "datasets", "lift", "ph", "low_dim_v15.hdf5")


def build_config(args):
    # default BC algo == feed-forward MLP policy with a regression head
    config = config_factory(algo_name="bc")

    # ----- experiment -----
    config.experiment.name = args.name
    config.experiment.validate = args.validate
    config.experiment.logging.terminal_output_to_txt = False   # keep stdout visible
    config.experiment.logging.log_tb = True
    config.experiment.logging.log_wandb = False

    # epoch = a fixed number of gradient steps (not a full dataset pass)
    config.experiment.epoch_every_n_steps = args.steps
    config.experiment.validation_epoch_every_n_steps = 10

    # ----- checkpoint saving -----
    config.experiment.save.enabled = True
    config.experiment.save.every_n_epochs = args.epochs        # save the final model at least
    config.experiment.save.on_best_validation = False
    config.experiment.save.on_best_rollout_return = False
    config.experiment.save.on_best_rollout_success_rate = args.rollout

    # ----- closed-loop rollout eval (covariate-shift check) -----
    config.experiment.rollout.enabled = args.rollout
    config.experiment.render = False
    config.experiment.render_video = False                     # headless: success rate only, no GL needed
    if args.rollout:
        config.experiment.rollout.n = args.rollout_n
        config.experiment.rollout.horizon = args.rollout_horizon
        config.experiment.rollout.rate = args.epochs           # roll out once, at the end
        config.experiment.rollout.warmstart = 0
        config.experiment.rollout.terminate_on_success = True

    # ----- data / loader -----
    config.train.data = args.dataset
    config.train.output_dir = args.output
    config.train.num_data_workers = 0
    config.train.hdf5_cache_mode = "all"                       # small dataset -> cache in RAM
    config.train.hdf5_use_swmr = True
    config.train.hdf5_normalize_obs = False
    config.train.hdf5_filter_key = "train" if args.validate else None
    config.train.hdf5_validation_filter_key = "valid" if args.validate else None

    # ----- learning -----
    config.train.cuda = not args.cpu
    config.train.batch_size = args.batch_size
    config.train.num_epochs = args.epochs
    config.train.seed = args.seed

    # ----- observations: robosuite low-dim keys present in lift/can low_dim datasets -----
    config.observation.modalities.obs.low_dim = [
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
        "object",
    ]
    config.observation.modalities.obs.rgb = []
    config.observation.modalities.goal.low_dim = []
    config.observation.modalities.goal.rgb = []
    return config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--output", default=os.path.join(HERE, "bc_output"))
    p.add_argument("--name", default="lift_bc")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--steps", type=int, default=100, help="gradient steps per epoch")
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--validate", action="store_true")
    p.add_argument("--rollout", action="store_true", help="run closed-loop eval (needs robosuite env)")
    p.add_argument("--rollout-n", type=int, default=20)
    p.add_argument("--rollout-horizon", type=int, default=400)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    assert os.path.isfile(args.dataset), f"dataset not found: {args.dataset}"
    config = build_config(args)
    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)
    train(config, device=device)


if __name__ == "__main__":
    main()

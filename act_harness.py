"""
act_harness.py -- Action Chunking Transformer (ACT) testing utils.

ACT = a CVAE whose decoder predicts a CHUNK of k future actions conditioned on the
current observation and a latent z. Training objective is the ELBO:

    L = L1(pred_chunk, expert_chunk)  +  beta * KL( q(z | obs, chunk) || N(0, I) )

At test time z = 0 (the prior mean) and we use TEMPORAL ENSEMBLING: query the model
every step (each query predicts k future actions) and blend the overlapping
predictions for the current timestep with an exponential weighting.

This module REUSES bc_harness for env creation / video / state-building, and adds:
  - load_chunk_dataset : (state, action-chunk, pad-mask) tensors from a robomimic hdf5
  - train_act          : ELBO training loop with a live loss plot
  - rollout_act / evaluate_act : closed-loop eval with temporal ensembling

The ACT *model* itself lives in the notebook (the "edit freely" part), mirroring how
the BC policy lives in bc_playground. The model must expose:
    compute_loss(s, a_chunk, is_pad, kl_weight) -> (total_loss, {"recon":.., "kl":..})
    predict_chunk(s) -> [B, k, action_dim]      (z = 0 inside)
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("__EGL_VENDOR_LIBRARY_DIRS", "/usr/share/glvnd/egl_vendor.d")

from dataclasses import dataclass

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# reuse the BC plumbing -- env, rendering, state-building, video display
from bc_harness import (DEFAULT_OBS_KEYS, build_state, make_env, _seed_env,
                        show_video, show_video_grid, show_gif, show_gif_grid,
                        save_gif, save_gif_grid)


# --------------------------------------------------------------------------- #
# Data -- (state, action-chunk, pad-mask)
# --------------------------------------------------------------------------- #
@dataclass
class ACTData:
    states: np.ndarray          # [N, obs_dim]      z-score before the model
    action_chunks: np.ndarray   # [N, k, action_dim]   raw actions
    is_pad: np.ndarray          # [N, k] bool       True = past episode end (ignore)
    obs_keys: tuple
    obs_dim: int
    action_dim: int
    chunk_size: int
    obs_mean: np.ndarray
    obs_std: np.ndarray
    dataset_path: str
    n_demos: int

    def normalize(self, s):
        return (s - self.obs_mean) / self.obs_std


def load_chunk_dataset(dataset_path, obs_keys=DEFAULT_OBS_KEYS, chunk_size=16, filter_key=None):
    """Load (state, action-chunk, pad-mask) triples from a robomimic low_dim HDF5.

    Chunks are built PER DEMO so a chunk never crosses an episode boundary; a chunk
    that runs past the episode end is padded with the last action and flagged in
    is_pad (the recon loss and the encoder ignore padded steps).
    """
    obs_keys = tuple(obs_keys)
    states, chunks, pads = [], [], []
    with h5py.File(dataset_path, "r") as f:
        if filter_key is not None:
            demos = [d.decode() if isinstance(d, bytes) else d for d in f["mask"][filter_key][:]]
        else:
            demos = list(f["data"].keys())
        for d in demos:
            g = f["data"][d]
            S = np.concatenate([g["obs"][k][:] for k in obs_keys], axis=1).astype(np.float32)
            A = g["actions"][:].astype(np.float32)
            T, adim = A.shape
            for t in range(T):
                end = min(t + chunk_size, T)
                ch = np.empty((chunk_size, adim), np.float32)
                ch[:end - t] = A[t:end]
                ch[end - t:] = A[T - 1]                 # pad with the last action
                pad = np.zeros(chunk_size, dtype=bool)
                pad[end - t:] = True
                states.append(S[t]); chunks.append(ch); pads.append(pad)
    states = np.asarray(states, np.float32)
    chunks = np.asarray(chunks, np.float32)
    pads = np.asarray(pads, dtype=bool)
    obs_mean = states.mean(axis=0); obs_std = states.std(axis=0) + 1e-6
    return ACTData(
        states=states, action_chunks=chunks, is_pad=pads, obs_keys=obs_keys,
        obs_dim=states.shape[1], action_dim=chunks.shape[2], chunk_size=chunk_size,
        obs_mean=obs_mean, obs_std=obs_std,
        dataset_path=os.path.abspath(dataset_path), n_demos=len(demos),
    )


# --------------------------------------------------------------------------- #
# Training (ELBO, with live loss plot)
# --------------------------------------------------------------------------- #
def _live_plot_act(history, title="ACT training"):
    from IPython.display import clear_output
    import matplotlib.pyplot as plt
    clear_output(wait=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(history["epoch"], history["total"], label="total (ELBO)")
    ax[0].plot(history["epoch"], history["recon"], label="recon (L1)")
    if history.get("val_recon"):
        ax[0].plot(history["epoch"], history["val_recon"], "--", label="val recon")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].set_title(title)
    ax[0].legend(); ax[0].grid(True, alpha=0.3)
    ax[1].plot(history["epoch"], history["kl"], color="tab:green", label="KL(z)")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("KL(q || N(0,I))")
    ax[1].legend(); ax[1].grid(True, alpha=0.3)
    plt.show()


def train_act(model, data, *, epochs=100, batch_size=256, lr=1e-4, weight_decay=1e-4,
              kl_weight=10.0, grad_clip=1.0, device=None, plot=True, plot_every=1,
              val_frac=0.0, seed=0):
    """Fit the ACT CVAE by the ELBO. States are z-scored; action chunks are raw.

    kl_weight (beta): trades off action accuracy vs. a tidy latent prior. ACT uses ~10.
    Returns a history dict {"epoch","total","recon","kl","val_recon"}.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed); np.random.seed(seed)
    model.to(device)

    Sn = data.normalize(data.states).astype(np.float32)
    S = torch.from_numpy(Sn)
    A = torch.from_numpy(data.action_chunks)
    P = torch.from_numpy(data.is_pad)
    N = len(S)
    idx = np.random.permutation(N)
    n_val = int(N * val_frac)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    tr = DataLoader(TensorDataset(S[tr_idx], A[tr_idx], P[tr_idx]),
                    batch_size=batch_size, shuffle=True, drop_last=True)
    val = None
    if n_val > 0:
        val = (S[val_idx].to(device), A[val_idx].to(device), P[val_idx].to(device))

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = {k: [] for k in ("epoch", "total", "recon", "kl", "val_recon")}
    for ep in range(1, epochs + 1):
        model.train(); tot = rec = kl = 0.0; nb = 0
        for s, a, p in tr:
            s, a, p = s.to(device), a.to(device), p.to(device)
            # l1 loss + kl-divergence
            loss, parts = model.compute_loss(s, a, p, kl_weight=kl_weight)
            opt.zero_grad(); loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            tot += loss.item(); rec += parts["recon"]; kl += parts["kl"]; nb += 1
        history["epoch"].append(ep)
        history["total"].append(tot / nb); history["recon"].append(rec / nb); history["kl"].append(kl / nb)
        if val is not None:
            model.eval()
            with torch.no_grad():
                _, vp = model.compute_loss(*val, kl_weight=kl_weight)
            history["val_recon"].append(vp["recon"])
        if plot and (ep % plot_every == 0 or ep == epochs):
            _live_plot_act(history)
    return history


# --------------------------------------------------------------------------- #
# Temporal ensembling + closed-loop rollout
# --------------------------------------------------------------------------- #
class TemporalEnsembler:
    """Blend overlapping action-chunk predictions for the current timestep.

    Every env step the model predicts a fresh chunk of k actions covering steps
    t..t+k-1. So at time t we hold up to k predictions for that one step (made at
    times t-k+1..t). We combine them with weights exp(-m * age), oldest-first --
    the original ACT scheme -- which smooths the trajectory and damps the jitter you
    get from naive open-loop chunk execution.
    """
    def __init__(self, horizon, chunk_size, action_dim, m=0.01, device="cpu"):
        self.k, self.m, self.t = chunk_size, m, 0
        self.buf = torch.zeros(horizon, horizon + chunk_size, action_dim, device=device)
        self.filled = torch.zeros(horizon, horizon + chunk_size, dtype=torch.bool, device=device)

    def step(self, chunk):
        """chunk: [k, action_dim] predicted now (time self.t). Returns the action for time self.t."""
        t, k = self.t, self.k
        self.buf[t, t:t + k] = chunk
        self.filled[t, t:t + k] = True
        preds = self.buf[:t + 1, t][self.filled[:t + 1, t]]        # all predictions targeting time t
        # oldest action prediction gets highest weight -> this is to preserve smoothness (trusting newer actions results in jerkiness)
        w = torch.exp(-self.m * torch.arange(len(preds), device=preds.device))  
        w = w / w.sum()
        action = (preds * w[:, None]).sum(0)
        self.t += 1
        return action


@torch.no_grad()
def rollout_act(model, env, data, *, horizon=400, chunk_size=None, temporal_agg=True, m=0.01,
                device=None, render=True, camera="agentview", img_hw=(256, 256), seed=None):
    """One closed-loop ACT episode. temporal_agg=True -> query every step + ensemble;
    temporal_agg=False -> execute each predicted chunk open-loop for k steps then re-query."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    k = chunk_size or data.chunk_size
    if seed is not None:
        np.random.seed(seed); torch.manual_seed(seed)
        _seed_env(env, seed)
    obs = env.reset()
    frames, visited, total_r, success = [], [], 0.0, False
    ens = TemporalEnsembler(horizon, k, data.action_dim, m=m, device=device) if temporal_agg else None
    chunk, chunk_i, t = None, 0, 0
    for t in range(horizon):
        s = build_state(obs, data.obs_keys); visited.append(s)
        sn = torch.from_numpy(data.normalize(s).astype(np.float32)).unsqueeze(0).to(device)
        if temporal_agg:
            a = ens.step(model.predict_chunk(sn)[0]).cpu().numpy()
        else:
            if chunk is None or chunk_i >= k:
                chunk = model.predict_chunk(sn)[0].cpu().numpy(); chunk_i = 0
            a = chunk[chunk_i]; chunk_i += 1
        obs, r, done, _ = env.step(a)
        total_r += float(r)
        if render:
            frames.append(env.render(mode="rgb_array", height=img_hw[0], width=img_hw[1], camera_name=camera))
        if env.is_success()["task"]:
            success = True
            break
        if done:
            break
    return dict(frames=frames, success=success, total_reward=total_r,
                length=t + 1, states=np.asarray(visited))


def evaluate_act(model, env, data, *, n_episodes=20, horizon=400, chunk_size=None,
                 temporal_agg=True, m=0.01, device=None, render_first=True, render_all=False,
                 render_n=0, img_hw=(256, 256), camera="agentview", seed=0):
    """Run N ACT rollouts, report success rate. Mirrors bc_harness.evaluate.

    render_n: render only the first `render_n` episodes (enough for a small grid) --
    much lighter than render_all=True. Success rate is always over all n_episodes.
    """
    episodes = []
    for i in range(n_episodes):
        do_render = render_all or (i < render_n) or (render_first and i == 0)
        ep_seed = None if seed is None else seed + i
        out = rollout_act(model, env, data, horizon=horizon, chunk_size=chunk_size,
                          temporal_agg=temporal_agg, m=m, device=device, render=do_render,
                          img_hw=img_hw, camera=camera, seed=ep_seed)
        episodes.append(out)
    sr = float(np.mean([e["success"] for e in episodes]))
    return dict(success_rate=sr, episodes=episodes,
                first_frames=episodes[0]["frames"] if episodes else [])

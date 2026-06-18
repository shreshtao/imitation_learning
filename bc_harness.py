"""
bc_harness.py -- Behavior Cloning Testing Utils
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("__EGL_VENDOR_LIBRARY_DIRS", "/usr/share/glvnd/egl_vendor.d")

import base64
import tempfile
from dataclasses import dataclass, field

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# obs keys present in the lift/can low_dim datasets (concatenated -> state vector)
DEFAULT_OBS_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
@dataclass
class BCData:
    states: np.ndarray          # [N, obs_dim]   z-score these before the policy
    actions: np.ndarray         # [N, action_dim] raw
    obs_keys: tuple
    obs_dim: int
    action_dim: int
    obs_mean: np.ndarray
    obs_std: np.ndarray
    dataset_path: str
    n_demos: int

    def normalize(self, s):
        return (s - self.obs_mean) / self.obs_std


def build_state(obs_dict, obs_keys=DEFAULT_OBS_KEYS):
    """Concatenate the obs-dict entries (in fixed order) into one 1-D state vector."""
    return np.concatenate([np.asarray(obs_dict[k], dtype=np.float32).ravel() for k in obs_keys])


def load_bc_dataset(dataset_path, obs_keys=DEFAULT_OBS_KEYS, filter_key=None):
    """Load all (state, action) pairs from a robomimic low_dim HDF5.

    filter_key: e.g. "train"/"valid" to use only that split; None = all demos.
    """
    obs_keys = tuple(obs_keys)
    states, actions = [], []
    with h5py.File(dataset_path, "r") as f:
        if filter_key is not None:
            demos = [d.decode() if isinstance(d, bytes) else d for d in f["mask"][filter_key][:]]
        else:
            demos = list(f["data"].keys())
        for d in demos:
            g = f["data"][d]
            S = np.concatenate([g["obs"][k][:] for k in obs_keys], axis=1)
            states.append(S.astype(np.float32))
            actions.append(g["actions"][:].astype(np.float32))
    states = np.concatenate(states, axis=0)
    actions = np.concatenate(actions, axis=0)
    obs_mean = states.mean(axis=0)
    obs_std = states.std(axis=0) + 1e-6
    return BCData(
        states=states, actions=actions, obs_keys=obs_keys,
        obs_dim=states.shape[1], action_dim=actions.shape[1],
        obs_mean=obs_mean, obs_std=obs_std,
        dataset_path=os.path.abspath(dataset_path), n_demos=len(demos),
    )

# --------------------------------------------------------------------------- #
# Training (with live loss plot)
# --------------------------------------------------------------------------- #
def _live_plot(history, title="BC training"):
    from IPython.display import clear_output
    import matplotlib.pyplot as plt
    clear_output(wait=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history["epoch"], history["loss"], label="train")
    if history.get("val_loss"):
        ax.plot(history["epoch"], history["val_loss"], label="val")
        ax.legend()
    ax.set_xlabel("epoch"); ax.set_ylabel("BC loss")
    series = history["loss"] + (history.get("val_loss") or [])
    if series and min(series) > 0:                 # NLL can go negative -> log scale would break
        ax.set_yscale("log")
    ax.set_title(title); ax.grid(True, alpha=0.3)
    plt.show()


def train_bc(policy, data, *, epochs=50, batch_size=256, lr=1e-3, weight_decay=0.0,
             device=None, plot=True, plot_every=1, val_frac=0.0, seed=0):
    """Fit `policy` to (state, action) pairs by BC. States are z-scored; actions raw.

    Returns a history dict {"epoch", "loss", "val_loss"}.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed); np.random.seed(seed)
    policy.to(device)

    Sn = data.normalize(data.states).astype(np.float32)
    A = data.actions.astype(np.float32)
    N = len(Sn)
    idx = np.random.permutation(N)
    n_val = int(N * val_frac)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    S_t = torch.from_numpy(Sn); A_t = torch.from_numpy(A)
    tr = DataLoader(TensorDataset(S_t[tr_idx], A_t[tr_idx]),
                    batch_size=batch_size, shuffle=True, drop_last=True)
    val = None
    if n_val > 0:
        val = (S_t[val_idx].to(device), A_t[val_idx].to(device))

    opt = torch.optim.Adam(policy.parameters(), lr=lr, weight_decay=weight_decay)

    def loss_fn(s, a):
        # honor a policy-provided loss (e.g. Gaussian/GMM NLL); otherwise default to
        # MSE -- which is itself the NLL of a fixed-variance unimodal Gaussian.
        if hasattr(policy, "compute_loss"):
            return policy.compute_loss(s, a)
        return F.mse_loss(policy(s), a)

    history = {"epoch": [], "loss": [], "val_loss": []}
    for ep in range(1, epochs + 1):
        policy.train(); losses = []
        for s, a in tr:
            s, a = s.to(device), a.to(device)
            loss = loss_fn(s, a)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        history["epoch"].append(ep)
        history["loss"].append(float(np.mean(losses)))
        if val is not None:
            policy.eval()
            with torch.no_grad():
                history["val_loss"].append(float(loss_fn(*val).item()))
        if plot and (ep % plot_every == 0 or ep == epochs):
            _live_plot(history)
    return history


# --------------------------------------------------------------------------- #
# Rollouts in robosuite (closed-loop) + rendering
# --------------------------------------------------------------------------- #
def make_env(data, render=True):
    """Build the robosuite env that matches the dataset (same obs keys + controller)."""
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.obs_utils as ObsUtils
    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": {"low_dim": list(data.obs_keys), "rgb": []}})
    env_meta = FileUtils.get_env_metadata_from_dataset(data.dataset_path)
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta, render=False, render_offscreen=render, use_image_obs=False,
    )
    return env


def _seed_env(env, seed):
    """Reseed robosuite's reset RNG in place. robosuite samples object placements
    from the env's own Generator (env.rng), which the placement sampler holds by
    reference -- NOT the global np.random -- so we reset that generator's state."""
    base = getattr(env, "env", env)              # robomimic EnvRobosuite -> robosuite env
    rng = getattr(base, "rng", None)
    if rng is not None and hasattr(rng, "bit_generator"):
        rng.bit_generator.state = np.random.default_rng(seed).bit_generator.state


@torch.no_grad()
def rollout(policy, env, data, *, horizon=400, device=None, render=True,
            camera="agentview", img_hw=(256, 256), seed=None):
    """Run one closed-loop episode of `policy` in `env`. Returns frames/success/etc."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device).eval()
    if seed is not None:
        np.random.seed(seed); torch.manual_seed(seed)
        _seed_env(env, seed)               # robosuite resets from env.rng, not global np.random
    obs = env.reset()
    frames, visited, total_r, success = [], [], 0.0, False
    t = 0
    for t in range(horizon):
        s = build_state(obs, data.obs_keys)
        visited.append(s)
        sn = torch.from_numpy(data.normalize(s).astype(np.float32)).unsqueeze(0).to(device)
        a = policy.act(sn) if hasattr(policy, "act") else policy(sn)
        a = a.squeeze(0).detach().cpu().numpy()
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


def evaluate(policy, env, data, *, n_episodes=20, horizon=400, device=None,
             render_first=True, render_all=False, img_hw=(256, 256), camera="agentview", seed=0):
    """Run N rollouts, report success rate.

    render_first: render only episode 0 (cheap; for a single video via show_video).
    render_all:   render *every* episode (heavier; needed for show_video_grid).
    seed:         base seed -> episode k uses seed+k, so the whole eval is reproducible.
                  Pass seed=None for fresh randomness each call.
    """
    episodes = []
    for i in range(n_episodes):
        do_render = render_all or (render_first and i == 0)
        ep_seed = None if seed is None else seed + i
        out = rollout(policy, env, data, horizon=horizon, device=device,
                      render=do_render, img_hw=img_hw, camera=camera, seed=ep_seed)
        episodes.append(out)
    sr = float(np.mean([e["success"] for e in episodes]))
    return dict(success_rate=sr, episodes=episodes,
                first_frames=episodes[0]["frames"] if episodes else [])


# --------------------------------------------------------------------------- #
# Notebook video display
# --------------------------------------------------------------------------- #
def show_video(frames, fps=20, width=320, save=None, max_frames=None):
    """RGB frames -> mp4.

    save:       if given, WRITE the mp4 to that path and return a small link instead of
                embedding a multi-MB base64 data-URI in the page (the latter can OOM the
                browser tab). Open the file from the Jupyter file browser.
    max_frames: subsample to at most this many frames -> keeps the payload small.
    """
    from IPython.display import HTML
    import imageio
    if not frames:
        return HTML("<i>no frames (render=False?)</i>")
    arr = [np.asarray(f, dtype=np.uint8) for f in frames]
    if max_frames and len(arr) > max_frames:
        arr = arr[:: int(np.ceil(len(arr) / max_frames))]
    if save:
        os.makedirs(os.path.dirname(os.path.abspath(save)) or ".", exist_ok=True)
        imageio.mimwrite(save, arr, fps=fps, codec="libx264", macro_block_size=1)
        return HTML(f'<i>saved → <b>{save}</b></i> &nbsp;(open it from the Jupyter file browser)')
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        path = tmp.name
    imageio.mimwrite(path, arr, fps=fps, codec="libx264", macro_block_size=1)
    with open(path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode()
    os.remove(path)
    return HTML(
        f'<video width="{width}" controls autoplay loop>'
        f'<source src="data:video/mp4;base64,{b64}" type="video/mp4"></video>'
    )


def _encode_mp4_html(frames, fps, width):
    from IPython.display import HTML
    import imageio
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        path = tmp.name
    # macro_block_size=1 -> don't silently pad dims to a multiple of 16; we keep them even ourselves
    imageio.mimwrite(path, frames, fps=fps, codec="libx264", macro_block_size=1)
    with open(path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode()
    os.remove(path)
    return HTML(
        f'<video width="{width}" controls autoplay loop>'
        f'<source src="data:video/mp4;base64,{b64}" type="video/mp4"></video>'
    )


def _build_grid_frames(episodes, ncols=5, fill="freeze", border=6, label=True):
    """Tile rollouts into a synced list of grid frames (HxWx3 uint8). Shared by the
    mp4 and gif grid helpers. Returns [] if no episode has frames."""
    eps = [e for e in episodes if e.get("frames")]
    if not eps:
        return []
    try:
        import cv2
    except Exception:
        cv2 = None

    T = max(len(e["frames"]) for e in eps)
    n = len(eps)
    nrows = (n + ncols - 1) // ncols
    h, w = np.asarray(eps[0]["frames"][0]).shape[:2]
    ch, cw = h + 2 * border, w + 2 * border
    blank = np.zeros((ch, cw, 3), np.uint8)

    def pane(ep, idx, t):
        fr = ep["frames"]
        f = fr[t] if t < len(fr) else (fr[-1] if fill == "freeze" else np.zeros_like(fr[0]))
        f = np.asarray(f, dtype=np.uint8).copy()
        color = (0, 200, 0) if ep["success"] else (220, 0, 0)   # RGB (frames are RGB)
        if cv2 is not None:
            f = cv2.copyMakeBorder(f, border, border, border, border, cv2.BORDER_CONSTANT, value=color)
            if label:
                tag = f"{idx} {'OK' if ep['success'] else 'X'}"
                cv2.putText(f, tag, (border + 3, border + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        else:
            bb = np.empty((ch, cw, 3), np.uint8); bb[:] = color
            bb[border:border + h, border:border + w] = f
            f = bb
        return f

    grid_frames = []
    for t in range(T):
        cells = [pane(eps[i], i, t) if i < n else blank for i in range(nrows * ncols)]
        rows = [np.hstack(cells[r * ncols:(r + 1) * ncols]) for r in range(nrows)]
        grid = np.vstack(rows)
        H_, W_ = grid.shape[:2]
        if H_ % 2 or W_ % 2:                       # libx264 needs even dimensions
            grid = grid[:H_ - (H_ % 2), :W_ - (W_ % 2)]
        grid_frames.append(grid)
    return grid_frames


def show_video_grid(episodes, ncols=5, fps=20, width=900, fill="freeze", border=6, label=True,
                    save=None, max_frames=None):
    """Tile many rollouts into ONE synced grid video (inline base64 mp4, or `save` to disk).

    episodes: list of dicts with "frames" (HxWx3 uint8) and "success" (bool) --
              i.e. `evaluate(..., render_all=True)["episodes"]`. Green/red border + OK/X
              label per pane shows which rollouts drifted. NOTE: the base64 <video> this
              produces does NOT render on GitHub -- use show_gif_grid for that.
    """
    from IPython.display import HTML
    grid_frames = _build_grid_frames(episodes, ncols, fill, border, label)
    if not grid_frames:
        return HTML("<i>no frames — call evaluate(..., render_all=True) first</i>")
    if max_frames and len(grid_frames) > max_frames:
        grid_frames = grid_frames[:: int(np.ceil(len(grid_frames) / max_frames))]
    if save:
        import imageio
        os.makedirs(os.path.dirname(os.path.abspath(save)) or ".", exist_ok=True)
        imageio.mimwrite(save, grid_frames, fps=fps, codec="libx264", macro_block_size=1)
        return HTML(f'<i>saved grid → <b>{save}</b></i> &nbsp;(open it from the Jupyter file browser)')
    return _encode_mp4_html(grid_frames, fps=fps, width=width)


# --------------------------------------------------------------------------- #
# Animated GIF display -- renders inline AND on GitHub (GitHub does NOT play the
# base64 <video> outputs above). Keep frames/scale small so the notebook stays light.
# --------------------------------------------------------------------------- #
def _gif_bytes(frames, fps=12, max_frames=80, scale=1.0):
    import imageio, io
    arr = [np.asarray(f, dtype=np.uint8) for f in frames]
    if max_frames and len(arr) > max_frames:
        arr = arr[:: int(np.ceil(len(arr) / max_frames))]
    if scale != 1.0:
        try:
            import cv2
            arr = [cv2.resize(f, (max(2, int(f.shape[1] * scale)), max(2, int(f.shape[0] * scale))),
                              interpolation=cv2.INTER_AREA) for f in arr]
        except Exception:
            pass
    buf = io.BytesIO()
    imageio.mimwrite(buf, arr, format="gif", fps=fps, loop=0)
    return buf.getvalue()


def show_gif(frames, fps=12, max_frames=80, scale=1.0, save=None):
    """RGB frames -> an inline animated GIF (image/gif output). Unlike show_video this
    renders on GitHub and is lighter on the browser. `save` also writes the .gif."""
    from IPython.display import Image, HTML
    if not frames:
        return HTML("<i>no frames (render=False?)</i>")
    gif = _gif_bytes(frames, fps=fps, max_frames=max_frames, scale=scale)
    if save:
        os.makedirs(os.path.dirname(os.path.abspath(save)) or ".", exist_ok=True)
        with open(save, "wb") as fh:
            fh.write(gif)
    return Image(data=gif, format="gif")


def show_gif_grid(episodes, ncols=5, fps=10, max_frames=60, scale=0.6, fill="freeze",
                  border=6, label=True, save=None):
    """Tile rollouts into one synced animated GIF (renders on GitHub). scale<1 shrinks it."""
    from IPython.display import Image, HTML
    grid_frames = _build_grid_frames(episodes, ncols, fill, border, label)
    if not grid_frames:
        return HTML("<i>no frames — render some episodes first</i>")
    gif = _gif_bytes(grid_frames, fps=fps, max_frames=max_frames, scale=scale)
    if save:
        os.makedirs(os.path.dirname(os.path.abspath(save)) or ".", exist_ok=True)
        with open(save, "wb") as fh:
            fh.write(gif)
    return Image(data=gif, format="gif")


# --------------------------------------------------------------------------- #
# Save GIF to a FILE (no inline output). GitHub does NOT render image/gif notebook
# outputs, so commit the file and reference it from a markdown cell: ![](media/x.gif)
# -- markdown relative images DO render on GitHub.
# --------------------------------------------------------------------------- #
def save_gif(frames, path, fps=8, max_frames=80, scale=1.0):
    """Write frames to an animated GIF file (commit it + show via `![](path)` in a
    markdown cell). Prints the path; returns it. No inline cell output."""
    if not frames:
        print("[no frames to save (render=False?)]"); return None
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(_gif_bytes(frames, fps=fps, max_frames=max_frames, scale=scale))
    print(f"[saved {path}]"); return path


def save_gif_grid(episodes, path, ncols=5, fps=8, max_frames=60, scale=0.6,
                  fill="freeze", border=6, label=True):
    """Write a synced grid of rollouts to a GIF file (commit + reference from markdown)."""
    grid_frames = _build_grid_frames(episodes, ncols, fill, border, label)
    if not grid_frames:
        print("[no frames to save -- render some episodes first]"); return None
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(_gif_bytes(grid_frames, fps=fps, max_frames=max_frames, scale=scale))
    print(f"[saved {path}]"); return path

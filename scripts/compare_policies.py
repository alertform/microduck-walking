"""Benchmark a walking policy against a fixed velocity command.

Runs N independent environments (different random initial states) so the
reported numbers are mean +/- std across starts rather than a single rollout.
Velocity samples are only collected once an env has been alive for `warmup`
steps since its last reset, so post-fall recovery does not pollute the stats;
falls are reported separately as their own metric.

The twist command ranges are collapsed to single points, so every policy sees
an identical command.
"""

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

TASK = "Mjlab-Velocity-Flat-MicroDuck"
DEVICE = "cuda:0"


def build_env(num_envs, cmd, width, height, render):
    cfg = load_env_cfg(TASK, play=True)
    # Dense Jacobian's tile_cholesky kernel will not build on this GPU.
    cfg.sim.mujoco.jacobian = "sparse"
    cfg.scene.num_envs = num_envs
    cfg.viewer.width = width
    cfg.viewer.height = height

    twist = cfg.commands["twist"]
    twist.ranges.lin_vel_x = (cmd[0], cmd[0])
    twist.ranges.lin_vel_y = (cmd[1], cmd[1])
    twist.ranges.ang_vel_z = (cmd[2], cmd[2])
    for attr, val in (
        ("rel_standing_envs", 0.0),
        ("rel_turn_in_place_envs", 0.0),
        ("rel_forward_envs", 1.0),
        ("init_velocity_prob", 0.0),
    ):
        if hasattr(twist, attr):
            setattr(twist, attr, val)
    if hasattr(cfg, "curriculum"):
        cfg.curriculum = {}

    return ManagerBasedRlEnv(
        cfg=cfg, device=DEVICE, render_mode="rgb_array" if render else None
    )


def load_pt_policy(env, ckpt):
    agent_cfg = load_rl_cfg(TASK)
    runner = load_runner_cls(TASK)(env, asdict(agent_cfg), device=DEVICE)
    runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location=DEVICE)
    return runner.get_inference_policy(device=DEVICE)


def load_onnx_policy(path):
    import onnxruntime as ort

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    batched = sess.get_inputs()[0].shape[0] in (None, "batch", -1)

    def policy(obs):
        arr = as_actor_tensor(obs).detach().cpu().numpy().astype(np.float32)
        if batched:
            out = sess.run(None, {name: arr})[0]
        else:
            out = np.concatenate(
                [sess.run(None, {name: arr[i : i + 1]})[0] for i in range(len(arr))]
            )
        return torch.from_numpy(out).to(DEVICE)

    return policy


def as_actor_tensor(obs):
    while hasattr(obs, "keys"):
        keys = list(obs.keys())
        obs = obs["actor"] if "actor" in keys else obs[keys[0]]
    return obs


def body_vel(env):
    data = env.unwrapped.scene["robot"].data
    lin = next(
        getattr(data, n)
        for n in ("root_lin_vel_b", "root_com_lin_vel_b", "root_link_lin_vel_b")
        if hasattr(data, n)
    )
    ang = next(
        getattr(data, n)
        for n in ("root_ang_vel_b", "root_com_ang_vel_b", "root_link_ang_vel_b")
        if hasattr(data, n)
    )
    return lin, ang


def summarize(name, per_env, unit):
    m = per_env.mean()
    s = per_env.std()
    return f"{name:<22} {m:+.3f} +/- {s:.3f} {unit}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint")
    ap.add_argument("--onnx")
    ap.add_argument("--label", default=None)
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--vx", type=float, default=0.3)
    ap.add_argument("--vy", type=float, default=0.0)
    ap.add_argument("--wz", type=float, default=0.0)
    ap.add_argument("--video-out")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    if bool(args.checkpoint) == bool(args.onnx):
        ap.error("give exactly one of --checkpoint / --onnx")

    configure_torch_backends()
    record = args.video_out is not None
    n = args.envs
    raw = build_env(n, (args.vx, args.vy, args.wz), args.width, args.height, record)
    env = RslRlVecEnvWrapper(raw, clip_actions=None)

    if args.checkpoint:
        policy = load_pt_policy(env, args.checkpoint)
        label = args.label or f"ours({Path(args.checkpoint).name})"
    else:
        policy = load_onnx_policy(args.onnx)
        label = args.label or f"official({Path(args.onnx).name})"

    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]

    alive = torch.zeros(n, dtype=torch.long, device=DEVICE)
    sums = {k: torch.zeros(n, device=DEVICE) for k in ("vx", "vy", "wz", "vx2", "wz2")}
    counts = torch.zeros(n, device=DEVICE)
    falls = torch.zeros(n, device=DEVICE)
    frames = []

    for _ in range(args.steps):
        with torch.inference_mode():
            act = policy(obs)
        out = env.step(act)
        obs, dones = out[0], out[2]
        dones = dones.bool().flatten()

        lin, ang = body_vel(raw)
        alive += 1
        mask = (alive >= args.warmup).float()
        counts += mask
        sums["vx"] += lin[:, 0] * mask
        sums["vy"] += lin[:, 1] * mask
        sums["wz"] += ang[:, 2] * mask
        sums["vx2"] += (lin[:, 0] ** 2) * mask
        sums["wz2"] += (ang[:, 2] ** 2) * mask

        # `dones` includes time-limit truncation, which every env hits on the
        # 20 s episode boundary -- counting it would report a constant 3/min
        # for any policy. Only reset_terminated is a real failure.
        terminated = raw.unwrapped.termination_manager.terminated.bool().flatten()
        falls += terminated.float()
        alive[dones] = 0

        if record:
            frames.append(raw.render())

    c = counts.clamp(min=1)
    vx = (sums["vx"] / c).cpu().numpy()
    vy = (sums["vy"] / c).cpu().numpy()
    wz = (sums["wz"] / c).cpu().numpy()
    # Within-env temporal std, then averaged across envs.
    vx_std = torch.sqrt((sums["vx2"] / c - (sums["vx"] / c) ** 2).clamp(min=0))
    wz_std = torch.sqrt((sums["wz2"] / c - (sums["wz"] / c) ** 2).clamp(min=0))
    vx_std = vx_std.cpu().numpy()
    wz_std = wz_std.cpu().numpy()
    err = np.abs(vx - args.vx)
    dt = raw.unwrapped.step_dt
    secs = args.steps * dt
    fall_rate = (falls / secs * 60).cpu().numpy()

    print("=" * 68)
    print(f"policy        : {label}")
    print(f"command       : vx={args.vx} vy={args.vy} wz={args.wz}")
    print(f"envs          : {n}   steps: {args.steps} ({secs:.1f}s each)")
    print("-" * 68)
    print(summarize("vx mean", vx, "m/s"))
    print(summarize("vx error", err, "m/s"))
    print(summarize("vx std (temporal)", vx_std, "m/s"))
    print(summarize("vy mean", vy, "m/s"))
    print(summarize("wz mean", wz, "rad/s"))
    print(summarize("wz std (temporal)", wz_std, "rad/s"))
    print(summarize("falls per min", fall_rate, ""))
    print("=" * 68)

    if record:
        import imageio.v2 as imageio

        imageio.mimsave(args.video_out, frames, fps=int(round(1 / dt)))
        print(f"video -> {args.video_out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()

"""Render a trained policy tracking reference clips to MP4, with the reference as a translucent ghost.

Same settings as the benchmark (scripts/eval_benchmark.py): by default deterministic (no noise, no DR, clip from its
first frame, paper terminations); ``--train-noise --eval-seed k`` plays under the checkpoint's training perturbations
(same ``EvalCondition`` as the benchmark; one env, so the DR draw differs from the benchmark's env for that clip).
Per step it also records the two termination quantities (pelvis height error, tilt error = difference of the gravity z
in the pelvis frame) and reports which one ended a failed clip. ``--no-terminate`` keeps the two failure terminations
from ending the episode (only the time-out does), so the clip plays to its end; the first step at which a threshold is
crossed is still reported (``WOULD-FAIL@t``) and stored as ``fail_step``.

  MUJOCO_GL=egl python scripts/render_video.py --ckpt logs/run/ckpt_0044000.pt \
      --library $SM_BANK/bones_seed/library/bones_seed_compact.pt --pattern '^flip_180_' --max-clips 3 --out tmp/videos
  MUJOCO_GL=egl python scripts/render_video.py --ckpt logs/run/ckpt_latest.pt --npz-dir data/npz/lafan1 --clips dance1_subject1

Outputs per clip: <out>/<clip>.mp4 and <clip>.npz (per-step height / tilt errors and thresholds).
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import numpy as np
import torch
import tyro

from supermimic.envs.mdp import terminations as sm_term

# Record the termination quantities where the termination manager evaluates them (the env resets a terminated
# episode inside step(), so they cannot be read back afterwards).
_rec: dict[str, list[float]] = {"height_err": [], "tilt_err": []}
_bad_root_height, _bad_root_tilt = sm_term.bad_root_height, sm_term.bad_root_tilt
_no_terminate = [False]


def _mask(r: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(r, dtype=torch.bool) if _no_terminate[0] else r


def _height(env, command_name, threshold=0.25):
    cmd = env.command_manager.get_term(command_name)
    _rec["height_err"].append(float((cmd.root_pos_w[0, 2] - cmd.robot_root_pos_w[0, 2]).abs()))
    return _mask(_bad_root_height(env, command_name, threshold))


def _tilt(env, command_name, threshold=0.8):
    from mjlab.utils.lab_api.math import quat_apply_inverse

    cmd = env.command_manager.get_term(command_name)
    g = cmd.robot.data.gravity_vec_w
    d = quat_apply_inverse(cmd.root_quat_w, g)[0, 2] - quat_apply_inverse(cmd.robot_root_quat_w, g)[0, 2]
    _rec["tilt_err"].append(float(d.abs()))
    return _mask(_bad_root_tilt(env, command_name, threshold))


sm_term.bad_root_height, sm_term.bad_root_tilt = _height, _tilt


def _overlay(frame: np.ndarray, lines: list[str], red: set[int]) -> np.ndarray:
    from PIL import Image, ImageDraw

    im = Image.fromarray(frame)
    d = ImageDraw.Draw(im)
    for i, s in enumerate(lines):
        d.text((12, 10 + 18 * i), s, fill=(220, 30, 30) if i in red else (255, 255, 255))
    return np.asarray(im)


def main(
    ckpt: Path,
    out: Path,
    npz_dir: Path = Path("data/npz/lafan1"),
    library: Path | None = None,
    clips: list[str] | None = None,
    pattern: str | None = None,
    max_clips: int = 5,
    width: int = 960,
    height: int = 540,
    stride: int = 1,
    device: str = "cuda:0",
    train_noise: bool = False,
    eval_seed: int = 0,
    no_terminate: bool = False,
) -> None:
    import mediapy as media
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.torch import configure_torch_backends

    from supermimic.config import obs_mode_from_config
    from supermimic.envs.mdp.motion_library import discover_npz
    from supermimic.envs.tracking_env_cfg import make_env_cfg
    from supermimic.eval.benchmark import EvalCondition, benchmark_env_args, load_policy_from_checkpoint

    _no_terminate[0] = no_terminate
    policy, cfg = load_policy_from_checkpoint(ckpt, device)
    files = [str(library)] if library else [str(f) for f in discover_npz(npz_dir)]
    cond = EvalCondition.from_config(cfg, train_noise, train_noise, train_noise, train_noise, seed=eval_seed)
    a = benchmark_env_args(files, 1, cfg["env"]["future_frames"], obs_mode_from_config(cfg), library is not None, cond)
    a = dataclasses.replace(a, debug_vis=True)  # reference ghost
    env_cfg = make_env_cfg(a)
    env_cfg.viewer.width, env_cfg.viewer.height = width, height
    configure_torch_backends()
    env = ManagerBasedRlEnv(env_cfg, device=device, render_mode="rgb_array")
    cmd = env.command_manager.get_term("motion")
    lib = cmd.library
    names = [c.name for c in lib.clips]
    if clips:
        sel = [names.index(c) for c in clips]
    else:
        rx = re.compile(pattern or ".")
        sel = [i for i, n in enumerate(names) if rx.search(n)][:max_clips]
    out.mkdir(parents=True, exist_ok=True)
    fps = round(1.0 / env.step_dt / stride)
    h_thr = env_cfg.terminations["root_height"].params["threshold"]
    t_thr = env_cfg.terminations["root_tilt"].params["threshold"]
    print(f"[INFO] condition {cond.summary()}" + ("  [no-terminate: failure terminations disabled]" if no_terminate else ""))
    print(f"[INFO] {len(sel)} clip(s), obs_mode {a.obs_mode}, {fps} fps, thresholds height {h_thr} m / tilt {t_thr}")
    try:
        for ci in sel:
            name, length = names[ci], lib.clips[ci].length
            cmd.forced_frame[:] = lib.clip_start[ci]
            obs, _ = env.reset()
            for v in _rec.values():
                v.clear()
            frames, ret, failed, t, cross = [], 0.0, False, 0, -1
            with torch.no_grad():
                while True:
                    if t % stride == 0:
                        h = _rec["height_err"][-1] if _rec["height_err"] else 0.0
                        g = _rec["tilt_err"][-1] if _rec["tilt_err"] else 0.0
                        lines = [f"{name} [{cond.name}, seed {cond.seed}]", f"step {t}/{length}  return/step {ret / max(t, 1) / 0.1:.3f}", f"height err {h:.3f} (thr {h_thr})  tilt err {g:.3f} (thr {t_thr})"]
                        red = {0} if h > 0.8 * h_thr or g > 0.8 * t_thr else set()
                        if cross >= 0:
                            lines.append(f"termination threshold crossed at step {cross} (playing on)")
                            red.add(len(lines) - 1)
                        frames.append(_overlay(env.render(), lines, red))
                    obs, rew, term, tout, _ = env.step(policy(obs["policy"], obs["goal"]))
                    ret += float(rew[0])
                    t += 1
                    if cross < 0 and (_rec["height_err"][-1] > h_thr or _rec["tilt_err"][-1] > t_thr):
                        cross = t
                    if bool(term[0] | tout[0]):
                        failed = bool(term[0])
                        break
            he, te = np.array(_rec["height_err"]), np.array(_rec["tilt_err"])
            if no_terminate:
                failed = cross >= 0
            fail_step = (cross if no_terminate else t) if failed else -1
            cause = "-"
            if failed:
                cause = "+".join(k for k, v, thr in (("height", he[fail_step - 1], h_thr), ("tilt", te[fail_step - 1], t_thr)) if v > thr) or "?"
            tag = (f"WOULD-FAIL@{fail_step}({cause})" if no_terminate else f"FAIL@{t}({cause})") if failed else "ok"
            print(f"  {name:40s} {tag:22s} steps {t}/{length}  norm return {ret / (0.1 * max(length, t)):.3f}  max height err {he.max():.3f}  max tilt err {te.max():.3f}", flush=True)
            media.write_video(out / f"{name}.mp4", frames, fps=fps)
            np.savez(out / f"{name}.npz", height_err=he, tilt_err=te, height_thr=h_thr, tilt_thr=t_thr, failed=failed, fail_step=fail_step, length=length, no_terminate=no_terminate)
    finally:
        env.close()


if __name__ == "__main__":
    tyro.cli(main)

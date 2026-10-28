"""Pack a motion set of the `steering` repo (flat qpos.npy + index.npz) into a compact library (.pt).

The `steering` repo packs BONES-SEED G1 CSVs (the same `bones-studio/seed` g1.tar.gz) into flat arrays:
qpos.npy (N, 36) [root xyz m, root quat wxyz, 29 joints rad in G1_JOINT_NAMES order] and index.npz
(names `<session>/<clip stem>`, starts, lengths, body_names, fps). Its kinematic tree (bodies, offsets,
axes) is identical to mjlab's g1.xml and the pelvis body frame is the free-joint frame, so qpos[:7] is the
pelvis pose of the library rows. Only qpos is read: velocities are recomputed with this repo's
`motion_convert.resample_motion` (np.gradient for root linear / joint velocity, central SO(3) difference
for the WORLD-frame root angular velocity). steering's own qvel.npy uses forward mj_differentiatePos with
the angular part in the body frame, so it is not used. A set at another rate is resampled to 50 Hz by the
same function (at 50 Hz it is the identity on positions).

Clip names become the CSV stem (`<session>/` is dropped), as in stream_bones_seed.py; BONES-SEED's own
mirrored takes keep their `_M` suffix and count as separate clips (as in the full-dataset library).
Each clip also gets the Appendix E reasons of filter_bones_seed.py (name filter with the extra terms,
feasibility filter on CPU MuJoCo FK of mjlab's G1), written to the manifest; `--drop-filtered` removes
those clips from the library (default: keep the steering subset as is).

Outputs: the library (`ref` fp16 (N, 71), `names`, `starts`, `lengths`; build_library.py layout),
`<out>.manifest.csv` (name, source, frames, mirrored, keep, reason) and `<out>.names.txt` (library clips).

  uv run python scripts/data/steering_loco_to_library.py \
      --src <steering repo>/data/motions/loco --out data/library/bones_loco_compact.pt --workers 16
  uv run sm-train --data.library data/library/bones_loco_compact.pt ...

`--check-csv-dir <dir of original g1.tar.gz CSVs>` compares the rows with the CSV path (load_csv ->
resample_motion 120 -> 50 Hz -> forward_kinematics pelvis state, as stream_bones_seed.py) and prints the
max abs error per field. Measured on 6 loco clips: positions / quaternion / joints <= 2.4e-7, interior
velocities <= 1.1e-5 (fp32 storage of steering's qpos), all far below the fp16 storage error (~4e-3).
steering drops a clip's last frame when its duration lands exactly on the 50 Hz grid (one clip of six),
which changes that clip's last-frame one-sided velocities.
"""

from __future__ import annotations

import csv
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
import torch
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parent))
from filter_bones_seed import EXTRA_KEYWORDS, KEYWORDS, feasibility, name_filter

FIELDS = {"root_pos": slice(0, 3), "root_quat": slice(3, 7), "joint_pos": slice(7, 36), "root_lin_vel_w": slice(36, 39), "root_ang_vel_w": slice(39, 42), "joint_vel": slice(42, 71)}
OUT_FPS = 50.0

_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        from supermimic.data.motion_convert import load_g1_model

        _MODEL = load_g1_model()
    return _MODEL


def clip_ref(qpos: np.ndarray, fps: float):
    """(T, 36) steering qpos (wxyz) at `fps` -> ResampledMotion at 50 Hz with this repo's velocity conventions."""
    from supermimic.data import motion_convert as mc

    q = qpos.astype(np.float64)
    raw = np.concatenate([q[:, 0:3], mc.quat_wxyz_to_xyzw(q[:, 3:7]), q[:, 7:]], axis=1)  # CSV layout (xyzw)
    return mc.resample_motion(raw, fps, OUT_FPS)


def ref71(m) -> np.ndarray:
    return np.concatenate([m.root_pos, m.root_quat, m.joint_pos, m.root_lin_vel_w, m.root_ang_vel_w, m.joint_vel], axis=1)


def clip_reason(name: str, m) -> str | None:
    """filter_bones_seed.py's Appendix E reason (name filter incl. extra terms, then feasibility); None = kept."""
    import mujoco

    r = name_filter(name, KEYWORDS + [k.lower() for k in EXTRA_KEYWORDS])
    if r is not None:
        return r
    from supermimic.data.motion_convert import forward_kinematics

    model = _model()
    b = forward_kinematics(model, m)
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(1, model.nbody)]
    return feasibility(names, {"body_pos_w": b.pos_w, "body_quat_w": b.quat_w, "joint_vel": m.joint_vel, "joint_pos": m.joint_pos, "body_lin_vel_w": b.lin_vel_w})


def _task(args: tuple[str, float, list[tuple[int, str, int, int]], bool]):
    src, fps, clips, check = args
    qpos = np.load(Path(src) / "qpos.npy", mmap_mode="r")
    out = []
    for i, name, start, length in clips:
        m = clip_ref(np.array(qpos[start : start + length]), fps)
        out.append((i, ref71(m).astype(np.float16), clip_reason(name, m) if check else None))
    return out


def _init() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")


def check_csv(src: Path, csv_dir: Path) -> None:
    """Max abs error per field: our rows (from steering qpos) vs the repo's CSV path on the original CSVs."""
    from supermimic.data import motion_convert as mc

    with np.load(src / "index.npz") as z:
        stems = {str(n).split("/")[-1]: (str(n), int(s), int(l)) for n, s, l in zip(z["names"], z["starts"], z["lengths"])}
        fps = float(z["fps"])
    qpos = np.load(src / "qpos.npy", mmap_mode="r")
    model = _model()
    pel = 0  # pelvis is the first body after world
    for path in sorted(csv_dir.glob("*.csv")):
        if path.stem not in stems:
            print(f"{path.stem}: not in {src}")
            continue
        full, start, length = stems[path.stem]
        gold_m = mc.resample_motion(mc.load_csv(path), 120.0, OUT_FPS)
        b = mc.forward_kinematics(model, gold_m)
        gold = np.concatenate([b.pos_w[:, pel], b.quat_w[:, pel], gold_m.joint_pos, b.lin_vel_w[:, pel], b.ang_vel_w[:, pel], gold_m.joint_vel], axis=1)
        ours = ref71(clip_ref(np.array(qpos[start : start + length]), fps))
        n = min(len(gold), len(ours))
        g, o = gold[:n], ours[:n].copy()
        flip = np.sum(g[:, 3:7] * o[:, 3:7], axis=1) < 0
        o16 = o.astype(np.float16).astype(np.float64)
        errs = {k: np.abs(g[:, s] - o[:, s]).max() for k, s in FIELDS.items()}
        o[flip, 3:7] *= -1
        errs["root_quat (sign-aligned)"] = np.abs(g[:, 3:7] - o[:, 3:7]).max()
        interior = {k: np.abs(g[1:-2, s] - o[1:-2, s]).max() for k, s in FIELDS.items() if "vel" in k}
        e16 = np.abs(o16 - o).max()
        print(f"{full}: frames csv-path {len(gold)} ours {len(ours)}; quat sign flips {int(flip.sum())}/{n}; fp16 storage err {e16:.1e}")
        print("  max abs err  " + "  ".join(f"{k} {v:.2e}" for k, v in errs.items()))
        print("  interior vel " + "  ".join(f"{k} {v:.2e}" for k, v in interior.items()))


def main(
    src: Path,
    out: Path = Path("data/library/bones_loco_compact.pt"),
    workers: int = 8,
    drop_filtered: bool = False,
    check_feasibility: bool = True,
    limit: int | None = None,
    check_csv_dir: Path | None = None,
) -> None:
    """``--check-csv-dir`` only runs the comparison against original CSVs and writes nothing."""
    if check_csv_dir is not None:
        check_csv(src, check_csv_dir)
        return
    with np.load(src / "index.npz") as z:
        full_names = [str(n) for n in z["names"]][:limit]
        starts = z["starts"][:limit].astype(np.int64)
        lengths = z["lengths"][:limit].astype(np.int64)
        fps = float(z["fps"])
    names = [n.split("/")[-1] for n in full_names]
    if len(set(names)) != len(names):
        raise SystemExit("clip stems are not unique")
    print(f"{len(names)} clips, {int(lengths.sum())} frames at {fps:g} Hz from {src} ({sum(n.endswith('_M') for n in names)} mirrored)", flush=True)

    tasks, batch, size = [], [], 0
    for i, (n, s, l) in enumerate(zip(names, starts.tolist(), lengths.tolist())):
        batch.append((i, n, s, l))
        size += l
        if size >= 50_000:
            tasks.append((str(src), fps, batch, check_feasibility))
            batch, size = [], 0
    if batch:
        tasks.append((str(src), fps, batch, check_feasibility))
    refs: list[np.ndarray | None] = [None] * len(names)
    reasons: list[str | None] = [None] * len(names)
    with mp.Pool(workers, initializer=_init) as pool:
        for k, res in enumerate(pool.imap_unordered(_task, tasks)):
            for i, r, why in res:
                refs[i], reasons[i] = r, why
            if k % 20 == 0:
                print(f"  {k + 1}/{len(tasks)} tasks", flush=True)

    keep = [r is None or not drop_filtered for r in reasons]
    out.parent.mkdir(parents=True, exist_ok=True)
    stem = out.with_suffix("")
    with open(f"{stem}.manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "source", "frames", "mirrored", "keep", "reason"])
        for n, full, r, why, k in zip(names, full_names, refs, reasons, keep):
            w.writerow([n, full, len(r), int(n.endswith("_M")), int(k), why or ""])
    kept = [i for i in range(len(names)) if keep[i]]
    lens = [len(refs[i]) for i in kept]
    ref = torch.from_numpy(np.concatenate([refs[i] for i in kept], 0))
    if not torch.isfinite(ref).all():
        raise SystemExit("non-finite values in the library")
    torch.save({"ref": ref, "names": [names[i] for i in kept], "starts": np.cumsum([0] + lens[:-1]).tolist(), "lengths": lens}, out)
    Path(f"{stem}.names.txt").write_text("".join(names[i] + "\n" for i in kept))
    from collections import Counter

    print("filter reasons (all clips):", dict(Counter(r or "kept" for r in reasons).most_common()))
    print(f"saved {out}: {len(kept)} clips ({sum(names[i].endswith('_M') for i in kept)} mirrored), {ref.shape[0]} frames ({ref.shape[0] / OUT_FPS / 3600:.2f} h), {ref.numel() * 2 / 1e9:.2f} GB fp16")


if __name__ == "__main__":
    tyro.cli(main)

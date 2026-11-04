"""Benchmark execution condition: flag -> env args plumbing, output naming, JSON schema (CPU)."""

import json

from supermimic.eval.benchmark import ClipResult, EvalCondition, benchmark_env_args, save_results

CFG = {"env": {"obs_noise": True, "action_noise_std": 0.1, "domain_randomization": True, "reset_noise": True}}


def test_default_is_paper_protocol():
    c = EvalCondition()
    assert c.name == "det" and c == EvalCondition.from_config(CFG, False, False, False, False)
    a = benchmark_env_args(["lib.pt"], 8, 10, "beyondmimic_seq", True, c)
    assert (a.obs_noise, a.action_noise_std, a.domain_randomization, a.reset_noise, a.seed) == (False, 0.0, False, False, 0)
    assert a.sampling_mode == "start" and a.episode_length_s >= 1e4 and not a.record_transitions
    assert str(c.default_out("logs/r/ckpt_0100000.pt")) == "logs/r/benchmark_ckpt_0100000.json"


def test_train_noise_takes_checkpoint_values():
    c = EvalCondition.from_config(CFG, True, True, True, True, seed=3)
    assert c == EvalCondition(True, 0.1, True, True, 3) and c.name == "noisy"
    a = benchmark_env_args(["lib.pt"], 8, 10, "beyondmimic_seq", True, c)
    assert (a.obs_noise, a.action_noise_std, a.domain_randomization, a.reset_noise, a.seed) == (True, 0.1, True, True, 3)
    assert str(c.default_out("logs/r/ckpt_latest.pt")) == "logs/r/benchmark_noisy_ckpt_latest_seed3.json"
    assert str(EvalCondition.from_config(CFG, True, True, True, True).default_out("r/ckpt_latest.pt")) == "r/benchmark_noisy_ckpt_latest.json"
    # A component the run trained without stays off; partial conditions get their own name.
    off = {"env": {**CFG["env"], "obs_noise": False}}
    assert EvalCondition.from_config(off, True, True, True, True).name == "act+dr+rst"
    assert EvalCondition.from_config(CFG, False, True, False, False).name == "act"


def test_json_schema(tmp_path):
    r = [ClipResult("a", 10, 10, 0.9, False, -1, {"t": 0.01}, 0.9), ClipResult("b", 20, 5, 0.2, True, 5, {"t": 0.02}, 0.8)]
    c = EvalCondition(True, 0.1, True, True, 0)
    save_results(r, tmp_path / "x.json", c, num_envs=2)
    d = json.loads((tmp_path / "x.json").read_text())
    s = d["summary"]
    assert s["clips"] == 2 and s["failed"] == 1 and {"mean_norm_return", "length_weighted_norm_return", "terms"} <= s.keys()
    assert s["condition"] == {"name": "noisy", "obs_noise": True, "action_noise_std": 0.1, "domain_randomization": True, "reset_noise": True, "seed": 0, "num_envs": 2}
    assert d["clips"][1] == {"name": "b", "frames": 20, "steps": 5, "norm_return": 0.2, "failed": True, "fail_step": 5, "terms": {"t": 0.02}, "norm_return_per_step": 0.8}

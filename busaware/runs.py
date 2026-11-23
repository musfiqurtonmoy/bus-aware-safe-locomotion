"""Training runs under results/<run>/: names and configuration.

scripts/train.py writes results/<run>/config.json (mode, seed, penalty_w, w_energy, init, ...).
The nine from-scratch runs (baseline/penalty/filter_s0..2) were trained before config.json
existed, with train.py's defaults: penalty weight 0.5 (then hard-coded in leg_env) and
w_energy 0.0. run_config() returns those defaults for them, from the run name.
"""
import json
import pathlib
import re

from .params import Params

DEFAULT_PENALTY_W = 0.5
DEFAULT_W_ENERGY = 0.0
MODES = ("baseline", "penalty", "filter", "lagrangian")
# training/test mode -> HoppingLegEnv mode underneath (lagrangian: baseline env + cost wrapper,
# busaware/lagrangian.py; its evaluation return is the baseline reward, without lam * cost)
ENV_MODE = {"baseline": "baseline", "penalty": "penalty", "filter": "filter", "lagrangian": "baseline",
            "perjoint": "perjoint"}   # perjoint: evaluation-only joint-level limiter (CHANGELOG #38)


def env_mode(mode):
    return ENV_MODE[mode]


def run_name(mode, seed, penalty_w=DEFAULT_PENALTY_W, init=None):
    """e.g. baseline_s0, penalty_s0, filter_ws_s0, penalty_w0.05_ws_s1, penalty_w0.5_ws_s0,
    penalty_w2.0_ws_s0, lagrangian_ws_s0. The weight is written with repr(float(w)).
    Warm-started penalty runs always carry their weight; from-scratch penalty runs only when it
    differs from the default (the names of the original runs)."""
    tag = mode
    if mode == "penalty" and (init or penalty_w != DEFAULT_PENALTY_W):
        tag += f"_w{float(penalty_w)!r}"
    if init:
        tag += "_ws"
    return f"{tag}_s{seed}"


def run_config(run_dir):
    """Training configuration of a run directory (config.json, or the defaults of a
    from-scratch run name)."""
    run_dir = pathlib.Path(run_dir)
    base = dict(penalty_w=DEFAULT_PENALTY_W, w_energy=DEFAULT_W_ENERGY, init=None)
    f = run_dir / "config.json"
    if f.exists():
        return {**base, **json.loads(f.read_text())}
    m = re.fullmatch(r"(baseline|penalty|filter)_s(\d+)", run_dir.name)
    if not m:
        raise ValueError(f"{run_dir}: no config.json and not a from-scratch run name")
    return {**base, "mode": m.group(1), "seed": int(m.group(2))}


# ----------------------------------------------------------------------------------------------
# Evaluation settings (Step 9): plant/parameter variants, filter gamma, estimation error.
# The variant applies to the whole Params object, i.e. to the plant and to the filter's model
# alike (except current_lag and gear_eff, which only the plant uses); r0_scale scales the plant's
# R0 only, est_noise perturbs the filter's estimates only (HoppingLegEnv).
EVAL_VARIANTS = ("cold",) + Params.VARIANTS


def variant_params(name):
    """Params for an evaluation variant: None or "base" -> None (the env's default Params()),
    "cold" -> Params.cold(), otherwise Params.variant(name)."""
    if name in (None, "base"):
        return None
    if name == "cold":
        return Params.cold()
    return Params.variant(name)


ESTIMATORS = {("true", "fixed"): "", ("ekf", "fixed"): "_est-ekf", ("ekf", "confidence"): "_est-ekfconf"}


def eval_filename(variant=None, gamma=None, r0_scale=1.0, est_noise=0.0, estimator="true", margin="fixed",
                  controller="filter", margin_v=None, margin_i=None):
    """results/<run>/<file> for one evaluation setting. The default setting keeps the original
    name eval.csv, and r0_scale/est_noise keep their original tag _r<r0>_n<noise>, so earlier
    files are found under the same names. The filter's estimator (Step 8) adds _est-ekf (EKF,
    fixed margin) or _est-ekfconf (EKF, confidence-scaled margin). Examples: eval.csv,
    eval_var-cold.csv, eval_g0.6.csv, eval_r1.5_n0.0.csv, eval_est-ekfconf_r2.0_n0.0.csv."""
    tag = ""
    if variant not in (None, "base"):
        tag += f"_var-{variant}"
    if gamma is not None:
        tag += f"_g{float(gamma)!r}"
    if (estimator, margin) not in ESTIMATORS:
        raise ValueError(f"no estimator setting {(estimator, margin)}; choose from {list(ESTIMATORS)}")
    tag += ESTIMATORS[(estimator, margin)]
    # CHANGELOG #38: the joint-level limiter (test mode perjoint) and fixed margins other than the
    # env's defaults (None = default; the tag is written even when the value equals the default)
    if controller not in ("filter", "perjoint"):
        raise ValueError(f"no controller {controller!r}; choose filter or perjoint")
    if controller == "perjoint":
        tag += "_ctl-perjoint"
    if margin_v is not None:
        tag += f"_mv{float(margin_v)!r}"
    if margin_i is not None:
        tag += f"_mi{float(margin_i)!r}"
    if not (r0_scale == 1.0 and est_noise == 0.0):
        tag += f"_r{r0_scale}_n{est_noise}"
    return f"eval{tag}.csv"


def eval_done(path, test_modes, socs, episodes, counters=("a3_fail", "rem1_fail", "a4_fail", "a6_empty")):
    """True if the evaluation file at path is complete for this request and was written by the
    current evaluation code: it has the proof-assumption counter columns (added in CHANGELOG #25;
    files written before them are re-evaluated) and exactly `episodes` episodes for every
    requested (test_mode, soc). Used by evaluate.py --skip_done so an interrupted batch can be
    restarted."""
    import csv
    from collections import Counter
    path = pathlib.Path(path)
    if not path.exists():
        return False
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or any(f"{c}_n" not in rows[0] for c in counters):
        return False
    have = Counter((r["test_mode"], float(r["soc"])) for r in rows)
    return all(have.get((tm, float(s)), 0) == episodes for tm in test_modes for s in socs)

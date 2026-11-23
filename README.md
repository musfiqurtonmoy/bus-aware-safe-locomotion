# busaware: safe learned locomotion under battery and DC-bus limits

Code, results and manuscript sources for the paper *Safe Learned Locomotion Under Battery and
Direct-Current Bus Limits: A Barrier-Based Power Filter*. One leg of a Go1-sized quadruped hops on a
vertical rail in MuJoCo; its two drives share one battery (a Thevenin model with one RC pair, fitted to
public LG 18650HG2 cell data) and one DC bus with a capacitor and a brake chopper. A barrier-based power
filter turns a lowest bus voltage and a largest battery current into one power budget per 0.5 ms
substep and projects the drive currents onto it. Policies are trained with soft actor-critic
(Stable-Baselines3) with the filter in the loop and compared with baseline policies, the same filter
added at test time, reward penalties and SAC-Lagrangian, over three states of charge, plant variants,
battery-model errors and an extended Kalman filter for the battery parameters. Every number in the
paper is computed by a script from the files under `results/`, which are included.

## Requirements

- Python 3.14.6 (64-bit). Exact package versions: `requirements-lock.txt` (Stable-Baselines3 2.9.2,
  PyTorch 2.15.0, MuJoCo 4.0.1, Gymnasium 1.4.0, NumPy, SciPy, pandas, Matplotlib, pytest).
- Tested on Windows 10 (10.0.19045), Intel Core i5-10400, 8 GB RAM. CPU only: no GPU is used.
- The batch scripts `scripts/run_*.ps1` are Windows PowerShell scripts and expect the virtual
  environment at `.venv\` in the repository root. On other systems, run the same `train.py` and
  `evaluate.py` commands by hand (most batch scripts print their job list with `-List`).
- To rebuild the manuscript: a LaTeX distribution with `latexmk` (tested with MiKTeX).

## Install

```
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements-lock.txt
```

## Download the source data

The raw data are not redistributed. Download them into `data/raw/` (standard library only; every file
is checked against its SHA-256):

```
python scripts/download_data.py
```

This fetches the six LG 18650HG2 files used by the battery fit (Mendeley Data,
DOI [10.17632/b5mj79w5w9.2](https://doi.org/10.17632/b5mj79w5w9.2)) and the MuJoCo Menagerie Unitree Go1
model at commit `370c43965323c0944b433b202aac5058b5f907a9`. The actuator thesis (Katz 2018,
<https://hdl.handle.net/1721.1/118671>) is cited for drive parameters but is not redistributed or
downloaded; no script reads it. The data are needed only to redo the battery fit
(`scripts/fit_battery.py`); everything else runs from the included `results/`.

## Quick check

```
python -m pytest -q tests          # 87 passed, 3 skipped (the 3 skipped need --release and built PDFs)
python scripts/smoke_test.py       # random-action rollout in every mode
```

## Reproduce

All commands run from the repository root. Training and evaluation use fixed seeds (training seeds
0, 1, 2; evaluation reset seeds 1000 + episode). The included `results/` already holds every output, so
steps 1 to 3 can be skipped to regenerate only the tables, figures and manuscript (steps 4 and 5).
`scripts/train.py` refuses to overwrite a run that has `model.zip`; move `results/` aside first to
retrain.

**1. Battery and drive model**

```
python -I scripts/fit_battery.py       # results/battery_fit/ (needs data/raw/lg_hg2/)
python -I scripts/derive_pack.py       # results/pack_proposal/ (6S2P pack, one leg's share)
python scripts/validate_drive.py       # results/validation/ (averaged vs switching-level drive)
```

**2. Training** (one run at a time; output in `results/<run>/`)

From scratch, 300 000 steps, for each seed `S` in 0, 1, 2:

```
python scripts/train.py --mode baseline --steps 300000 --seed S
python scripts/train.py --mode penalty  --steps 300000 --seed S     # penalty weight w = 0.5
python scripts/train.py --mode filter   --steps 300000 --seed S
```

Warm-started from `baseline_s<S>`, 150 000 steps each:

```
powershell -ExecutionPolicy Bypass -File scripts\run_warmstart_all.ps1                     # filter, penalty w = 0.05 and 0.1, baseline; seeds 0-2
powershell -ExecutionPolicy Bypass -File scripts\run_warmstart_all.ps1 -Set penalty_extra  # penalty w = 0.2 and 0.5, seed 0
powershell -ExecutionPolicy Bypass -File scripts\run_round3.ps1                            # SAC-Lagrangian seeds 0-2; penalty w = 2.0 (seed 0), w = 0.5 (seeds 1, 2)
```

**3. Evaluation** (restartable: finished evaluations are skipped)

```
powershell -ExecutionPolicy Bypass -File scripts\run_final_batch.ps1            # plant variants, barrier rates, model errors, then the main evaluation of every run
powershell -ExecutionPolicy Bypass -File scripts\run_perjoint_margin_batch.ps1  # per-joint limit and margins
powershell -ExecutionPolicy Bypass -File scripts\run_ekf_batch.ps1              # battery EKF
powershell -ExecutionPolicy Bypass -File scripts\run_penalty_stress.ps1         # penalty w = 0.5 with the test-time filter, stress tests
python scripts/rollout_plot.py --run baseline_ws_s0                              # single-episode traces (figure)
python scripts/rollout_plot.py --run filter_ws_s0
python scripts/time_filter.py --run filter_s0                                    # filter compute time
python scripts/plot_training.py --run <run>                                      # training summary of one run
```

**4. Tables and figures**

```
python scripts/aggregate.py              # results/summary/ (main evaluation, mean and std over seeds)
python scripts/aggregate_ablations.py    # results/summary/ (ablations and stress tests)
python scripts/make_figures.py           # results/figures/ (vector PDFs)
python scripts/export_paper.py           # paper/generated/ (result macros, tables, figure copies)
```

**5. Manuscript** (anonymised manuscript and supplementary material)

```
cd paper
latexmk -pdf supplementary.tex
latexmk -pdf main.tex
```

`main.tex` cites the supplement's tables through `xr-hyper`, so the supplement is built first
(`paper/latexmkrc` does this automatically). Numbers enter the text only as `\res{key}` macros,
tables as `\gentable{name}` and figures as `\genfig{name}{width}`, all read from `paper/generated/`.

## Expected run times (Core i5-10400, 8 GB, CPU)

From the logs in `results/`:

| Step | Per run | Total |
|---|---|---|
| Training from scratch, 300 000 steps (9 runs) | 0.87 to 1.01 h | 8.4 h |
| Warm-started training, 150 000 steps (20 runs) | 0.44 to 0.55 h | 9.4 h |
| Main evaluation, 20 episodes x 3 states of charge, two test modes (23 files) | median 527 s | about 3.4 h |
| Main evaluation, one test mode (6 files) | median 308 s | about 0.5 h |
| Setting evaluations, 10 episodes x 3 states of charge, one test mode (147 files) | median 167 s | about 6.8 h |
| Setting evaluations, two test modes (30 files) | median 267 s | about 2.2 h |

Training times are the last wall-clock time stamp (`t`) in each `results/<run>/monitor.monitor.csv`.
Evaluation times are the medians of the gaps between the write times of consecutive evaluation files
within a batch, so they are estimates. Steps 4 and 5 take a few minutes.

## Repository map

| Path | Contents |
|---|---|
| `busaware/` | battery, bus, drive, barrier filter, EKF, MuJoCo environment, rollouts, run names, parameters |
| `assets/leg.xml` | MuJoCo model of the leg on a rail (values from the Menagerie Go1 model) |
| `scripts/` | data download, battery fit, drive validation, training, evaluation, aggregation, figures, LaTeX export |
| `tests/` | test suite (pytest) |
| `results/<run>/` | trained policy (`model.zip`), training log (`monitor.monitor.csv`, `config.json`, `training/`), per-episode evaluations (`eval*.csv`), rollouts |
| `results/summary/`, `results/figures/` | aggregated tables (CSV) and figures |
| `results/battery_fit/`, `results/pack_proposal/`, `results/validation/` | battery fit, pack derivation, drive validation |
| `paper/` | LaTeX sources of the anonymised manuscript and supplement; `paper/generated/` is written by `scripts/export_paper.py` |
| `params_provenance.md` | every model parameter with value, unit and source, or ASSUMED with the reason |
| `model_provenance.md`, `DATA_SOURCES.md` | provenance of the MuJoCo model and of the public data |
| `RESULTS.md` | recorded observations, each with its source file and row |

Comments in the code refer to the author's development log ("CHANGELOG #n"), step plan ("PROMPTS
Step n") and working rules ("CLAUDE.md rule n"). These files are not part of the release; the
references are kept so that the comments stay as they were when the results were produced.

## Data and third-party licences

The code in this repository is released under the MIT licence (`LICENSE`). Third-party material:

- **LG 18650HG2 Li-ion Battery Data** (P. Kollmeyer, M. Naguib, M. Skells), Mendeley Data, version 2,
  DOI 10.17632/b5mj79w5w9.2. Licence stated by Mendeley Data for this dataset: CC BY 4.0 (Creative
  Commons Attribution 4.0 International, <http://creativecommons.org/licenses/by/4.0>): "You can share,
  copy and modify this dataset so long as you give appropriate credit, provide a link to the CC BY
  license, and indicate if changes were made, but you may not do so in a way that suggests the rights
  holder has endorsed you or your use of the dataset." The dataset's readme adds: "If this data is
  utilized for any purpose, it should be appropriately referenced." The raw files are not included;
  `results/battery_fit/` holds parameters fitted to them (a change: open-circuit voltage curves and
  one-RC Thevenin parameters derived from the HPPC and C/20 tests).
- **MuJoCo Menagerie, Unitree Go1 model** (`unitree_go1/`, commit
  `370c43965323c0944b433b202aac5058b5f907a9`). Its licence file (`unitree_go1/LICENSE`) is the BSD
  3-Clause License, "Copyright (c) 2016-2022 HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics")".
  `assets/leg.xml` copies masses, inertias and dimensions from `go1.xml`; that licence is reproduced in
  `assets/LICENSE-unitree-go1`. Cite the Menagerie as in its `CITATION.cff` (Zakka, Tassa and the MuJoCo
  Menagerie Contributors, 2022).
- **Katz (2018)**, "A low cost modular actuator for dynamic robots", S.M. thesis, MIT,
  <https://hdl.handle.net/1721.1/118671>: cited for drive parameters, not redistributed.
- Python packages: see `requirements-lock.txt`; each is under its own licence.

## Citation

Citation will be added on publication.

## Contact

Added on publication.

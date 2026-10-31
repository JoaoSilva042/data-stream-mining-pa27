# PA27 — Business-Oriented Data Stream Mining

Practical Assignment I for **Data Stream Mining — NOVA IMS**.

## Assignment

| Item | Value |
|---|---|
| Assignment code | PA27 |
| Target | T3 — Cold Stress Event |
| City A | Porto |
| City B | Bragança |
| Business scenario | Winter Road-Service Operator |
| Business question | Which city experiences a greater frequency of cold-stress periods relevant to winter operations? |
| Historical period | 2016-01-01 to 2025-12-31 |
| Sampling frequency | Hourly |
| Warm-up | 2016-01-01 to 2017-12-31 |
| Evaluation | 2018-01-01 to 2025-12-31 |
| Random seed | 42 |
| Python | 3.12 |
| Environment manager | uv |

## Authors

João Teixeira Silva - 20260492
Beatriz Partidário - 20231631

## Project objective

The project builds two chronological hourly weather streams for Porto and Bragança and evaluates incremental classifiers for the official **T3 Cold Stress** target. It includes **20 city-specific experiments** and a **selected unified learning protocol**, in which a shared classifier learns from both city streams.

The shared target threshold is estimated once from the pooled 2016–2017 warm-up observations:

```text
cold_threshold = Q0.20(pooled warm-up temperature_2m) = 8.7 °C
cold_stress_t = 1 if temperature_2m_t <= 8.7 °C, otherwise 0
```

The threshold is frozen before the 2018–2025 evaluation period. Current target-defining temperature is never used as a current predictor.

## Repository structure

```text
.
├── config/
│   └── experiment.py              # Frozen experiment configuration
├── data/
│   ├── raw/                       # Open-Meteo raw JSON + provenance
│   └── processed/                 # Normalised hourly CSV streams
├── notebooks/
│   ├── 01_data_quality_eda.ipynb  # Data quality, EDA and feature decisions
│   └── 02_results_analysis.ipynb  # Model results and diagnostics
├── outputs/
│   ├── eda_figures/
│   ├── eda_tables/
│   ├── results_figures/
│   └── results_tables/
├── results/
│   ├── experiment_manifest.json   # Frozen experiment metadata
│   ├── summaries/                 # Final 20-run summary
│   ├── histories/                 # City-specific per-event histories (git-ignored)
│   ├── warmup_states/             # City-specific warm-up snapshots (git-ignored)
│   └── joint/                     # Selected unified-learning experiments
│       ├── joint_manifest.json
│       ├── summaries/
│       ├── histories/
│       └── warmup_states/
├── src/
│   ├── data.py                    # Data acquisition and raw validation
│   ├── preprocessing.py           # Loading, quality checks, target construction
│   ├── features.py                # Leakage-free online feature engineering
│   ├── models.py                  # River model factories
│   ├── evaluation.py              # Prequential evaluation and metrics
│   ├── run_experiments.py         # 20 city-specific runs
│   └── run_joint_experiments.py   # Selected unified-learning runs
├── tests/                         # Unit + integration tests
├── TECHNICAL_GUIDE.md             # Detailed architecture/function documentation
├── pyproject.toml                 # Project and direct dependencies
├── uv.lock                        # Fully locked dependency graph
└── requirements.txt               # Empty placeholder in this snapshot (use uv.lock)
```

## Requirements

The project was developed with **Python 3.12** and `uv`.

The following versions are resolved in the committed `uv.lock` (the `pyproject.toml` declarations are lower bounds):

```text
jupyter==1.1.1
matplotlib==3.11.2
numpy==2.5.3
pandas==3.0.6
requests==2.34.2
river==0.26.1
```

For exact transitive versions, `uv.lock` is the authoritative lock file.

## 1. Install uv

If `uv` is already installed, skip this section.

On macOS with Homebrew:

```bash
brew install uv
```

Check the installation:

```bash
uv --version
```

## 2. Create / synchronise the environment

From the repository root:

```bash
uv python install 3.12
uv sync --python 3.12 --frozen
```

`uv sync` creates the local `.venv` and installs the versions recorded in `uv.lock`.

### Optional: activate the virtual environment

You do **not** need to activate the environment when using `uv run`. If you want an activated shell anyway:

macOS / Linux:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

To leave the activated environment:

```bash
deactivate
```

## 3. Run the tests

Run the complete test suite from the project root:

```bash
uv run python -m unittest discover -s tests -v
```

The test suite checks, among other things:

- Open-Meteo payload validation;
- chronological construction of the streams;
- pooled warm-up threshold estimation;
- exact Representation A/B feature definitions;
- no use of the current observation in the current feature vector;
- 6-hour and 24-hour sliding-window calculations;
- bounded online feature memory;
- `PREDICT -> EVALUATE -> LEARN` ordering;
- continuity from warm-up into evaluation;
- confusion-matrix metrics;
- real Porto/Bragança period sizes and threshold;
- fresh model creation and complete 20-experiment orchestration;
- synchronized joint-stream timestamps, shared-model prediction/update order, and separate temporal feature states.

## 4. Data acquisition

The repository already contains the raw and processed data snapshots used for the official experiment. Therefore, **re-downloading data is not required to reproduce the submitted results**.

To retrieve the data again from Open-Meteo:

```bash
uv run python -m src.data
```

This command:

1. downloads Porto hourly weather;
2. saves the untouched API response and provenance;
3. converts it to the processed CSV stream;
4. waits before the second request to respect API limits;
5. repeats the procedure for Bragança.

Re-downloading data at a later date may produce a different upstream snapshot if the provider revises its historical dataset. For exact auditability, use the committed data snapshots.

## 5. Run the EDA notebook

Interactive mode:

```bash
uv run jupyter lab
```

Then open:

```text
notebooks/01_data_quality_eda.ipynb
```

Or execute it non-interactively from the repository root:

```bash
uv run jupyter nbconvert \
  --to notebook \
  --execute \
  --inplace \
  notebooks/01_data_quality_eda.ipynb
```

The notebook generates EDA tables and figures under `outputs/eda_tables/` and `outputs/eda_figures/`.

## 6. Run the official 20 experiments

From the repository root:

```bash
uv run python -m src.run_experiments
```

The runner executes:

```text
2 cities × 2 representations × 5 models = 20 experiments
```

For each experiment it:

1. creates a fresh model;
2. creates a fresh streaming feature builder;
3. performs the 2016–2017 warm-up;
4. continues the same feature state into 2018;
5. evaluates each event before learning its label;
6. updates the model and feature state online;
7. stores final predictive/computational metrics;
8. optionally stores the complete event-level history and warm-up state.

Main generated files:

```text
results/experiment_manifest.json
results/summaries/experiment_summary.csv
results/histories/*.csv
results/warmup_states/*.json
```

`results/histories/` and `results/warmup_states/` are intentionally git-ignored because they are large generated artefacts and can be reproduced by rerunning the experiment. Both city-specific and joint event-level histories and warm-up snapshots are excluded from Git because they are reproducible generated artefacts. The experiment summaries and manifests remain versioned.


## 6a. Run the selected joint experiments

The unified protocol is **not** a repetition of all ten possible model/representation pairs. It first reads the city-specific summary generated by Section 6, selects the best **(model, representation)** combination independently for Porto and Bragança, and removes duplicate combinations.

Selection is deterministic and uses these criteria, in order:

1. highest cumulative **Macro-F1**;
2. highest positive-class **Recall** (tie-breaker);
3. lowest prediction-plus-update **milliseconds per event** (tie-breaker);
4. model name and representation (deterministic final tie-breakers).

Selection is performed across **both representations A and B** for each city. Accordingly, **one or two distinct joint configurations** are executed, not ten. The current saved results select **Online Bagging, Representation B for both cities**, so there is **one joint model run**, reported separately for Porto and Bragança (two summary rows).

For each selected configuration:

- one fresh River classifier is shared by both cities, rather than one classifier per city;
- the two feature configurations remain separate in `config/experiment.py` but use the **same eleven weather predictors in the same order**; `snowfall` remains in the source data, not in the predictive feature vectors;
- Porto and Bragança use separate `StreamFeatureBuilder` objects for lagged features, 6-hour temperature statistics, and 24-hour event counts;
- hourly UTC timestamps are aligned across streams;
- both predictions for time `t` are evaluated **before** either `t` label updates the shared classifier;
- after both predictions, the model learns sequentially from the two labelled events, then the city-specific feature states are updated;
- evaluation metrics and prediction/update timings are recorded **separately for each city**.

Both streams use the same frozen pooled warm-up threshold (8.7 °C). The joint warm-up supplies 17,520 model updates per city, or 35,040 updates to the shared classifier. Each city contributes 70,128 predictions during evaluation.

Run **after** the 20 city-specific experiments:

```bash
uv run python -m src.run_joint_experiments
```

Generated outputs:

```text
results/joint/joint_manifest.json
results/joint/summaries/joint_experiment_summary.csv
results/joint/histories/*.csv
results/joint/warmup_states/*.json
```

**Important interpretation limit:** the selected configurations are chosen using the same 2018–2025 evaluation results later used for the unified comparison. The joint-versus-individual analysis is **exploratory**, not an independent confirmatory test of superiority. The joint model also receives twice the total warm-up examples of either independent city model. Do not attribute differences solely to a change in algorithm.

### Results in the current saved snapshot

The following values are read from `results/summaries/experiment_summary.csv` and `results/joint/summaries/joint_experiment_summary.csv`. They refer to **Online Bagging / Representation B** and to the same 2018–2025 city-specific evaluation periods.

| City | Protocol | Accuracy | Macro-F1 | Precision (Cold Stress) | Recall (Cold Stress) | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|
| Porto | Individual | 0.98909 | 0.96307 | 0.94185 | 0.92248 | 324 | 441 |
| Porto | Joint | 0.98948 | 0.96447 | 0.94150 | 0.92793 | 328 | 410 |
| Bragança | Individual | 0.97527 | 0.96999 | 0.95880 | 0.95598 | 837 | 897 |
| Bragança | Joint | 0.97663 | 0.97157 | 0.96441 | 0.95480 | 718 | 921 |

These are descriptive results for the saved run; metrics may change if the data snapshot, implementation, or evaluation protocol changes.

## 7. Run the results notebook

The results notebook needs **both** the city-specific and joint outputs. Run `src.run_experiments` first and then `src.run_joint_experiments` before executing the notebook.

Interactive mode:

```bash
uv run jupyter lab
```

Open:

```text
notebooks/02_results_analysis.ipynb
```

Or execute it non-interactively:

```bash
uv run jupyter nbconvert \
  --to notebook \
  --execute \
  --inplace \
  notebooks/02_results_analysis.ipynb
```

The notebook exports report-ready material to:

```text
outputs/results_tables/
outputs/results_figures/
```

## 8. Reproduce the complete workflow from a fresh clone

```bash
git clone https://github.com/JoaoSilva042/data-stream-mining-pa27.git
cd data-stream-mining-pa27

uv python install 3.12
uv sync --python 3.12 --frozen

uv run python -m unittest discover -s tests -v

# Optional: re-run EDA using the committed data snapshot
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_quality_eda.ipynb

# Reproduce 20 independent city-specific experiments
uv run python -m src.run_experiments

# Select the city-specific winners and run the unified protocol
uv run python -m src.run_joint_experiments

# Recreate result tables/figures
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/02_results_analysis.ipynb
```

## Experimental design summary

### Feature Representation A

Uses the previous observation (`t-1`) of each of **eleven** selected weather variables. Porto and Bragança retain separate entries in `BASE_FEATURES`, but both entries list the exact same variables in the same order. `snowfall` is excluded from the predictive feature sets in both cities.

### Feature Representation B

Contains all Representation A features plus:

```text
temperature_delta_1h
6-hour temperature rolling mean
6-hour temperature rolling sample standard deviation
number of Cold Stress hours in the previous 24 hours
```

All four additional features are computed from observations strictly before the target event. Representation A has **11 predictors** and B has **15 predictors** for either city. No city identifier and no other city's current or past measurements are included as model inputs.

### Models

The five mandatory incremental classifiers are:

1. Gaussian Naive Bayes;
2. Online Logistic Regression;
3. Hoeffding Tree;
4. Online Bagging;
5. Adaptive Random Forest.

Model parameters are frozen in `config/experiment.py` and recorded in `results/experiment_manifest.json`.

### Streaming protocol

Warm-up:

```text
BUILD FEATURES -> OBSERVE TARGET -> LEARN -> UPDATE FEATURE STATE
```

Evaluation:

```text
BUILD FEATURES -> PREDICT -> OBSERVE TARGET -> EVALUATE -> LEARN -> UPDATE FEATURE STATE
```

The current label and current target-defining temperature can therefore never influence the prediction being evaluated for the same event.

**Unified evaluation** adds a timestamp-level barrier:

```text
BUILD PORTO X_t + BRAGANÇA X_t
-> PREDICT BOTH
-> EVALUATE BOTH
-> LEARN FROM BOTH (same classifier)
-> UPDATE TWO INDEPENDENT FEATURE STATES
```

The order of model updates within each timestamp is fixed (Porto, then Bragança); the two predictions precede both updates.

## Reproducibility notes

- chronology is preserved throughout;
- no random train/test split is used;
- the T3 threshold is estimated only from pooled warm-up data;
- the threshold is frozen before evaluation;
- all city-specific models for a given city/representation receive the same evaluation history;
- stochastic learners use random seed `42`;
- every city-specific run creates a fresh model and a fresh feature builder;
- every selected joint run creates a fresh **shared** model and two independent city-specific feature builders;
- the joint selection criteria and selected combinations are recorded in `results/joint/joint_manifest.json`;
- model selection for joint learning uses the evaluation-period summary, so the joint comparison is exploratory;
- official outputs are traceable to the submitted code and configuration;
- raw/processed data snapshots are committed for auditability.

## Further documentation

See [`TECHNICAL_GUIDE.md`](TECHNICAL_GUIDE.md) for a detailed explanation of the architecture, each Python module and function, feature-state mechanics, modelling decisions, experiment lifecycle, generated artefacts and design rationale.

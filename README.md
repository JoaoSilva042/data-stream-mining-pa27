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

> **Before submission:** add both student names (and the group number, if required) here. The assignment handbook explicitly requires the README to identify the students.

## Project objective

The project builds two chronological hourly weather streams for Porto and Bragança and evaluates incremental classifiers for the official **T3 Cold Stress** target.

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
│   ├── histories/                 # Generated per-event histories (git-ignored)
│   └── warmup_states/             # Generated warm-up snapshots (git-ignored)
├── src/
│   ├── data.py                    # Data acquisition and raw validation
│   ├── preprocessing.py           # Loading, quality checks, target construction
│   ├── features.py                # Leakage-free online feature engineering
│   ├── models.py                  # River model factories
│   ├── evaluation.py              # Prequential evaluation and metrics
│   └── run_experiments.py         # 20-run experiment orchestration
├── tests/                         # Unit + integration tests
├── TECHNICAL_GUIDE.md             # Detailed architecture/function documentation
├── pyproject.toml                 # Project and direct dependencies
├── uv.lock                        # Fully locked dependency graph
└── requirements.txt               # Exact direct dependency versions
```

## Requirements

The project was developed with **Python 3.12** and `uv`.

Direct package versions locked for the project are:

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
- fresh model creation and complete 20-experiment orchestration.

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

`results/histories/` and `results/warmup_states/` are intentionally git-ignored because they are large generated artefacts and can be reproduced by rerunning the experiment.

## 7. Run the results notebook

The results notebook expects the official experiment outputs to exist. Therefore run the experiments first if `results/histories/` is absent.

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

# Reproduce the official model experiment
uv run python -m src.run_experiments

# Recreate result tables/figures
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/02_results_analysis.ipynb
```

## Experimental design summary

### Feature Representation A

Uses the previous observation (`t-1`) of the frozen city-specific weather variables. At minimum this includes previous temperature, relative humidity, precipitation and wind speed.

### Feature Representation B

Contains all Representation A features plus:

```text
temperature_delta_1h
6-hour temperature rolling mean
6-hour temperature rolling sample standard deviation
number of Cold Stress hours in the previous 24 hours
```

All features are computed from observations strictly before the target event.

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

## Reproducibility notes

- chronology is preserved throughout;
- no random train/test split is used;
- the T3 threshold is estimated only from pooled warm-up data;
- the threshold is frozen before evaluation;
- all models for a city/representation receive the same evaluation history;
- stochastic learners use random seed `42`;
- every experiment uses a fresh model and feature state;
- official outputs are traceable to the submitted code and configuration;
- raw/processed data snapshots are committed for auditability.

## Further documentation

See [`TECHNICAL_GUIDE.md`](TECHNICAL_GUIDE.md) for a detailed explanation of the architecture, each Python module and function, feature-state mechanics, modelling decisions, experiment lifecycle, generated artefacts and design rationale.

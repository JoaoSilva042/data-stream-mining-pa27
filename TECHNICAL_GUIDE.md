# PA27 Technical Guide

This document describes the architecture and implementation of the PA27 Data Stream Mining project in detail. It is intended to make the repository auditable and to make every important implementation decision explainable without having to reverse-engineer the code.

---

## 1. System overview

The project implements a complete chronological streaming-classification pipeline:

```text
Open-Meteo historical data
        |
        v
raw JSON + provenance
        |
        v
processed hourly city streams
        |
        v
quality checks / EDA
        |
        v
pooled warm-up threshold (2016–2017)
        |
        v
Cold Stress target construction
        |
        +------------------------------+
        |                              |
        v                              v
Representation A                 Representation B
previous-state features          A + temporal context
        |                              |
        +--------------+---------------+
                       |
                       v
             incremental classifiers
                       |
                       v
              chronological warm-up
                       |
                       v
        PREDICT -> EVALUATE -> LEARN
                       |
                       v
        histories + summaries + figures
```

The design is deliberately modular. Data acquisition, preprocessing, feature engineering, model construction, evaluation and experiment orchestration are separated into different modules so that each responsibility can be tested independently.

---

## 2. Frozen experimental design

The central configuration lives in `config/experiment.py` and is imported by the rest of the project. This avoids silently using different values in different files.

### Assignment

- Code: `PA27`
- Cities: Porto and Bragança
- Scenario: Winter Road-Service Operator
- Target: T3 Cold Stress Event
- Historical period: 2016-01-01 through 2025-12-31
- Sampling frequency: hourly
- Warm-up: 2016-01-01 through 2017-12-31
- Evaluation: 2018-01-01 through 2025-12-31
- Random seed: `42`

### Target

The T3 target is defined using a pooled warm-up quantile:

```text
threshold = Q0.20(temperature_2m from Porto + Bragança warm-up)
```

For the committed data snapshot:

```text
threshold = 8.7 °C
```

Then:

```text
cold_stress_t = 1  if temperature_2m_t <= 8.7
cold_stress_t = 0  otherwise
```

The threshold is estimated from the warm-up only and is never recomputed during evaluation.

### Why a pooled threshold?

A single threshold is used for both cities so their Cold Stress event frequencies remain directly comparable. Pooling only the warm-up observations avoids using future evaluation information.

---

## 3. Repository architecture

### `config/experiment.py`

Contains the immutable experiment settings shared by every module.

It defines:

- assignment metadata;
- historical, warm-up and evaluation dates;
- target quantile and target source column;
- raw Open-Meteo variables;
- API endpoints and time zones;
- fixed Porto/Bragança coordinates;
- city-specific base feature sets;
- 6-hour and 24-hour temporal windows;
- 168-hour recent evaluation window;
- model names and frozen hyperparameters;
- random seed.

Centralising these values prevents duplicated configuration and makes the official run reproducible.

---

# 4. `src/data.py` — data acquisition

This module is responsible only for obtaining and normalising the source data. It does not create the target or model features.

## Module-level paths

`PROJECT_ROOT`, `RAW_DIR` and `PROCESSED_DIR` are derived from the location of the source file rather than the current shell directory. This allows commands to be run from the repository root without hard-coded user-specific paths.

## `fetch_json(url, params=None, headers=None, timeout=60, max_retries=3)`

Purpose: generic JSON HTTP helper used by the Open-Meteo requests.

Behaviour:

1. adds a project-specific User-Agent;
2. performs a `requests.get` call;
3. handles HTTP 429 rate limiting;
4. respects `Retry-After` when supplied;
5. otherwise waits before retrying;
6. raises normal HTTP errors for non-success responses;
7. returns both the parsed JSON and the response object.

Why return the response object? The provenance file stores the exact request URL used by the API call.

## `geocode_city(city_name)`

Purpose: resolve a Portuguese city using the Open-Meteo geocoding endpoint.

The function remains available as a helper, but the official experiment uses fixed coordinates stored in `config/experiment.py`. Fixed coordinates were preferred for reproducibility: future changes in geocoder ranking cannot move the data source location.

## `save_json(data, path)`

Purpose: write a Python object as human-readable UTF-8 JSON.

It creates missing parent directories and uses indentation so raw/provenance files can be inspected manually.

## `fetch_historical_weather(latitude, longitude)`

Purpose: request the official hourly historical stream from Open-Meteo.

The request is constructed from the frozen configuration:

- start/end dates;
- all requested hourly weather variables;
- GMT API timeline;
- Unix timestamps;
- ERA5 model.

Using UTC/GMT for the canonical stream avoids daylight-saving gaps or duplicated hours. A Lisbon-local timestamp is added separately for interpretation.

## `validate_weather_payload(data, city_name)`

Purpose: reject malformed or incomplete API responses before they become part of the project data.

Checks:

- an `hourly` object exists;
- a `time` field exists;
- every requested weather variable was returned;
- all hourly arrays have exactly the same length.

This is an early structural validation layer.

## `weather_json_to_dataframe(data, city_name)`

Purpose: transform the Open-Meteo JSON into the canonical hourly DataFrame.

Steps:

1. validates the payload;
2. converts Unix timestamps to timezone-aware UTC timestamps;
3. derives Europe/Lisbon local timestamps from UTC;
4. adds the city label;
5. removes the raw Unix `time` field;
6. enforces the expected column order;
7. confirms chronological order.

The UTC timestamp is the authoritative ordering key used by the stream pipeline.

## `basic_stream_quality_report(df)`

Purpose: create the first lightweight quality snapshot immediately after acquisition.

Reports:

- row count;
- duplicated rows;
- duplicated timestamps;
- missing values per weather variable;
- chronological ordering;
- first and final UTC timestamps.

This report is saved with the processed data before the deeper EDA checks.

## `build_provenance(...)`

Purpose: record where the data came from and how it was requested.

The provenance JSON contains:

- provider and dataset;
- requested and returned coordinates;
- elevation;
- API/local time zones;
- historical dates;
- Open-Meteo model;
- requested variables;
- units;
- retrieval timestamp;
- request URL;
- raw and processed file paths.

This supports auditability and makes the data snapshot traceable.

## `download_city(city_name)`

Purpose: execute the complete acquisition pipeline for one city.

Sequence:

```text
fixed coordinates
-> fetch weather
-> validate response
-> save raw JSON
-> convert to DataFrame
-> save processed CSV
-> save provenance
-> save initial quality report
```

## `main()`

Purpose: command-line entry point for data acquisition.

It downloads Porto, waits between requests to avoid rate-limit problems, downloads Bragança and prints a final acquisition summary.

Run with:

```bash
uv run python -m src.data
```

---

# 5. `src/preprocessing.py` — validation, periods and target

This module loads the committed processed snapshots and performs deterministic preprocessing. It deliberately avoids model-specific transformations.

## `load_city_data(city_name)`

Purpose: load `data/processed/<city>_hourly.csv` safely.

Key decisions:

- a comprehensive list of textual missing-value tokens is recognised;
- UTC timestamps are parsed strictly;
- local timestamps are rebuilt from UTC rather than trusting CSV string formatting;
- DST conversion is therefore handled by pandas/timezone rules.

No rows are silently deleted here.

## `expected_hourly_index()`

Purpose: construct the exact hourly UTC timeline that should exist between the configured historical start/end dates.

For 2016–2025 it contains 87,672 hourly observations.

This independent expected index is used to detect temporal gaps rather than assuming that row count alone proves completeness.

## `structural_quality_report(df, city_name)`

Purpose: verify the structural integrity of a complete city stream.

Checks include:

- required columns;
- observed versus expected row count;
- exact first/last timestamps;
- missing values;
- duplicate rows/timestamps;
- missing expected timestamps;
- unexpected timestamps;
- infinite numeric values;
- chronological order;
- numeric weather datatypes;
- city label values.

## `missing_values_report(df)`

Purpose: provide per-column missing counts and percentages for EDA/reporting.

## `invalid_values_report(df)`

Purpose: detect values that violate basic semantic/physical constraints without automatically deleting unusual weather.

Examples:

- relative humidity outside 0–100%;
- negative precipitation/rain/snowfall/wind speed/radiation/etc.;
- wind direction outside 0–360 degrees;
- non-positive pressure.

This separates clearly invalid values from valid extreme observations.

## `compare_city_timelines(city_a, city_b)`

Purpose: ensure Porto and Bragança contain exactly the same ordered UTC timestamps.

Why it matters: city comparisons and model comparisons should refer to compatible chronological periods.

## `unexpected_non_numeric_report(df)`

Purpose: find textual/non-numeric contamination in weather columns by coercing values to numeric only for the purpose of diagnosis.

## `select_period(df, start_date, end_date)`

Purpose: select an inclusive UTC interval where the end date includes the complete final day through 23:00.

This avoids a common bug where date-based slicing accidentally ends at midnight at the start of the final date.

## `get_warmup(df)`

Thin wrapper around `select_period` for the configured 2016–2017 warm-up.

## `get_evaluation_stream(df)`

Thin wrapper around `select_period` for the configured 2018–2025 evaluation stream.

## `compute_pooled_cold_threshold(city_datasets)`

Purpose: implement the official T3 pooled-warm-up threshold rule.

Steps:

1. extracts each city's warm-up only;
2. concatenates their `temperature_2m` values;
3. computes the configured 0.20 quantile;
4. returns a Python float.

Crucially, evaluation observations never participate in threshold estimation.

## `add_cold_stress_target(df, threshold)`

Purpose: create the binary T3 target without mutating the input DataFrame.

```text
1 if temperature_2m <= threshold
0 otherwise
```

The current `temperature_2m` is used to define the target, but the feature builder later ensures that this same current value is not used to predict that target.

---

# 6. `src/features.py` — online feature state

This is the most important leakage-control module. Features are not precomputed from the full dataset. They are built incrementally from a bounded state containing only observations that have already occurred.

## `InstanceSlidingWindow`

A fixed-capacity online window supporting exact incremental statistics.

It stores:

- `deque` of active values;
- running sum;
- running sum of squares.

When a new value arrives and the window is full, the oldest value is removed from all three representations before the new value is added.

This gives constant-size memory and avoids recalculating window statistics from the entire stream.

### `__init__(size)`

Validates a positive capacity and initialises the empty deque and sufficient statistics.

### `append(value)`

Adds one value, removes the expired oldest value if required, and updates `sum` and `sum_sq`.

Returns the expired value or `None`.

### `count`

Number of observations currently inside the active window.

### `mean`

Returns the current arithmetic mean from the running sum. Returns NaN for an empty window.

### `variance`

Computes the **sample variance** using denominator `n - 1`:

```text
(sum_sq - sum^2 / n) / (n - 1)
```

A `max(0, ...)` guard prevents very small negative values caused by floating-point round-off.

### `std`

Square root of the sample variance.

### `values`

Returns a copy of the active values for audit/debug snapshots.

## `get_base_features(city_name)`

Returns the frozen city-specific Representation A source variables from configuration.

Porto uses 11 base variables. Bragança uses the same conceptual set plus snowfall, which was retained because it is a non-constant physically relevant feature in Bragança.

## `get_feature_columns(city_name, representation)`

Purpose: describe exactly which model-input names the feature builder will emit.

Representation A converts every selected source variable to a `_lag1` predictor.

Representation B contains all A features plus:

- `temperature_delta_1h`;
- `temperature_mean_6h`;
- `temperature_std_6h`;
- `cold_stress_hours_24h`.

## `StreamFeatureBuilder`

Stateful online feature transformer. The key design rule is:

```text
transform_one() happens BEFORE update_one(current_row, current_target)
```

Therefore features for time `t` can only contain information accumulated through `t-1`.

### `__init__(city_name, representation)`

Initialises:

- frozen city-specific base features;
- previous full weather state;
- last two temperatures for the delta;
- 6-hour temperature window;
- 24-hour target-history window;
- observation counter.

### `ready`

Returns whether enough history exists to construct a complete input.

Both A and B deliberately use the same 24-hour initialisation requirement. This means their comparison begins after exactly the same amount of stream history rather than giving A an earlier start.

### `transform_one()`

Builds `X_t` **without changing state**.

Representation A:

```text
weather variables from t-1
```

Representation B additionally calculates:

```text
T(t-1) - T(t-2)
mean of previous 6 temperatures
sample std of previous 6 temperatures
number of Cold Stress labels in previous 24 hours
```

If initial history is incomplete, it returns `None`.

### `update_one(row, y)`

Called only after the current event has been processed.

It:

1. validates that the target is binary;
2. extracts the current selected weather values;
3. appends current temperature to temperature history/windows;
4. appends the now-observed target to the 24-hour event window;
5. stores the current weather row as the future `t-1` state;
6. increments the observation counter.

This sequencing is what prevents target leakage.

### `state_snapshot()`

Returns a read-only dictionary describing the feature state at a stream boundary, including active windows and the next feature vector if ready.

The warm-up state files use this for auditability and to prove that state continuity exists at the 2017/2018 boundary.

---

# 7. `src/models.py` — incremental learners

All model constructors create **fresh River model instances**. Reusing a fitted instance between A/B or cities would invalidate the controlled comparison, so the experiment runner works with factories rather than persistent global models.

## `make_gaussian_nb()`

Creates River `GaussianNB` using the frozen parameter dictionary (defaults in this project).

Role: compact incremental probabilistic baseline.

## `make_logistic_regression()`

Creates online Logistic Regression with SGD learning rate `0.05`.

When `standardize=True`, the classifier is wrapped in a River online pipeline:

```text
StandardScaler | LogisticRegression
```

The scaler itself learns incrementally, so no future observations are used to fit scaling parameters.

## `make_base_tree()`

Creates the frozen Hoeffding Tree:

- grace period 200;
- Hellinger split criterion;
- delta `1e-7`;
- tau `0.05`;
- Naive Bayes Adaptive (`nba`) leaves.

The same constructor is used for the standalone Hoeffding Tree and as the Online Bagging base learner, keeping the ensemble comparison interpretable.

## `make_model_factories()`

Returns the five mandatory factory callables:

- GaussianNB;
- Online Logistic Regression;
- Hoeffding Tree;
- Online Bagging;
- Adaptive Random Forest.

Online Bagging uses 10 Hoeffding base models and seed 42.

Adaptive Random Forest uses 10 trees plus the frozen randomisation/tree parameters from configuration.

Every call to a factory produces an independent model.

---

# 8. `src/evaluation.py` — online evaluation

This module implements the prequential metrics and the central chronological evaluation order.

## `OnlineConfusionMatrix`

An incremental confusion matrix for fixed class labels.

### `__init__(labels)`

Creates a label-to-index mapping and an integer square matrix initialised to zero.

### `update(y_true, y_pred)`

Adds one evaluated prediction to the appropriate cell.

### `remove(y_true, y_pred)`

Subtracts an old prediction. This makes a true sliding recent-performance window possible without recomputing metrics from scratch.

### `total`

Total number of predictions represented by the matrix.

### `correct`

Trace of the matrix, i.e. number of correct predictions.

### `accuracy()`

Returns `correct / total`, or NaN if there are no predictions.

### `precision(label)`

Class-specific precision:

```text
TP / all predictions of that class
```

Supports an explicit fallback for mathematically undefined cases.

### `recall(label)`

Class-specific recall:

```text
TP / all actual observations of that class
```

### `f1(label)`

Harmonic mean of class-specific Precision and Recall.

### `macro_f1()`

Unweighted mean of the F1 scores of both classes. Each class therefore contributes equally even when the target is imbalanced.

## `StreamingEvaluator`

Maintains two confusion matrices simultaneously:

- `cumulative`: all evaluated events since evaluation began;
- `recent`: only the latest `EVALUATION_WINDOW=168` events.

A deque stores the recent `(y_true, y_pred)` pairs so the oldest one can be removed exactly when the window advances.

### `__init__(labels, window_size, positive_class)`

Initialises the cumulative/recent state and counters. It validates that the positive class is known and the window size is positive.

### `observe(y_true, y_pred)`

Called **after prediction and before model learning**.

It:

1. counts the observation;
2. handles models unable to predict during a cold start;
3. updates cumulative metrics;
4. expires the oldest recent prediction if the window is full;
5. adds the new prediction to the recent window;
6. returns a metric snapshot.

### `snapshot()`

Returns current:

- cumulative Accuracy;
- recent-window Accuracy;
- cumulative Macro-F1;
- recent-window Macro-F1;
- cumulative/recent positive-class Precision;
- cumulative/recent positive-class Recall.

## `_component_objects(model)`

Internal audit helper that extracts the top-level objects inside River pipelines where available.

It exists so warm-up snapshots can report useful structural information without serialising entire trained models.

## `summarize_model_state(model)`

Builds a compact model description after warm-up.

Depending on model type, it attempts to capture:

- model/component classes;
- number of learned weights;
- intercept;
- scaler feature counts;
- class counts;
- ensemble size;
- tree nodes/leaves/height/active leaves where exposed by River.

The function intentionally uses defensive `getattr`/type checks because different River estimators expose different internal attributes.

## `build_warmup_state(...)`

Creates an audit snapshot immediately after the complete warm-up.

Includes:

- warm-up rows;
- number of actual model updates;
- positive/negative warm-up event counts;
- model structural summary;
- full feature-builder state;
- warm-up start/end timestamps.

For the official streams there are 17,544 warm-up observations and 17,520 model updates because the common 24-hour feature history is built before the first usable model input.

## `warm_up_online(...)`

Runs the 2016–2017 sequential initialisation.

For every observation:

```text
BUILD X_t FROM PAST
-> OBSERVE y_t
-> LEARN if X_t is ready
-> UPDATE FEATURE STATE with observation t
```

No evaluation metrics are collected during warm-up.

The returned feature builder is the same stateful object later passed into evaluation; it is not reset at the boundary.

## `evaluate_model(...)`

Implements the official prequential loop:

```text
BUILD X_t FROM PAST
-> PREDICT
-> reveal y_t
-> EVALUATE prediction
-> LEARN (X_t, y_t)
-> UPDATE FEATURE STATE with observation t
```

Important details:

- it raises if the feature builder is unexpectedly unready during evaluation;
- prediction and update times are measured separately with `perf_counter`;
- only selected temporal diagnostic features are written to history to control file size;
- each history row stores timestamp, true/predicted label, tracked features and cumulative/recent metrics.

## `build_evaluation_summary(...)`

Converts the final evaluator state into the required report metrics:

- observations processed;
- predictions evaluated;
- Accuracy;
- Macro-F1;
- positive-class Precision;
- positive-class Recall;
- TN, FP, FN, TP;
- cold starts;
- prediction milliseconds per event;
- update milliseconds per event.

For the official experiment cold starts are zero, so all 70,128 evaluation observations are scored.

---

# 9. `src/run_experiments.py` — experiment orchestration

This module connects the otherwise independent pieces into the official controlled experiment.

## Output directories

- `results/summaries/`
- `results/histories/`
- `results/warmup_states/`

## `prepare_experiment_data()`

Executes the shared preprocessing once:

1. loads Porto and Bragança;
2. verifies identical timelines;
3. computes the pooled warm-up threshold;
4. creates the T3 target for both cities;
5. creates full, warm-up and evaluation DataFrames.

Returning the same prepared stream objects for all models prevents accidental differences in data construction.

## `run_single_experiment(...)`

Runs exactly one:

```text
city × representation × model
```

Steps:

1. validate the requested model name;
2. create a **fresh model**;
3. create a **fresh StreamFeatureBuilder**;
4. warm up sequentially on 2016–2017;
5. retain the feature state at the boundary;
6. evaluate online on 2018–2025;
7. verify that every evaluation observation entered the history;
8. build final summary metrics;
9. attach experiment metadata;
10. attach city/representation/model identifiers to the history.

Metadata includes frozen model parameters, seed, threshold, window size, period sizes and warm-up state counts.

## `save_warmup_state(...)`

Writes one JSON warm-up audit file per experiment.

These files are generated artefacts and are git-ignored because they can be reproduced.

## `save_experiment_history(...)`

Writes the complete event-level prequential history for one run as CSV.

Histories are intentionally git-ignored because 20 hourly histories are large; rerunning `src.run_experiments` recreates them.

## `save_experiment_manifest(...)`

Writes `results/experiment_manifest.json`, the compact record of the official experiment design.

It records:

- assignment/cities;
- full/warm-up/evaluation periods and row counts;
- frozen threshold;
- target;
- seed;
- representations;
- models;
- 168-hour recent metric window;
- all model parameters;
- exact warm-up and evaluation protocol order;
- statement that parameters were selected before official evaluation rather than tuned on 2018–2025.

## `run_all_experiments(save_histories=True)`

Runs the Cartesian product:

```text
2 cities × 2 representations × 5 models = 20 runs
```

Before starting, it confirms that a factory exists for every configured model.

Each run receives a new model and feature builder. Results are accumulated into one master DataFrame and saved to:

```text
results/summaries/experiment_summary.csv
```

## `main()`

CLI entry point. Runs all experiments, saves artefacts and prints the final master summary.

Run with:

```bash
uv run python -m src.run_experiments
```

---

# 10. Notebook architecture

## `notebooks/01_data_quality_eda.ipynb`

Purpose: understand and validate the streams before interpreting model performance.

Main sections:

1. setup and data loading;
2. data quality / invalid values;
3. descriptive statistics;
4. warm-up-only weather correlations;
5. lagged-feature correlation with the target;
6. feature selection;
7. warm-up temporal-dependence diagnostics;
8. temporal evolution;
9. target frequency/class distribution;
10. rolling behaviour;
11. unusual/extreme observations;
12. business-context descriptive analysis.

### Important design decision: warm-up-only modelling decisions

Correlations and temporal diagnostics that can influence feature/model design use the 2016–2017 warm-up only. This avoids choosing features/windows after inspecting the evaluation period.

Descriptive evaluation-period figures can still be used to characterise the realised operational stream, provided they are not fed back into model design.

## `notebooks/02_results_analysis.ipynb`

Purpose: audit and analyse the outputs of the frozen 20-run experiment.

Main sections:

1. setup;
2. output loading;
3. mechanical audit;
4. final predictive performance;
5. Representation A versus B deltas;
6. confusion-matrix / FP-FN evidence;
7. recent-window behaviour over time;
8. computational evidence / cost-performance;
9. persistence baseline;
10. transition diagnostics.

The notebook also exports compact result tables and figures under `outputs/results_*`.

---

# 11. Major modelling and feature decisions

## 11.1 Why hourly observations?

The experiment uses an hourly stream because the source is hourly and the required temporal representation is naturally defined around recent hours. This preserves a dense chronological stream and supports 1-hour deltas, 6-hour windows and 24-hour recent-condition features.

Sampling frequency and prediction horizon are different concepts. This project predicts the event at the current stream time from information available strictly before that time; it does not implement a delayed six-hour-ahead label protocol.

## 11.2 Why current temperature is excluded

The target is literally defined by current temperature. If `temperature_2m_t` were included in `X_t`, the model would receive the answer. Therefore the temperature predictor is `temperature_2m_lag1` and all rolling features terminate at `t-1`.

## 11.3 Why Representation A is city-specific

The assignment requires core previous-state weather information but permits justified additional variables. Warm-up EDA was used to identify redundant/constant variables without consulting evaluation results.

Important decisions include:

- retain only total cloud cover rather than multiple redundant cloud layers;
- exclude weather code as an untreated categorical variable;
- exclude raw wind direction because circular angles should not be treated as ordinary continuous values without sine/cosine transformation;
- exclude apparent temperature where redundant with direct temperature/dew-point information;
- exclude VPD when highly redundant with relative humidity;
- exclude rain when redundant with precipitation;
- exclude surface pressure when redundant with mean sea-level pressure;
- retain one representative deeper soil temperature/moisture measure;
- exclude snowfall in Porto because it is constant/near-noninformative there;
- retain snowfall in Bragança because it occurs and has operational meaning.

Low simple Pearson correlation by itself was not treated as sufficient reason to remove a feature, because nonlinear stream learners can use relationships not captured by one linear coefficient.

## 11.4 Why 6-hour and 24-hour windows?

Representation B adds temporal context beyond the immediately previous observation:

- 1-hour delta: short-term direction/change;
- 6-hour mean: short recent temperature level;
- 6-hour std: short recent variability;
- 24-hour Cold Stress count: persistence over a complete daily cycle.

Warm-up-only lag diagnostics were used as supporting evidence rather than optimising the window lengths on evaluation performance.

## 11.5 Why both representations wait 24 observations

Although A itself could technically produce `t-1` features earlier, both A and B start model updates only after the common 24-hour history is available. This removes a hidden comparison difference: both representations are trained/evaluated from equivalent stream positions.

## 11.6 Why a two-year warm-up?

Warm-up allows incremental models, online scaling and temporal feature state to be initialised using historical observations before the official evaluation begins. It also supplies the only observations allowed to estimate the target threshold and make modelling-design diagnostics.

## 11.7 Why state is not reset on 1 January 2018

Resetting would artificially discard legitimate information available immediately before evaluation. The final state from 31 December 2017 therefore flows directly into the first 2018 prediction.

## 11.8 Why prequential evaluation?

A streaming system must predict before observing the answer. The implementation therefore enforces:

```text
PREDICT -> EVALUATE -> LEARN
```

Learning first would leak the current label into the scored prediction.

## 11.9 Why a 168-hour recent metric window?

168 hourly observations equal seven complete days. The window provides a recent operational view while spanning all hours of a weekly interval. Cumulative metrics remain available simultaneously.

## 11.10 Why Hellinger trees?

The Hoeffding Tree and ARF configurations use the Hellinger split criterion, a sensible fixed choice for an imbalanced binary streaming target. Parameters were frozen before the official evaluation and were not selected by maximising the 2018–2025 results.

## 11.11 Why persistence and transition diagnostics were added

The official target is highly persistent at an hourly scale. A persistence reference:

```text
y_hat_t = y_(t-1)
```

provides context for high absolute scores without replacing any mandatory classifier.

Transition diagnostics isolate:

```text
0 -> 1  Cold Stress onset
1 -> 0  Cold Stress ending
```

These events are useful supplementary evidence because pure persistence necessarily fails whenever the state changes.

---

# 12. Generated artefacts

## Data

```text
data/raw/*_weather.json
data/raw/*_provenance.json
data/processed/*_hourly.csv
data/processed/*_quality_initial.json
```

## EDA outputs

```text
outputs/eda_tables/
outputs/eda_figures/
```

## Experiment outputs

```text
results/experiment_manifest.json
results/summaries/experiment_summary.csv
results/histories/*.csv
results/warmup_states/*.json
```

## Result-analysis outputs

```text
outputs/results_tables/
outputs/results_figures/
```

The histories and warm-up states are generated but excluded from Git due to their size. The summary, manifest, notebooks and report-ready compact outputs are sufficient to understand the submitted experiment, while histories can be regenerated with one command.

---

# 13. Test architecture

## `tests/test_data.py`

Checks acquisition helpers, API payload structure, chronological timestamp conversion, retry behaviour and JSON saving.

## `tests/test_preprocessing.py`

Checks pooled warm-up threshold isolation, target boundary behaviour, exact timeline comparison and inclusive period selection.

## `tests/test_features.py`

Checks:

- exact selected feature sets;
- A/B relationship;
- sliding-window hand calculations;
- common 24-hour initialisation;
- bounded memory;
- temporal-feature calculations against independent oracles;
- transform/update separation;
- protection against current-observation leakage.

## `tests/test_evaluation.py`

Checks:

- confusion-matrix arithmetic;
- cumulative/recent metric behaviour;
- 168-hour default window;
- undefined positive metrics;
- exact prequential call order;
- warm-up state construction;
- feature-state continuity into evaluation;
- evaluation summary and cost calculations.

## `tests/test_real_data_integration.py`

Checks the committed Porto/Bragança snapshots directly:

- common complete chronology;
- 87,672 full rows per city;
- 17,544 warm-up rows;
- 70,128 evaluation rows;
- frozen 8.7 °C threshold;
- structural quality;
- first 2018 B features against an independent DataFrame calculation;
- full stream processing without dropped evaluation events.

## `tests/test_models.py`

Checks that all five mandatory River factories exist, return independent fresh objects and support online `predict_one` / `learn_one` behaviour.

## `tests/test_run_experiments.py`

Checks the orchestrator and verifies that the complete experiment contains exactly the configured 20 city/representation/model combinations.

Run everything with:

```bash
uv run python -m unittest discover -s tests -v
```

---

# 14. Reproducibility and audit invariants

The following invariants should hold for the official data snapshot:

```text
Full rows per city:       87,672
Warm-up rows per city:    17,544
Evaluation rows per city: 70,128
Warm-up model updates:    17,520
Cold threshold:            8.7 °C
Official runs:            20
Cold starts in evaluation: 0
Random seed:              42
```

All 20 official histories must have the same 70,128 timestamps within each common 2018–2025 evaluation timeline.

The summary confusion-matrix counts must satisfy:

```text
TN + FP + FN + TP = 70,128
```

for every run.

---

# 15. Execution order for a full reproduction

```text
1. uv sync
2. run tests
3. use committed data snapshot (or optionally rerun data acquisition)
4. execute 01_data_quality_eda.ipynb
5. run python -m src.run_experiments
6. execute 02_results_analysis.ipynb
7. use exported outputs in the report
```

Commands are documented in `README.md`.

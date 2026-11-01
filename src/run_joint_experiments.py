

import json
from time import perf_counter

import pandas as pd

from config.experiment import (
    BASE_FEATURES, CITY_A, CITY_B, EVALUATION_LABELS,
    EVALUATION_WINDOW, MODEL_NAMES, MODEL_PARAMS, RANDOM_SEED,
    REPRESENTATIONS, TARGET_COLUMN,
)
from src.evaluation import (
    StreamingEvaluator,
    build_evaluation_summary,
    summarize_model_state,
)
from src.features import StreamFeatureBuilder, get_feature_columns
from src.models import make_model_factories
from src.run_experiments import PROJECT_ROOT, prepare_experiment_data

CITIES = (CITY_A, CITY_B)
JOINT_DIR = PROJECT_ROOT / "results" / "joint"


def aligned_rows(data):
    """Percorre timestamps sincronizados das duas cidades."""

    if set(data) != set(CITIES):
        raise ValueError("Esperados dados de Porto e Braganca")

    if len(data[CITY_A]) != len(data[CITY_B]):
        raise ValueError("Streams com comprimentos diferentes")

    previous = None

    for row_a, row_b in zip(
        data[CITY_A].itertuples(index=False),
        data[CITY_B].itertuples(index=False),
    ):
        ta = pd.Timestamp(row_a.timestamp_utc)
        tb = pd.Timestamp(row_b.timestamp_utc)

        if ta != tb:
            raise ValueError(f"Timestamps diferentes: {ta} / {tb}")

        if previous is not None and ta <= previous:
            raise ValueError("Stream conjunto fora da ordem cronologica")

        previous = ta

        yield {CITY_A: row_a, CITY_B: row_b}


def run_joint_experiment(
    *,
    representation,
    model_name,
    warmup_data,
    evaluation_data,
    model_factories,
    threshold,
):
    if BASE_FEATURES[CITY_A] != BASE_FEATURES[CITY_B]:
        raise ValueError("As features das cidades nao sao iguais")

    if (
        get_feature_columns(CITY_A, representation)
        != get_feature_columns(CITY_B, representation)
    ):
        raise ValueError("Representacoes com colunas diferentes")


    model = model_factories[model_name]()


    builders = {
        c: StreamFeatureBuilder(c, representation)
        for c in CITIES
    }

    warmup_updates = {c: 0 for c in CITIES}

    # WARM-UP: 2016-2017
    for rows in aligned_rows(warmup_data):

        x = {
            c: builders[c].transform_one()
            for c in CITIES
        }

        y = {
            c: int(getattr(rows[c], TARGET_COLUMN))
            for c in CITIES
        }

        for c in CITIES:
            if x[c] is not None:
                model.learn_one(x[c], y[c])
                warmup_updates[c] += 1

        for c in CITIES:
            builders[c].update_one(rows[c], y[c])

    if not all(builders[c].ready for c in CITIES):
        raise RuntimeError("Warm-up insuficiente")

    warmup_state = {
        "model_updates": sum(warmup_updates.values()),
        "model_updates_per_city": warmup_updates,
        "model": summarize_model_state(model),
        "builders": {
            c: builders[c].state_snapshot()
            for c in CITIES
        },
    }

  
    evaluators = {
        c: StreamingEvaluator(
            labels=EVALUATION_LABELS,
            window_size=EVALUATION_WINDOW,
        )
        for c in CITIES
    }

    histories = {c: [] for c in CITIES}
    prediction_times = {c: 0.0 for c in CITIES}
    update_times = {c: 0.0 for c in CITIES}

    # EVALUATION: 2018-2025
    for rows in aligned_rows(evaluation_data):

        x = {
            c: builders[c].transform_one()
            for c in CITIES
        }

        if any(x[c] is None for c in CITIES):
            raise RuntimeError(
                "Features indisponiveis durante a avaliacao"
            )

        # PREDICT BOTH
        predictions = {}

        for c in CITIES:
            start = perf_counter()
            predictions[c] = model.predict_one(x[c])
            prediction_times[c] += perf_counter() - start

        # EVALUATE BOTH
        y = {
            c: int(getattr(rows[c], TARGET_COLUMN))
            for c in CITIES
        }

        snapshots = {
            c: evaluators[c].observe(y[c], predictions[c])
            for c in CITIES
        }

        # LEARN BOTH
        # O MESMO classificador aprende com as duas cidades.
        for c in CITIES:
            start = perf_counter()
            model.learn_one(x[c], y[c])
            update_times[c] += perf_counter() - start

        # UPDATE CITY-SPECIFIC WINDOWS
        for c in CITIES:
            builders[c].update_one(rows[c], y[c])

            temporal = {
                key: val
                for key, val in x[c].items()
                if key == "temperature_2m_lag1"
                or key.startswith((
                    "temperature_delta_",
                    "temperature_mean_",
                    "temperature_std_",
                    "cold_stress_hours_",
                ))
            }

            histories[c].append({
                "city": c,
                "representation": representation,
                "model": model_name,
                "timestamp": rows[c].timestamp_utc,
                "y_true": y[c],
                "y_pred": predictions[c],
                **temporal,
                **snapshots[c],
            })

    
    summaries = {}

    for c in CITIES:
        result = build_evaluation_summary(
            evaluator=evaluators[c],
            predict_seconds=prediction_times[c],
            update_seconds=update_times[c],
        )

        result.update({
            "protocol": "joint",
            "city": c,
            "representation": representation,
            "model": model_name,
            "model_params": json.dumps(
                MODEL_PARAMS.get(model_name, {}),
                sort_keys=True,
            ),
            "random_seed": RANDOM_SEED,
            "cold_threshold": float(threshold),
            "evaluation_metric_window_hours": EVALUATION_WINDOW,
            "warmup_rows": len(warmup_data[c]),
            "evaluation_rows": len(evaluation_data[c]),
            "warmup_model_updates": warmup_state["model_updates"],
            "warmup_model_updates_local": warmup_updates[c],
            "warmup_feature_builder_ready": builders[c].ready,
        })

        summaries[c] = result

        if len(histories[c]) != len(evaluation_data[c]):
            raise RuntimeError(f"Historico incompleto: {c}")

    return (
        summaries,
        {c: pd.DataFrame(histories[c]) for c in CITIES},
        warmup_state,
        model,
    )


def run_all_joint_experiments():
    streams, threshold = prepare_experiment_data()
    factories = make_model_factories()



    summary_path = (
        PROJECT_ROOT
        / "results"
        / "summaries"
        / "experiment_summary.csv"
    )

    if not summary_path.exists():
        raise FileNotFoundError(
            "Executa primeiro src.run_experiments."
        )

    individual = pd.read_csv(summary_path)

    keys = ["city", "representation", "model"]

    required = set(keys) | {
        "macro_f1",
        "recall_positive",
        "prediction_ms_per_event",
        "update_ms_per_event",
        "cold_threshold",
    }

    if required - set(individual.columns):
        raise ValueError("Faltam colunas no summary individual.")

    expected = {
        (city, rep, model)
        for city in CITIES
        for rep in REPRESENTATIONS
        for model in MODEL_NAMES
    }

    actual = set(
        individual[keys].itertuples(index=False, name=None)
    )

    if actual != expected or len(individual) != len(expected):
        raise ValueError(
            "Os 20 resultados individuais nao estao completos."
        )

    if individual[list(required - set(keys))].isna().any().any():
        raise ValueError("Existem metricas individuais em falta.")

    if not all(
        abs(float(value) - float(threshold)) < 1e-9
        for value in individual["cold_threshold"]
    ):
        raise ValueError(
            "O threshold dos resultados individuais e diferente."
        )


    individual["total_ms_per_event"] = (
        individual["prediction_ms_per_event"]
        + individual["update_ms_per_event"]
    )


    best_per_city = (
        individual
        .sort_values(
            [
                "city",
                "macro_f1",
                "recall_positive",
                "total_ms_per_event",
                "model",
                "representation",
            ],
            ascending=[
                True, False, False, True, True, True
            ],
        )
        .drop_duplicates("city")
        .reset_index(drop=True)
    )

    selected_configs = list(
        best_per_city[
            ["model", "representation"]
        ]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )

    print("\nMelhores modelos individuais:")
    print(
        best_per_city[
            ["city", "model", "representation",
             "macro_f1", "recall_positive"]
        ].to_string(index=False)
    )

    print("\nConfiguracoes conjuntas:", selected_configs)


    missing = {
        model_name
        for model_name, rep in selected_configs
    } - set(factories)
    if missing:
        raise ValueError(f"Faltam modelos: {sorted(missing)}")

    summary_dir = JOINT_DIR / "summaries"
    history_dir = JOINT_DIR / "histories"
    warmup_dir = JOINT_DIR / "warmup_states"

    for directory in (summary_dir, history_dir, warmup_dir):
        directory.mkdir(parents=True, exist_ok=True)


    for path in history_dir.glob("*.csv"):
        path.unlink()

    for path in warmup_dir.glob("*.json"):
        path.unlink()


    manifest = {
        "protocol": "joint",
        "cities": list(CITIES),
        "threshold": float(threshold),
        "feature_columns": {
            rep: get_feature_columns(CITY_A, rep)
            for rep in REPRESENTATIONS
        },
        "selection_metric": "macro_f1",
        "selection_tiebreakers": [
            "recall_positive",
            "total_ms_per_event",
        ],
        "selected_by_city": best_per_city[
            [
                "city",
                "model",
                "representation",
                "macro_f1",
                "recall_positive",
            ]
        ].to_dict("records"),
        "joint_configurations": [
            {
                "model": model,
                "representation": rep,
            }
            for model, rep in selected_configs
        ],
        "update_order": list(CITIES),
        "rule": (
            "predict both -> evaluate both -> "
            "learn both -> update both builders"
        ),
    }


    (JOINT_DIR / "joint_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    results = []


    selected_set = set(selected_configs)

    for rep in REPRESENTATIONS:
        for model_name in MODEL_NAMES:

            if (model_name, rep) not in selected_set:
                continue

            print(
                f"Joint | {rep} | {model_name}",
                flush=True,
            )

            summaries, histories, state, _ = run_joint_experiment(
                representation=rep,
                model_name=model_name,
                warmup_data={
                    c: streams[c]["warmup"] for c in CITIES
                },
                evaluation_data={
                    c: streams[c]["evaluation"] for c in CITIES
                },
                model_factories=factories,
                threshold=threshold,
            )

            for c in CITIES:
                results.append(summaries[c])

                histories[c].to_csv(
                    history_dir / (
                        f"{c.lower()}_{rep.lower()}_{model_name}.csv"
                    ),
                    index=False,
                )

            (warmup_dir / (
                f"joint_{rep.lower()}_{model_name}.json"
            )).write_text(
                json.dumps(state, indent=2, allow_nan=False),
                encoding="utf-8",
            )

    df = pd.DataFrame(results)

    df.to_csv(
        summary_dir / "joint_experiment_summary.csv",
        index=False,
    )

    print(
        f"Concluidas {len(selected_configs)} execucoes conjuntas; "
        f"{len(df)} linhas de resultados."
    )

    return df


if __name__ == "__main__":
    run_all_joint_experiments()

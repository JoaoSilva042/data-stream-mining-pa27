import json
from pathlib import Path

import pandas as pd

from config.experiment import (
    CITY_A,
    CITY_B,
    MODEL_NAMES,
    MODEL_PARAMS,
    RANDOM_SEED,
    EVALUATION_WINDOW,
    TARGET_COLUMN,
    REPRESENTATIONS,
)

from src.preprocessing import (
    add_cold_stress_target,
    compare_city_timelines,
    compute_pooled_cold_threshold,
    get_evaluation_stream,
    get_warmup,
    load_city_data,
)

from src.features import (
    StreamFeatureBuilder,
)

from src.models import (
    make_model_factories,
)

from src.evaluation import (
    build_evaluation_summary,
    evaluate_model,
    warm_up_online,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

SUMMARY_DIR = (
    RESULTS_DIR
    / "summaries"
)

HISTORY_DIR = (
    RESULTS_DIR
    / "histories"
)

WARMUP_STATE_DIR = (
    RESULTS_DIR
    / "warmup_states"
)


def prepare_experiment_data():
    """
    Load both city streams, compute the shared
    pooled warm-up threshold, create the target,
    and split each stream into warm-up and
    evaluation periods.

    Returns
    -------
    city_streams : dict
        For each city:
            {
                "full": ...,
                "warmup": ...,
                "evaluation": ...
            }

    threshold : float
        Frozen pooled Cold Stress threshold.
    """


    #  LOAD RAW CITY STREAMS


    city_data = {
        CITY_A: load_city_data(
            CITY_A
        ),

        CITY_B: load_city_data(
            CITY_B
        ),
    }


    #  VERIFY COMMON TIMELINE


    if not compare_city_timelines(
        city_data[CITY_A],
        city_data[CITY_B],
    ):
        raise ValueError(
            "The two cities do not share "
            "the same chronological timeline."
        )


    #  COMPUTE SHARED TARGET THRESHOLD

    
    # The function uses pooled warm-up
    # temperatures only.

    threshold = (
        compute_pooled_cold_threshold(
            [
                city_data[CITY_A],
                city_data[CITY_B],
            ]
        )
    )


    #  CREATE TARGET FOR BOTH CITIES


    for city_name in city_data:

        city_data[city_name] = (
            add_cold_stress_target(
                city_data[city_name],
                threshold,
            )
        )


    # CHRONOLOGICAL PERIODS


    city_streams = {}

    for city_name, data in (
        city_data.items()
    ):

        warmup = get_warmup(
            data
        )

        evaluation = (
            get_evaluation_stream(
                data
            )
        )

        city_streams[
            city_name
        ] = {
            "full":
                data,

            "warmup":
                warmup,

            "evaluation":
                evaluation,
        }

    return (
        city_streams,
        threshold,
    )


def run_single_experiment(
    *,
    city_name,
    representation,
    model_name,
    warmup_data,
    evaluation_data,
    model_factories,
    threshold,
):
    """
    Run one controlled streaming experiment:

        one city
        one representation
        one model

    """


    #  CREATE FRESH MODEL


    if model_name not in model_factories:
        raise ValueError(
            f"Unknown model: "
            f"{model_name}"
        )

    model = (
        model_factories[
            model_name
        ]()
    )


    #  CREATE FRESH FEATURE STATE


    feature_builder = (
        StreamFeatureBuilder(
            city_name=city_name,
            representation=representation,
        )
    )


    #  WARM-UP 2016–2017:


    (
        model,
        feature_builder,
        warmup_state,
    ) = warm_up_online(
        model=model,
        warmup=warmup_data,
        feature_builder=
            feature_builder,
        target_column=
            TARGET_COLUMN,
        return_state=True,
    )

    # The feature builder is NOT reset here.



    # 4. PREQUENTIAL EVALUATION


    # 2018–2025:
    
    # BUILD FEATURES
    #PREDICT
    # OBSERVE TARGET
    #EVALUATE
    # LEARN
    # UPDATE FEATURE STATE

    (
        evaluator,
        history,
        predict_seconds,
        update_seconds,
    ) = evaluate_model(
        model=model,
        data=evaluation_data,
        feature_builder=
            feature_builder,
        target_column=
            TARGET_COLUMN,
        learn=True,
    )


    # SANITY CHECK

    
    # After two years of warm-up the feature builder must be ready, so every evaluation observation should have been processed.

    if len(history) != len(
        evaluation_data
    ):
        raise RuntimeError(
            "Evaluation history length "
            "does not match the evaluation "
            "stream length."
        )


    # 6. FINAL SUMMARY


    summary = (
        build_evaluation_summary(
            evaluator=
                evaluator,

            predict_seconds=
                predict_seconds,

            update_seconds=
                update_seconds,
        )
    )

    # Add experiment metadata.

    summary.update({
        "city":
            city_name,

        "representation":
            representation,

        "model":
            model_name,

                "model_params":
            json.dumps(
                MODEL_PARAMS.get(
                    model_name,
                    {},
                ),
                sort_keys=True,
            ),

        "random_seed":
            RANDOM_SEED,

        "evaluation_metric_window_hours":
            EVALUATION_WINDOW,

        "cold_threshold":
            threshold,

        "warmup_rows":
            len(warmup_data),

        "evaluation_rows":
            len(evaluation_data),

        "warmup_model_updates":
            warmup_state[
                "model_updates"
            ],

        "warmup_positive_events":
            warmup_state[
                "positive_events"
            ],

        "warmup_negative_events":
            warmup_state[
                "negative_events"
            ],

        "warmup_feature_builder_ready":
            warmup_state[
                "feature_builder"
            ][
                "ready"
            ],
    })

    # Useful metadata in the history too.

    history.insert(
        0,
        "city",
        city_name,
    )

    history.insert(
        1,
        "representation",
        representation,
    )

    history.insert(
        2,
        "model",
        model_name,
    )

    return (
        summary,
        history,
        warmup_state,
        model,
    )


def save_warmup_state(
    warmup_state,
    city_name,
    representation,
    model_name,
):
    """
    Save the pre-evaluation streaming-system
    state as JSON.
    """

    WARMUP_STATE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        f"{city_name.lower()}_"
        f"{representation.lower()}_"
        f"{model_name}.json"
    )

    path = (
        WARMUP_STATE_DIR
        / filename
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            warmup_state,
            file,
            indent=2,
            allow_nan=False,
        )

    return path


def save_experiment_history(
    history,
    city_name,
    representation,
    model_name,
):
    """
    Save the prequential evaluation history
    for one experiment.
    """

    HISTORY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        f"{city_name.lower()}_"
        f"{representation.lower()}_"
        f"{model_name}.csv"
    )

    path = (
        HISTORY_DIR
        / filename
    )

    history.to_csv(
        path,
        index=False,
    )

    return path

def save_experiment_manifest(
    *,
    city_streams,
    threshold,
):
    """
    Save the frozen experimental design
    used for the official PA27 runs.
    """

    manifest_path = (
        SUMMARY_DIR.parent
        / "experiment_manifest.json"
    )

    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    first_city = CITY_A

    warmup = (
        city_streams[
            first_city
        ][
            "warmup"
        ]
    )

    evaluation = (
        city_streams[
            first_city
        ][
            "evaluation"
        ]
    )

    full = (
        city_streams[
            first_city
        ][
            "full"
        ]
    )
    def period_info(
        data,
    ):
        """
        Return period metadata when timestamps
        are available.
        """

        result = {
            "rows":
                int(
                    len(data)
                ),
        }

        if (
            "timestamp_utc"
            in data.columns
            and len(data) > 0
        ):
            result[
                "start"
            ] = str(
                data[
                    "timestamp_utc"
                ].iloc[0]
            )

            result[
                "end"
            ] = str(
                data[
                    "timestamp_utc"
                ].iloc[-1]
            )

        return result

    manifest = {
        "assignment":
            "PA27",

        "cities": [
            CITY_A,
            CITY_B,
        ],

        "historical_period":
            period_info(
                full
            ),

        "warmup_period":
            period_info(
                warmup
            ),

        "evaluation_period":
            period_info(
                evaluation
            ),

        "cold_threshold":
            float(threshold),

        "target_column":
            TARGET_COLUMN,

        "random_seed":
            RANDOM_SEED,

        "representations":
            list(
                REPRESENTATIONS
            ),

        "models":
            list(
                MODEL_NAMES
            ),

        "evaluation_metric_window_hours":
            EVALUATION_WINDOW,

        "model_params":
            MODEL_PARAMS,

        "protocol": {
            "warmup":
                (
                    "BUILD -> OBSERVE TARGET "
                    "-> LEARN -> UPDATE STATE"
                ),

            "evaluation":
                (
                    "BUILD -> PREDICT -> "
                    "OBSERVE TARGET -> EVALUATE "
                    "-> LEARN -> UPDATE STATE"
                ),
        },

        "parameter_selection":
            (
                "Parameters selected a priori "
                "before official evaluation; "
                "no tuning on 2018-2025 data."
            ),
    }

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            manifest,
            file,
            indent=2,
            allow_nan=False,
        )

    return manifest_path

def run_all_experiments(
    *,
    save_histories=True,
):
    """
    Run the complete mandatory experiment:

        2 cities
        2 representations
        5 models
        = 20 experiments
    """


    # PREPARE COMMON DATA


    (
        city_streams,
        threshold,
    ) = prepare_experiment_data()

    manifest_path = (
        save_experiment_manifest(
            city_streams=
                city_streams,

            threshold=
                threshold,
        )
    )

    print(
        "Experiment manifest saved: "
        f"{manifest_path}"
    )


    # MODEL FACTORIES



    model_factories = (
        make_model_factories()
    )

    # Validate that the configured model names exist in the factory dictionary.

    missing_models = (
        set(MODEL_NAMES)
        - set(model_factories)
    )

    if missing_models:
        raise ValueError(
            "Missing model factories: "
            f"{sorted(missing_models)}"
        )

    summaries = []


    # COMPLETE EXPERIMENT LOOP


    total_experiments = (
        len(city_streams)
        * len(REPRESENTATIONS)
        * len(MODEL_NAMES)
    )

    experiment_number = 0

    for city_name, streams in (
        city_streams.items()
    ):

        warmup_data = (
            streams["warmup"]
        )

        evaluation_data = (
            streams["evaluation"]
        )

        for representation in REPRESENTATIONS:

            for model_name in (
                MODEL_NAMES
            ):

                experiment_number += 1

                print(
                    "\n"
                    f"[{experiment_number}/"
                    f"{total_experiments}] "
                    f"{city_name} | "
                    f"Representation "
                    f"{representation} | "
                    f"{model_name}"
                )

                (
                    summary,
                    history,
                    warmup_state,
                    _,
                ) = run_single_experiment(
                    city_name=
                        city_name,

                    representation=
                        representation,

                    model_name=
                        model_name,

                    warmup_data=
                        warmup_data,

                    evaluation_data=
                        evaluation_data,

                    model_factories=
                        model_factories,

                    threshold=
                        threshold,
                )

                summaries.append(
                    summary
                )

                if save_histories:

                    path = (
                        save_experiment_history(
                            history=
                                history,

                            city_name=
                                city_name,

                            representation=
                                representation,

                            model_name=
                                model_name,
                        )
                    )

                    print(
                        f"History saved: "
                        f"{path}"
                    )

                    warmup_path = (
                        save_warmup_state(
                            warmup_state=
                                warmup_state,

                            city_name=
                                city_name,

                            representation=
                                representation,

                            model_name=
                                model_name,
                        )
                    )

                    print(
                        "Warm-up state saved: "
                        f"{warmup_path}"
                    )



                print(
                    "Macro-F1: "
                    f"{summary['macro_f1']:.4f}"
                )

                print(
                    "Recall (+): "
                    f"{summary['recall_positive']:.4f}"
                )


    # MASTER RESULTS TABLE


    summaries_df = pd.DataFrame(
        summaries
    )

    SUMMARY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        SUMMARY_DIR
        / "experiment_summary.csv"
    )

    summaries_df.to_csv(
        summary_path,
        index=False,
    )

    print(
        "\nAll experiments completed."
    )

    print(
        f"Summary saved: "
        f"{summary_path}"
    )

    return summaries_df


def main():
    """Run the complete PA27 experiment."""

    summaries = (
        run_all_experiments(
            save_histories=True,
        )
    )

    print(
        "\n"
        "Final experiment summary:"
    )

    print(
        summaries.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
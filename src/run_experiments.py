from pathlib import Path

import pandas as pd

from config.experiment import (
    CITY_A,
    CITY_B,
    MODEL_NAMES,
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


    # 1. LOAD RAW CITY STREAMS


    city_data = {
        CITY_A: load_city_data(
            CITY_A
        ),

        CITY_B: load_city_data(
            CITY_B
        ),
    }


    # 2. VERIFY COMMON TIMELINE


    if not compare_city_timelines(
        city_data[CITY_A],
        city_data[CITY_B],
    ):
        raise ValueError(
            "The two cities do not share "
            "the same chronological timeline."
        )


    # 3. COMPUTE SHARED TARGET THRESHOLD

    #
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


    # 4. CREATE TARGET FOR BOTH CITIES


    for city_name in city_data:

        city_data[city_name] = (
            add_cold_stress_target(
                city_data[city_name],
                threshold,
            )
        )


    # 5. CHRONOLOGICAL PERIODS


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
        × one representation
        × one model

    A fresh model and a fresh feature builder
    are created for every experiment.
    """


    # 1. CREATE FRESH MODEL


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


    # 2. CREATE FRESH FEATURE STATE


    feature_builder = (
        StreamFeatureBuilder(
            city_name=city_name,
            representation=representation,
        )
    )


    # 3. WARM-UP

    #
    # 2016–2017:
    #
    # BUILD FEATURES
    # → OBSERVE TARGET
    # → LEARN
    # → UPDATE FEATURE STATE

    (
        model,
        feature_builder,
    ) = warm_up_online(
        model=model,
        warmup=warmup_data,
        feature_builder=
            feature_builder,
        target_column=
            TARGET_COLUMN,
    )

    # The feature builder is NOT reset here.
    #
    # Its final 2017 state must continue
    # directly into the first event of 2018.


    # 4. PREQUENTIAL EVALUATION

    #
    # 2018–2025:
    #
    # BUILD FEATURES
    # → PREDICT
    # → OBSERVE TARGET
    # → EVALUATE
    # → LEARN
    # → UPDATE FEATURE STATE

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


    # 5. SANITY CHECK

    #
    # After two years of warm-up the feature
    # builder must be ready, so every evaluation
    # observation should have been processed.

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

        "cold_threshold":
            threshold,

        "warmup_rows":
            len(warmup_data),

        "evaluation_rows":
            len(evaluation_data),
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
        model,
    )


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


def run_all_experiments(
    *,
    save_histories=True,
):
    """
    Run the complete mandatory experiment:

        2 cities
        × 2 representations
        × 5 models
        = 20 experiments
    """


    # 1. PREPARE COMMON DATA


    (
        city_streams,
        threshold,
    ) = prepare_experiment_data()


    # 2. MODEL FACTORIES

    #
    # Each factory call creates a new model.

    model_factories = (
        make_model_factories()
    )

    # Validate that the configured model names
    # exist in the factory dictionary.

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


    # 3. COMPLETE EXPERIMENT LOOP


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

                print(
                    "Macro-F1: "
                    f"{summary['macro_f1']:.4f}"
                )

                print(
                    "Recall (+): "
                    f"{summary['recall_positive']:.4f}"
                )


    # 4. MASTER RESULTS TABLE


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
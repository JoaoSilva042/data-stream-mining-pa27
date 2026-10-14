from config.experiment import (
    TARGET_COLUMN,
)

from src.evaluation import (
    build_evaluation_summary,
    evaluate_model,
    warm_up_online,
)

from src.models import (
    make_model_factories,
)


def run_single_experiment(
    *,
    city,
    representation,
    model_name,
    warmup_data,
    evaluation_data,
    feature_columns,
    target_column=TARGET_COLUMN,
):
    """
    Run one controlled
    city × representation × model experiment.
    """

    model_factories = (
        make_model_factories()
    )

    if model_name not in model_factories:
        raise ValueError(
            f"Unknown model: "
            f"{model_name}"
        )

    # Fresh model for this experiment
    model = model_factories[
        model_name
    ]()

    # 2016–2017
    warm_up_online(
        model=model,
        warmup=warmup_data,
        feature_columns=feature_columns,
        target_column=target_column,
    )

    # 2018–2025
    (
        evaluator,
        history,
        predict_seconds,
        update_seconds,
    ) = evaluate_model(
        model=model,
        data=evaluation_data,
        feature_columns=feature_columns,
        target_column=target_column,
        learn=True,
    )

    summary = (
        build_evaluation_summary(
            evaluator=evaluator,
            predict_seconds=
                predict_seconds,
            update_seconds=
                update_seconds,
        )
    )

    summary.update({
        "city":
            city,

        "representation":
            representation,

        "model":
            model_name,
    })

    return (
        summary,
        history,
        model,
    )
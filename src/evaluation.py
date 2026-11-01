from collections import deque
from time import perf_counter

import numpy as np
import pandas as pd

from config.experiment import (
    EVALUATION_LABELS,
    EVALUATION_WINDOW,
    POSITIVE_CLASS,
)


class OnlineConfusionMatrix:
    """
    Incremental confusion matrix.

    """

    def __init__(self, labels):
        self.labels = list(labels)

        self.index = {
            label: i
            for i, label in enumerate(self.labels)
        }

        self.matrix = np.zeros(
            (len(self.labels), len(self.labels)),
            dtype=int,
        )

    def update(self, y_true, y_pred):
        """Add one evaluated prediction."""

        self.matrix[
            self.index[y_true],
            self.index[y_pred],
        ] += 1

    def remove(self, y_true, y_pred):
        """
        Remove one prediction.

        """

        self.matrix[
            self.index[y_true],
            self.index[y_pred],
        ] -= 1

    @property
    def total(self):
        """Number of evaluated predictions."""

        return int(self.matrix.sum())

    @property
    def correct(self):
        """Number of correct predictions."""

        return int(np.trace(self.matrix))

    def accuracy(self):
        """Compute classification accuracy."""

        if not self.total:
            return np.nan

        return self.correct / self.total

    def precision(
        self,
        label,
        undefined=0.0,
    ):
        """Compute precision for one class."""

        i = self.index[label]

        true_positives = self.matrix[i, i]

        predicted_positive = (
            self.matrix[:, i].sum()
        )

        if not predicted_positive:
            return undefined

        return float(
            true_positives
            / predicted_positive
        )

    def recall(
        self,
        label,
        undefined=0.0,
    ):
        """Compute recall for one class."""

        i = self.index[label]

        true_positives = self.matrix[i, i]

        actual_positive = (
            self.matrix[i, :].sum()
        )

        if not actual_positive:
            return undefined

        return float(
            true_positives
            / actual_positive
        )

    def f1(self, label):
        """Compute F1-score for one class."""

        precision = self.precision(label)
        recall = self.recall(label)

        if not (precision + recall):
            return 0.0

        return (
            2
            * precision
            * recall
            / (precision + recall)
        )

    def macro_f1(self):
        """
        Compute the unweighted mean
        F1-score across all classes.
        """

        if not self.total:
            return np.nan

        return float(
            np.mean([
                self.f1(label)
                for label in self.labels
            ])
        )


class StreamingEvaluator:
    """
    Maintain cumulative and recent
    streaming classification metrics.

    """

    def __init__(
        self,
        labels,
        window_size=EVALUATION_WINDOW,
        positive_class=POSITIVE_CLASS,
    ):
        self.labels = list(labels)

        self.cumulative = (
            OnlineConfusionMatrix(
                self.labels
            )
        )

        self.recent = (
            OnlineConfusionMatrix(
                self.labels
            )
        )

        self.positive_class = (
            positive_class
        )

        if (
            self.positive_class
            not in self.labels
        ):
            raise ValueError(
                "positive_class must be "
                "one of the evaluator labels"
            )

        self.window_size = int(
            window_size
        )

        if self.window_size <= 0:
            raise ValueError(
                "window_size must be positive"
            )

        self.recent_pairs = deque()

        self.observations_seen = 0
        self.predictions_evaluated = 0
        self.cold_starts = 0

    def observe(
        self,
        y_true,
        y_pred,
    ):
        """
        Evaluate one prediction and update
        cumulative and recent metrics.
        """

        self.observations_seen += 1

        # Some online models may not be able to predict before seeing any data.
        if y_pred is None:
            self.cold_starts += 1
            return self.snapshot()

        self.predictions_evaluated += 1

        # Cumulative metrics use the  complete evaluation history.
        self.cumulative.update(
            y_true,
            y_pred,
        )

        # If the recent window is full, remove its oldest prediction.
        if (
            len(self.recent_pairs)
            == self.window_size
        ):
            (
                old_true,
                old_pred,
            ) = (
                self.recent_pairs
                .popleft()
            )

            self.recent.remove(
                old_true,
                old_pred,
            )

        # Add the new prediction to
        # the recent window.
        self.recent_pairs.append(
            (y_true, y_pred)
        )

        self.recent.update(
            y_true,
            y_pred,
        )

        return self.snapshot()

    def snapshot(self):
        """
        Return current cumulative and recent
        predictive performance.

        """

        return {
            "cumulative_accuracy":
                self.cumulative
                .accuracy(),

            "window_accuracy":
                self.recent
                .accuracy(),

            "cumulative_macro_f1":
                self.cumulative
                .macro_f1(),

            "window_macro_f1":
                self.recent
                .macro_f1(),

            "cumulative_precision_positive":
                self.cumulative
                .precision(
                    self.positive_class,
                    undefined=np.nan,
                ),

            "window_precision_positive":
                self.recent
                .precision(
                    self.positive_class,
                    undefined=np.nan,
                ),

            "cumulative_recall_positive":
                self.cumulative
                .recall(
                    self.positive_class,
                    undefined=np.nan,
                ),

            "window_recall_positive":
                self.recent
                .recall(
                    self.positive_class,
                    undefined=np.nan,
                ),
        }


def _component_objects(model):
    """
    Return top-level components of a River model
    without modifying the model.
    """

    components = [model]

    steps = getattr(
        model,
        "steps",
        None,
    )

    if isinstance(steps, dict):
        components.extend(
            steps.values()
        )

    elif isinstance(
        steps,
        (list, tuple),
    ):
        for step in steps:
            if (
                isinstance(step, tuple)
                and len(step) == 2
            ):
                components.append(
                    step[1]
                )
            else:
                components.append(step)

    return components


def summarize_model_state(model):
    """
    Return a compact structural description of
    the model after warm-up.

    """

    components = (
        _component_objects(model)
    )

    summary = {
        "model_class":
            type(model).__name__,

        "component_classes": [
            type(component).__name__
            for component in components
        ],
    }

    for component in components:
        weights = getattr(
            component,
            "weights",
            None,
        )

        if weights is not None:
            try:
                summary["n_weights"] = (
                    len(weights)
                )
            except TypeError:
                pass

        if hasattr(
            component,
            "intercept",
        ):
            intercept = getattr(
                component,
                "intercept",
            )

            if isinstance(
                intercept,
                (
                    int,
                    float,
                    np.integer,
                    np.floating,
                ),
            ):
                summary["intercept"] = (
                    float(intercept)
                )

        counts = getattr(
            component,
            "counts",
            None,
        )

        if counts is not None:
            try:
                summary[
                    "scaler_features_tracked"
                ] = len(counts)
            except TypeError:
                pass

        class_counts = getattr(
            component,
            "class_counts",
            None,
        )

        if class_counts is not None:
            try:
                summary["class_counts"] = {
                    str(label): int(count)
                    for label, count
                    in class_counts.items()
                }
            except (
                AttributeError,
                TypeError,
                ValueError,
            ):
                pass

    models = getattr(
        model,
        "models",
        None,
    )

    if models is not None:
        try:
            summary[
                "ensemble_size"
            ] = len(models)
        except TypeError:
            pass

    for attr in (
        "n_nodes",
        "n_leaves",
        "height",
        "n_active_leaves",
        "n_inactive_leaves",
    ):
        value = getattr(
            model,
            attr,
            None,
        )

        if isinstance(
            value,
            (
                int,
                float,
                np.integer,
                np.floating,
            ),
        ):
            summary[attr] = (
                float(value)
                if isinstance(
                    value,
                    (
                        float,
                        np.floating,
                    ),
                )
                else int(value)
            )

    return summary


def build_warmup_state(
    *,
    model,
    feature_builder,
    warmup,
    target_column,
    model_updates,
):
    """
    Capture the streaming-system state
    immediately before evaluation begins.
    """

    targets = (
        warmup[
            target_column
        ]
        .astype(int)
    )

    timestamps = warmup.get(
        "timestamp_utc"
    )

    state = {
        "warmup_rows":
            int(len(warmup)),

        "model_updates":
            int(model_updates),

        "positive_events":
            int(
                (
                    targets
                    == POSITIVE_CLASS
                ).sum()
            ),

        "negative_events":
            int(
                (
                    targets
                    != POSITIVE_CLASS
                ).sum()
            ),

        "model":
            summarize_model_state(
                model
            ),

        "feature_builder":
            feature_builder
            .state_snapshot(),
    }

    if (
        timestamps is not None
        and len(timestamps)
    ):
        state["warmup_start"] = str(
            timestamps.iloc[0]
        )

        state["warmup_end"] = str(
            timestamps.iloc[-1]
        )

    return state


def warm_up_online(
    model,
    warmup,
    feature_builder,
    target_column,
    *,
    return_state=False,
):
    """
    Sequentially initialize both the model
    and the streaming feature state.


    """

    model_updates = 0

    for row in warmup.itertuples(
        index=False
    ):
        # BUILD FEATURES FROM THE PAST
        x = (
            feature_builder
            .transform_one()
        )

        # OBSERVE CURRENT TARGET
        y = int(
            getattr(
                row,
                target_column,
            )
        )

        # LEARN
        
        # During the first 24 hours there
        # is not enough feature history yet.
        if x is not None:
            model.learn_one(
                x,
                y,
            )

            model_updates += 1

        # 4. UPDATE FEATURE STATE
        feature_builder.update_one(
            row,
            y,
        )

   
    if return_state:
        warmup_state = (
            build_warmup_state(
                model=model,
                feature_builder=
                    feature_builder,
                warmup=warmup,
                target_column=
                    target_column,
                model_updates=
                    model_updates,
            )
        )

        return (
            model,
            feature_builder,
            warmup_state,
        )

    return (
        model,
        feature_builder,
    )


def evaluate_model(
    model,
    data,
    feature_builder,
    target_column,
    *,
    learn=True,
    window_size=EVALUATION_WINDOW,
):
    """
    Evaluate an online classifier using
    streaming feature construction and the
    prequential protocol:

        BUILD FEATURES
        PREDICT
        OBSERVE TARGET
        EVALUATE
        LEARN
        UPDATE FEATURE STATE
    """

    evaluator = StreamingEvaluator(
        labels=EVALUATION_LABELS,
        window_size=window_size,
    )

    history = []

    predict_seconds = 0.0
    update_seconds = 0.0

    for row in data.itertuples(
        index=False
    ):
        # 1. BUILD X_t FROM PAST ONLY
        x = (
            feature_builder
            .transform_one()
        )

      
        if x is None:
            raise RuntimeError(
                "Feature builder is not "
                "ready during evaluation."
            )

        # 2. PREDICT
        start = perf_counter()

        y_pred = model.predict_one(x)

        predict_seconds += (
            perf_counter()
            - start
        )

        # OBSERVE CURRENT TARGET
        
        # Only now do we reveal y_t.
        y = int(
            getattr(
                row,
                target_column,
            )
        )

        # 4. EVALUATE
        snapshot = evaluator.observe(
            y,
            y_pred,
        )

        # 5. LEARN
        if learn:
            start = perf_counter()

            model.learn_one(
                x,
                y,
            )

            update_seconds += (
                perf_counter()
                - start
            )

        #  UPDATE FEATURE STATE
        
        # Observation t becomes historical
        # information for t+1.
        feature_builder.update_one(
            row,
            y,
        )

        # Keep only the model-evidence features
        # needed for temporal diagnostics.
        tracked_features = {
            key: value
            for key, value in x.items()
            if (
                key
                == "temperature_2m_lag1"

                or key.startswith(
                    "temperature_delta_"
                )

                or key.startswith(
                    "temperature_mean_"
                )

                or key.startswith(
                    "temperature_std_"
                )

                or key.startswith(
                    "cold_stress_hours_"
                )
            )
        }

        history.append({
            "timestamp":
                row.timestamp_utc,

            "y_true":
                y,

            "y_pred":
                y_pred,

            **tracked_features,
            **snapshot,
        })

    history_df = pd.DataFrame(
        history
    )

    return (
        evaluator,
        history_df,
        predict_seconds,
        update_seconds,
    )


def build_evaluation_summary(
    evaluator,
    predict_seconds,
    update_seconds,
    positive_class=POSITIVE_CLASS,
):
    """
    Build the final predictive and
    computational metrics required
    by the assignment.
    """

    evaluated = (
        evaluator
        .predictions_evaluated
    )

    processed = (
        evaluator
        .observations_seen
    )

    if evaluated == 0:
        raise ValueError(
            "No predictions were evaluated."
        )

    if processed == 0:
        raise ValueError(
            "No observations were processed."
        )

    confusion_matrix = (
        evaluator
        .cumulative
        .matrix
    )

    # For labels (0, 1):
    #
    #            Predicted
    #            0      1
    #
    # True 0     TN     FP
    # True 1     FN     TP

    tn = int(
        confusion_matrix[0, 0]
    )

    fp = int(
        confusion_matrix[0, 1]
    )

    fn = int(
        confusion_matrix[1, 0]
    )

    tp = int(
        confusion_matrix[1, 1]
    )

    return {
        "observations_processed":
            processed,

        "events":
            evaluated,

        "accuracy":
            evaluator
            .cumulative
            .accuracy(),

        "macro_f1":
            evaluator
            .cumulative
            .macro_f1(),

        "precision_positive":
            evaluator
            .cumulative
            .precision(
                positive_class
            ),

        "recall_positive":
            evaluator
            .cumulative
            .recall(
                positive_class
            ),

        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,

        "cold_starts":
            evaluator
            .cold_starts,

        "prediction_ms_per_event":
            (
                1000
                * predict_seconds
                / processed
            ),

        "update_ms_per_event":
            (
                1000
                * update_seconds
                / processed
            ),
    }

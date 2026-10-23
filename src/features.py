from collections import deque
import math

from config.experiment import (
    BASE_FEATURES,
    RECENT_EVENT_WINDOW_HOURS,
    REPRESENTATIONS,
    TEMPORAL_WINDOW_HOURS,
)


class InstanceSlidingWindow:
    """
    Fixed-size sliding window with exact
    incremental statistics.

    Adapted from the Week 3 lab.
    """

    def __init__(self, size):
        if size <= 0:
            raise ValueError(
                "size must be positive"
            )

        self.size = int(size)
        self.data = deque()
        self.sum = 0.0
        self.sum_sq = 0.0

    def append(self, value):
        """
        Add one value and remove the oldest
        one when the window is full.
        """

        x = float(value)
        expired = None

        if len(self.data) == self.size:
            expired = self.data.popleft()
            self.sum -= expired
            self.sum_sq -= (
                expired * expired
            )

        self.data.append(x)
        self.sum += x
        self.sum_sq += x * x

        return expired

    @property
    def count(self):
        """Number of active observations."""

        return len(self.data)

    @property
    def mean(self):
        """Current sliding-window mean."""

        if not self.count:
            return float("nan")

        return self.sum / self.count

    @property
    def variance(self):
        """
        Sample variance inside the
        current sliding window.
        """

        n = self.count

        if n < 2:
            return float("nan")

        numerator = (
            self.sum_sq
            - (
                self.sum
                * self.sum
            ) / n
        )

        return max(
            0.0,
            numerator / (n - 1),
        )

    @property
    def std(self):
        """Current sliding-window standard deviation."""

        if self.count < 2:
            return float("nan")

        return math.sqrt(
            self.variance
        )

    @property
    def values(self):
        """Current active observations."""

        return list(self.data)


def get_base_features(city_name):
    """
    Return the frozen base weather variables
    selected for one city.
    """

    if city_name not in BASE_FEATURES:
        raise ValueError(
            f"Unknown city: {city_name}"
        )

    return list(
        BASE_FEATURES[city_name]
    )


def get_feature_columns(
    city_name,
    representation,
):
    """
    Return the names of the predictors
    produced for one representation.
    """

    if representation not in REPRESENTATIONS:
        raise ValueError(
            "Unknown representation: "
            f"{representation}"
        )

    representation_a = [
        f"{feature}_lag1"
        for feature
        in get_base_features(city_name)
    ]

    if representation == "A":
        return representation_a

    representation_b_extra = [
        "temperature_delta_1h",

        (
            "temperature_mean_"
            f"{TEMPORAL_WINDOW_HOURS}h"
        ),

        (
            "temperature_std_"
            f"{TEMPORAL_WINDOW_HOURS}h"
        ),

        (
            "cold_stress_hours_"
            f"{RECENT_EVENT_WINDOW_HOURS}h"
        ),
    ]

    return (
        representation_a
        + representation_b_extra
    )


class StreamFeatureBuilder:
    """
    Build predictors incrementally from
    past observations only.

    At time t:

        transform_one()
            uses observations < t

        update_one(row_t, y_t)
            incorporates observation t

    Therefore, the current event can only
    affect predictions from t+1 onwards.
    """

    def __init__(
        self,
        city_name,
        representation,
    ):
        if city_name not in BASE_FEATURES:
            raise ValueError(
                f"Unknown city: {city_name}"
            )

        if representation not in REPRESENTATIONS:
            raise ValueError(
                "Unknown representation: "
                f"{representation}"
            )

        self.city_name = city_name
        self.representation = representation

        self.base_features = (
            get_base_features(
                city_name
            )
        )

        # Complete weather state from t-1.
        self.previous_values = None

        # Last two temperatures:
        # needed for T(t-1) - T(t-2).
        self.temperature_history = deque(
            maxlen=2
        )

        # Sliding window containing
        # the last 6 temperatures.
        self.temperature_window = (
            InstanceSlidingWindow(
                TEMPORAL_WINDOW_HOURS
            )
        )

        # Last 24 known Cold Stress labels.
        #
        # Since the target is binary,
        # window.sum is exactly the number
        # of Cold Stress hours.
        self.cold_stress_window = (
            InstanceSlidingWindow(
                RECENT_EVENT_WINDOW_HOURS
            )
        )

        self.observations_seen = 0

    @property
    def ready(self):
        """
        Whether enough historical information
        exists to build a model input.

        A and B deliberately use the same
        24-hour initialization period so their
        comparison uses the same history.
        """

        if (
            self.observations_seen
            < RECENT_EVENT_WINDOW_HOURS
        ):
            return False

        if self.previous_values is None:
            return False

        if len(
            self.temperature_history
        ) < 2:
            return False

        if (
            self.temperature_window.count
            < TEMPORAL_WINDOW_HOURS
        ):
            return False

        if (
            self.cold_stress_window.count
            < RECENT_EVENT_WINDOW_HOURS
        ):
            return False

        return True

    def transform_one(self):
        """
        Build X_t using only the state
        accumulated before time t.

        Returns None while the initial
        history is still being collected.
        """

        if not self.ready:
            return None

        # ----------------------------------
        # Representation A
        # ----------------------------------

        x = {
            f"{feature}_lag1":
                self.previous_values[
                    feature
                ]

            for feature
            in self.base_features
        }

        if self.representation == "A":
            return x

        # ----------------------------------
        # Representation B
        # ----------------------------------
        #
        # Representation B contains all
        # features from A plus temporal
        # context.

        x[
            "temperature_delta_1h"
        ] = (
            self.temperature_history[-1]
            - self.temperature_history[-2]
        )

        x[
            (
                "temperature_mean_"
                f"{TEMPORAL_WINDOW_HOURS}h"
            )
        ] = (
            self.temperature_window.mean
        )

        x[
            (
                "temperature_std_"
                f"{TEMPORAL_WINDOW_HOURS}h"
            )
        ] = (
            self.temperature_window.std
        )

        x[
            (
                "cold_stress_hours_"
                f"{RECENT_EVENT_WINDOW_HOURS}h"
            )
        ] = (
            self.cold_stress_window.sum
        )

        return x

    def update_one(
        self,
        row,
        y,
    ):
        """
        Incorporate the current observation
        into the feature state.

        This must be called only after the
        current event has been processed.
        """

        if y not in (0, 1):
            raise ValueError(
                "Target must be binary (0 or 1)."
            )

        y = int(y)

        current_values = {
            feature: float(
                getattr(
                    row,
                    feature,
                )
            )
            for feature
            in self.base_features
        }

        current_temperature = (
            current_values[
                "temperature_2m"
            ]
        )

        # Current temperature becomes
        # historical information.
        self.temperature_history.append(
            current_temperature
        )

        self.temperature_window.append(
            current_temperature
        )

        # Current target becomes available
        # for future event-frequency features.
        self.cold_stress_window.append(
            y
        )

        # Current weather state becomes t-1
        # when the next observation arrives.
        self.previous_values = (
            current_values
        )

        self.observations_seen += 1

    def state_snapshot(self):
        """
        Return a compact read-only snapshot of
        the feature state at the current stream
        position.

        This is used at the warm-up/evaluation
        boundary for auditability. Calling it
        must not modify the stream state.
        """

        next_features = (
            self.transform_one()
            if self.ready
            else None
        )

        return {
            "city":
                self.city_name,

            "representation":
                self.representation,

            "observations_seen":
                int(self.observations_seen),

            "ready":
                bool(self.ready),

            "base_feature_count":
                len(self.base_features),

            "temperature_history":
                list(
                    self.temperature_history
                ),

            "temperature_window":
                list(
                    self.temperature_window
                    .values
                ),

            "cold_stress_window":
                list(
                    self.cold_stress_window
                    .values
                ),

            "previous_values":
                (
                    dict(
                        self.previous_values
                    )
                    if self.previous_values
                    is not None
                    else None
                ),

            "next_features":
                (
                    dict(next_features)
                    if next_features
                    is not None
                    else None
                ),
        }

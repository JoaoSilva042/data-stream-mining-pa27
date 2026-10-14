from river import (
    ensemble,
    forest,
    linear_model,
    naive_bayes,
    optim,
    tree,
)

from config.experiment import (
    MODEL_PARAMS,
    RANDOM_SEED,
)


def make_gaussian_nb():
    """Create a fresh River Gaussian Naive Bayes classifier."""

    params = MODEL_PARAMS[
        "gaussian_nb"
    ].copy()

    return naive_bayes.GaussianNB(
        **params
    )


def make_logistic_regression():
    """Create a fresh online Logistic Regression classifier."""

    params = MODEL_PARAMS[
        "logistic_regression"
    ].copy()

    learning_rate = params.pop(
        "learning_rate"
    )

    return linear_model.LogisticRegression(
        optimizer=optim.SGD(
            learning_rate
        ),
        **params,
    )


def make_base_tree():
    """
    Create the Hoeffding Tree used alone
    and as the Online Bagging base learner.
    """

    params = MODEL_PARAMS[
        "hoeffding_tree"
    ].copy()

    return tree.HoeffdingTreeClassifier(
        **params
    )


def make_model_factories():
    """
    Return fresh-model factories for
    the five mandatory classifiers.
    """

    bagging_params = MODEL_PARAMS[
        "online_bagging"
    ].copy()

    arf_params = MODEL_PARAMS[
        "adaptive_random_forest"
    ].copy()

    return {
        "gaussian_nb":
            make_gaussian_nb,

        "logistic_regression":
            make_logistic_regression,

        "hoeffding_tree":
            make_base_tree,

        "online_bagging":
            lambda: ensemble.BaggingClassifier(
                model=make_base_tree(),
                seed=RANDOM_SEED,
                **bagging_params,
            ),

        "adaptive_random_forest":
            lambda: forest.ARFClassifier(
                seed=RANDOM_SEED,
                **arf_params,
            ),
    }
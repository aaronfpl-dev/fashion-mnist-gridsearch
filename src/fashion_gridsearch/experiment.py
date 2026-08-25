"""Reproduce the complete TensorFlow + SciKeras grid search."""

from __future__ import annotations

import random
from time import perf_counter

import numpy as np
from scikeras.wrappers import KerasClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

from .config import CV_FOLDS, EPOCHS, PARAM_GRID, RANDOM_SEED


def set_reproducible_seeds() -> None:
    import tensorflow as tf

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.keras.utils.set_random_seed(RANDOM_SEED)


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from tensorflow.keras.datasets import fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_images = train_images.astype("float32").reshape(-1, 784) / 255.0
    test_images = test_images.astype("float32").reshape(-1, 784) / 255.0
    return train_images, train_labels, test_images, test_labels


def create_model(optimizer: str = "adam", neurons: int = 128):
    from tensorflow.keras import Input, Sequential, layers

    model = Sequential(
        [
            Input(shape=(784,)),
            layers.Dense(neurons, activation="relu"),
            layers.Dense(neurons // 2, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def run() -> dict[str, object]:
    set_reproducible_seeds()
    x_train, y_train, x_test, y_test = load_data()
    estimator = KerasClassifier(model=create_model, epochs=EPOCHS, verbose=0)
    search = GridSearchCV(
        estimator=estimator,
        param_grid=PARAM_GRID,
        cv=CV_FOLDS,
        scoring="accuracy",
        refit=True,
        verbose=2,
        n_jobs=1,
    )
    started = perf_counter()
    search.fit(x_train, y_train)
    elapsed = perf_counter() - started
    predictions = search.predict(x_test)
    return {
        "best_params": search.best_params_,
        "mean_cv_accuracy": float(search.best_score_),
        "test_accuracy": float(accuracy_score(y_test, predictions)),
        "elapsed_seconds": elapsed,
        "classification_report": classification_report(
            y_test, predictions, output_dict=True
        ),
        "cv_results": search.cv_results_,
    }


if __name__ == "__main__":
    print(run())


"""Search-space definition kept independent from the TensorFlow runtime."""

CV_FOLDS = 3
EPOCHS = 10
RANDOM_SEED = 42

PARAM_GRID = {
    "model__optimizer": ["adam", "rmsprop"],
    "model__neurons": [64, 128, 256],
    "batch_size": [32, 64],
}


def number_of_candidates() -> int:
    total = 1
    for values in PARAM_GRID.values():
        total *= len(values)
    return total


def number_of_fits() -> int:
    return number_of_candidates() * CV_FOLDS


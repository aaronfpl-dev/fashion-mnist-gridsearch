# Fashion-MNIST Grid Search

A GPU-assisted hyperparameter search for a dense TensorFlow classifier on the official Fashion-MNIST split. The project evaluates **12 configurations × 3 folds = 36 fits**, keeps the 10,000-image test set untouched during selection, and documents how human judgment narrowed an LLM-suggested search space.

![Experiment summary](results/experiment_summary.png)

## Captured Colab T4 result

| Metric | Result |
|---|---:|
| Best batch size | 32 |
| Best hidden width | 128 neurons |
| Best optimizer | Adam |
| Mean 3-fold validation accuracy | **0.8841** |
| Held-out test accuracy | **0.8748** |
| Complete search time | 929.09 seconds |

The most difficult class was `Shirt` (F1 0.67), while `Trouser`, `Bag`, `Sandal`, and both footwear classes were classified substantially more reliably.

## What is included

- Executed notebook with all 36 fit logs, classification report, heatmap, and confusion matrix
- Reusable TensorFlow/SciKeras experiment module with deterministic seeds
- Captured run and class-level metrics in CSV
- A transparent note explaining which LLM suggestions were accepted or rejected
- Tests that verify the search contract and completed notebook conclusions

## Run

The complete search is designed for Python 3.10–3.12 and benefits from a CUDA-capable TensorFlow runtime:

```bash
python -m pip install -e ".[dev]"
python -m fashion_gridsearch.experiment
pytest
```

The Fashion-MNIST loader downloads the official 60,000 training and 10,000 test images automatically. GPU availability changes runtime, not the experiment definition.

## Data source

Fashion-MNIST was created by Zalando Research as a drop-in image-classification benchmark: 70,000 grayscale 28×28 images across 10 balanced classes, with the official 60,000/10,000 split. The source dataset is MIT-licensed and is not duplicated in this repository.

## Author

Aaron Fernandez Pinto — Data Science student at Universidad Autónoma de Baja California (UABC).


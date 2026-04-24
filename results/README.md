# Results

This directory contains the full evaluation results and training logs for our SimpleShot reproduction.

## Files

- **`results_summary.txt`** — Final numerical results comparing our reproduction to the paper. Contains two tables:
  - Table 1: Best epoch evaluation (early stopping via L2N 1-shot on val)
  - Table 2: Epoch 90 evaluation (final checkpoint)

- **`training_log.txt`** — Full training logs for all 4 backbones (Conv-4, ResNet-10, ResNet-18, DenseNet-121), including per-epoch loss, training accuracy, validation accuracy, and best-epoch tracking. Also contains the best-checkpoint evaluation over 10,000 episodes.

- **`epoch_90_evaluation.txt`** — Separate evaluation run using the final epoch 90 checkpoint for each backbone (rather than the best-val checkpoint).

## Summary

All 4 backbones trained for 90 epochs on miniImageNet base classes, evaluated on novel test classes over 10,000 few-shot episodes (5-way, 1-shot and 5-shot).

See the main [README](../README.md) for the results table and gap analysis.

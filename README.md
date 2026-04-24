# SimpleShot Reproduction

Reproduction of *SimpleShot: Revisiting Nearest-Neighbor Classification for Few-Shot Learning* (Wang et al., 2019) [[arXiv:1911.04623]](https://arxiv.org/abs/1911.04623) on the miniImageNet benchmark.

Original repo: [mileyan/simple_shot](https://github.com/mileyan/simple_shot)

## Team

Cornell University
- Woody Chang
- Jiarui Zhang
- Xinyu Tan
- Qiao Xiao

## Chosen Result

We target **Table 1** of the paper: five-way one-shot and five-shot accuracy on miniImageNet across multiple backbones (Conv-4, ResNet-10, ResNet-18, DenseNet-121) with three feature transformations (UN, L2N, CL2N). This table is the paper's central empirical claim that simple feature transformations on standard cross-entropy features can match complex meta-learning methods for few-shot learning. Reproducing it validates the paper's core thesis across architectures of varying capacity.

## GitHub Contents

- `code/` — Implementation notebooks:
  - `SimpleShotReproduce_4Models_AUTO.ipynb` — Main notebook. One Run All trains and evaluates all 4 models (skips training if checkpoint exists).
  - `JiaruiZhang_SimpleShotReproduce_4Models_AUTO.ipynb` — Epoch 90 evaluation variant.
  - `SimpleShotReproduce_EXACT_GITHUB.ipynb` — Original repo code run in our environment for comparison.
- `data/` — Dataset instructions (files hosted on Google Drive due to size).
- `results/` — Evaluation logs and result tables.
- `poster/poster.pdf` — Poster session PDF.
- `report/report.pdf` — Project report.
- `LICENSE` — MIT license.

## Reproduction Steps

1. Open `SimpleShotReproduce_4Models_AUTO.ipynb` in Google Colab.
2. Mount your Drive. Place the pickle files in a folder named `SimpleShot/` inside your Drive root (or update `PKL_CANDIDATES` in Cell 2).
3. Optionally download checkpoints from the Drive link above into `checkpoints_<model>/` to skip training.
4. Run All. The notebook will train any missing models and evaluate all four.

**Environment:** Tested on Google Colab G4 GPU (NVIDIA RTX PRO 6000, ~96GB VRAM) with PyTorch 2.x. Compatible with A100 and T4 (15GB) as well.

## Re-implementation Details

**Method:** SimpleShot shows that a nearest-centroid classifier on top of standard cross-entropy features, with simple feature transformations (mean-subtraction + L2-normalization), can match or exceed complex meta-learning methods for few-shot learning.

**Backbones:** Conv-4, ResNet-10, ResNet-18, DenseNet-121, reimplemented from scratch following the original repo's architecture specs.

**Training:** 90 epochs of cross-entropy on 64 base classes. SGD with momentum 0.9, lr 0.1 with MultiStepLR milestones [45, 66] and γ=0.1, batch size 256, weight decay 1e-4. Data augmentation: RandomResizedCrop(84) + HorizontalFlip + ImageNet normalization.

**Evaluation:** 10,000 episodes on 20 novel test classes, 5-way with 15 query images per class. Each episode samples a support set (1 or 5 shots), computes class centroids, and classifies queries by Euclidean distance. Three feature transformations tested: unnormalized (UN), L2-normalized (L2N), and centered + L2-normalized (CL2N). We report mean ± 95% confidence interval.

**Early stopping:** Every 4 epochs during training, we evaluate L2N 1-shot accuracy on the validation set over 500 episodes and keep the best checkpoint.

**Challenges:** The original repo's code fails to run in modern environments — `.view()` crashes on PyTorch 2.x, the scheduler-step order is wrong, and DataParallel causes OOM on smaller GPUs. We wrote our own training and evaluation loops rather than patching the repo's code.

## Results/Insights

All numbers are 5-way, 10,000 episodes, reported as mean ± 95% CI. "Gap" = our result − paper.

### Best Epoch (early stopping via L2N 1-shot on val)

| Model       | Transform | 1-shot Ours     | 1-shot Paper    | Gap    | 5-shot Ours     | 5-shot Paper    | Gap    |
|-------------|-----------|-----------------|-----------------|--------|-----------------|-----------------|--------|
| Conv-4      | UN        | 34.11 ± 0.17    | 33.17 ± 0.17    | +0.94  | 62.19 ± 0.17    | 63.25 ± 0.17    | -1.06  |
| Conv-4      | L2N       | 46.34 ± 0.18    | 48.08 ± 0.18    | -1.74  | 64.52 ± 0.17    | 66.49 ± 0.17    | -1.97  |
| Conv-4      | CL2N      | 47.98 ± 0.19    | 49.69 ± 0.19    | -1.71  | 65.12 ± 0.17    | 66.92 ± 0.17    | -1.80  |
| ResNet-10   | UN        | 52.01 ± 0.21    | 54.45 ± 0.21    | -2.44  | 74.39 ± 0.16    | 76.98 ± 0.15    | -2.59  |
| ResNet-10   | L2N       | 55.76 ± 0.20    | 57.85 ± 0.20    | -2.09  | 75.77 ± 0.15    | 78.73 ± 0.15    | -2.96  |
| ResNet-10   | CL2N      | 58.33 ± 0.20    | 60.85 ± 0.20    | -2.52  | 75.75 ± 0.16    | 78.40 ± 0.15    | -2.65  |
| ResNet-18   | UN        | 53.90 ± 0.21    | 56.06 ± 0.20    | -2.16  | 76.47 ± 0.15    | 78.63 ± 0.15    | -2.16  |
| ResNet-18   | L2N       | 58.05 ± 0.20    | 60.16 ± 0.20    | -2.11  | 77.59 ± 0.15    | 79.94 ± 0.14    | -2.35  |
| ResNet-18   | CL2N      | 60.34 ± 0.20    | 62.85 ± 0.20    | -2.51  | 77.61 ± 0.15    | 80.02 ± 0.14    | -2.41  |
| DenseNet-121| UN        | 55.40 ± 0.20    | 57.81 ± 0.21    | -2.41  | 78.17 ± 0.15    | 80.43 ± 0.15    | -2.26  |
| DenseNet-121| L2N       | 59.64 ± 0.20    | 61.49 ± 0.20    | -1.85  | 78.81 ± 0.14    | 81.48 ± 0.14    | -2.67  |
| DenseNet-121| CL2N      | 61.94 ± 0.20    | 64.29 ± 0.20    | -2.35  | 79.08 ± 0.15    | 81.50 ± 0.14    | -2.42  |

**Qualitative findings reproduced:**
- UN < L2N < CL2N for every model
- Conv-4 < ResNet-10 < ResNet-18 < DenseNet-121
- 1-shot < 5-shot

### Gap Analysis

Our absolute accuracy is consistently 1–3% below the paper. This matches [Issue #5](https://github.com/mileyan/simple_shot/issues/5) in the original repo, where a reproducer reported the same 2–3% drop when moving from PyTorch 1.0 to 1.4. This was confirmed by the original authors and remains unresolved. We use PyTorch 2.x, so the same effect applies. The dataset also differs slightly: the paper's raw JPEG images are no longer hosted (the original Google Drive link expired), so we use the publicly-available Kaggle pickle distribution.

## Data & Checkpoints

All pretrained checkpoints (`epoch_best.pth` and `epoch_90.pth` for every model) and the miniImageNet pickle files are hosted on Google Drive:

📂 **[All resources](https://drive.google.com/drive/folders/1juXTTaoG5aqjIFmnaDSJLed5es5qT78a?usp=drive_link)**

Dataset source: [whitemoon/miniimagenet on Kaggle](https://www.kaggle.com/datasets/whitemoon/miniimagenet).

## Conclusion

We successfully reproduced the three qualitative findings of SimpleShot on miniImageNet:

1. **UN < L2N < CL2N for every backbone:** feature transformations are essential, with centering + L2-normalization consistently best.
2. **Deeper backbones perform better:** Conv-4 < ResNet-10 < ResNet-18 < DenseNet-121.
3. **1-shot is harder than 5-shot:** nearest-centroid benefits from multiple support examples.

Our absolute accuracy is 1–3% below the paper, matching the known [Issue #5](https://github.com/mileyan/simple_shot/issues/5) where reproducers report a 2–3% drop in newer PyTorch versions, confirmed by the original authors but never resolved. This demonstrates that the paper's core conclusions are robust and reproducible, even when exact numbers shift due to framework versions.

Notable finding: our reimplementation outperforms the original repo's code by ~1% in the same environment, suggesting that our cleaner training loop avoids some of the compatibility issues in the legacy codebase.

## References

1. Wang, Y., Chao, W.-L., Weinberger, K. Q., & van der Maaten, L. (2019). SimpleShot: Revisiting Nearest-Neighbor Classification for Few-Shot Learning. *arXiv preprint arXiv:1911.04623*.
2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR*.
3. Huang, G., Liu, Z., van der Maaten, L., & Weinberger, K. Q. (2017). Densely Connected Convolutional Networks. *CVPR*.
4. Vinyals, O., et al. (2016). Matching Networks for One Shot Learning. *NeurIPS*.
5. Snell, J., Swersky, K., & Zemel, R. (2017). Prototypical Networks for Few-Shot Learning. *NeurIPS*.

## Citation

```bibtex
@article{wang2019simpleshot,
  title={SimpleShot: Revisiting Nearest-Neighbor Classification for Few-Shot Learning},
  author={Wang, Yan and Chao, Wei-Lun and Weinberger, Kilian Q. and van der Maaten, Laurens},
  journal={arXiv preprint arXiv:1911.04623},
  year={2019}
}
```

## Acknowledgements

This project was completed as part of CS 4782 (Deep Learning) at Cornell University, Spring 2026. We thank the teaching staff for their feedback and guidance during the poster session and report iterations.

We also thank the original SimpleShot authors for making their code and checkpoints publicly available, and the wider community (particularly [bertinetto](https://github.com/mileyan/simple_shot/issues/5)) for documenting reproduction challenges in the original GitHub repository.

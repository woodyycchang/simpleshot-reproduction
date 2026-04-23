# SimpleShot Reproduction

Reproduction of *SimpleShot: Revisiting Nearest-Neighbor Classification for Few-Shot Learning* (Wang et al., 2019) [[arXiv:1911.04623]](https://arxiv.org/abs/1911.04623) on the miniImageNet benchmark.

Original repo: [mileyan/simple_shot](https://github.com/mileyan/simple_shot)

## Team

Cornell University
- Woody Chang
- Jiarui Zhang
- Xinyu Tan
- Qiao Xiao

## Overview

SimpleShot shows that a nearest-centroid classifier on top of standard cross-entropy features, with simple feature transformations (mean-subtraction + L2-normalization), can match or exceed complex meta-learning methods for few-shot learning.

We reproduced the main miniImageNet results (Table 1 of the paper) on four backbone architectures:

- **Conv-4**
- **ResNet-10**
- **ResNet-18**
- **DenseNet-121**

Each model was trained from scratch for 90 epochs on the 64 base classes, then evaluated on the 20 novel test classes over 10,000 few-shot episodes (5-way, 1-shot and 5-shot).

## Results

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

## Repository Structure

- `SimpleShotReproduce_4Models_AUTO.ipynb` — Main notebook. One Run All trains and evaluates all 4 models sequentially (skips training if checkpoint exists).
- `JiaruiZhang_SimpleShotReproduce_4Models_AUTO.ipynb` — Epoch 90 evaluation variant.
- `SimpleShotReproduce_EXACT_GITHUB.ipynb` — Ran the original repo code in our environment for comparison. Our reimplementation actually beats it by ~1% in the same environment.
- `poster.pdf` — Poster session PDF.
- `report.pdf`

## Data & Checkpoints

All pretrained checkpoints (`epoch_best.pth` and `epoch_90.pth` for every model) and the miniImageNet pickle files are hosted on Google Drive:

📂 **[All resources](https://drive.google.com/drive/folders/1juXTTaoG5aqjIFmnaDSJLed5es5qT78a?usp=drive_link)**

Dataset source: [whitemoon/miniimagenet on Kaggle](https://www.kaggle.com/datasets/whitemoon/miniimagenet).

## How to Run

1. Open `SimpleShotReproduce_4Models_AUTO.ipynb` in Google Colab.
2. Mount your Drive. Place the pickle files in a folder named `SimpleShot/` inside your Drive root (or update `PKL_CANDIDATES` in Cell 2).
3. Optionally download checkpoints from the Drive link above into `checkpoints_<model>/` to skip training.
4. Run All. The notebook will train any missing models and evaluate all four.

**Environment:** Tested on Google Colab G4 GPU (NVIDIA RTX PRO 6000, ~96GB VRAM) with PyTorch 2.x. Compatible with A100 and T4 (15GB) as well.

## Citation

```bibtex
@article{wang2019simpleshot,
  title={SimpleShot: Revisiting Nearest-Neighbor Classification for Few-Shot Learning},
  author={Wang, Yan and Chao, Wei-Lun and Weinberger, Kilian Q. and van der Maaten, Laurens},
  journal={arXiv preprint arXiv:1911.04623},
  year={2019}
}
```

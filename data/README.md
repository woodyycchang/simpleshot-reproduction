# Data

This project uses the **miniImageNet** dataset for few-shot learning, 
following the Ravi & Larochelle split (64 base / 16 validation / 20 novel classes).

## Files needed

- `mini-imagenet-cache-train.pkl` (775 MB)
- `mini-imagenet-cache-val.pkl` (194 MB)
- `mini-imagenet-cache-test.pkl` (242 MB)

## Download

The pickle files are too large for GitHub and are hosted on Google Drive:

📂 **[Download miniImageNet pickle files](https://drive.google.com/drive/folders/1juXTTaoG5aqjIFmnaDSJLed5es5qT78a?usp=drive_link)**

Original source: [whitemoon/miniimagenet on Kaggle](https://www.kaggle.com/datasets/whitemoon/miniimagenet).

## Usage

After downloading, place the three `.pkl` files in a folder named `SimpleShot/` 
inside your Google Drive root. The main notebook will mount your Drive and 
automatically locate them.

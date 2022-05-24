# spacetitanic

Exploring DVC and XGBoost using the [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic/overview) Kaggle competition.

## Objectives

The goals of this repository are:

1. Practice using [DVC](https://realpython.com/python-data-version-control/) to create pipelines and version control data, models and metrics for machine learning projects.
2. Gain a practical understanding of the XGBoost algorithm.

## Installation

This project uses DVC and conda for reproducibility. If you would like to reproduce any stage of the pipeline, you can follow the instructions below:

```bash
# clone the tag/step of interest from GitHub
git clone -b xgb-tuned git@github.com:dzhang32/spacetitanic.git
cd spacetitanic

# create conda env with the necessary dependencies
conda env create -n spacetitanic --file environment.yml
conda activate spacetitanic

# install the functions from the spacetitanic repo
python3 -m pip install -e .

# download the raw data from kaggle
# as this is a personal project developed for learning
# I have not hosted the DVC remote storage in a public location (e.g. S3)
# if it were, the below could be replaced with a dvc pull
mkdir data/raw data/processed data/output model metrics
kaggle competitions download -c spaceship-titanic -p data/raw
unzip data/raw/spaceship-titanic.zip -d data/raw && rm data/raw/spaceship-titanic.zip

# rerun the entire pipeline using dvc
# evaluation metrics are stored as a .json in metrics/
# and the models are stored as .joblib files in model/
dvc repro evaluate
```

## Approach

The following steps were taken sequentially to develop the machine learning pipeline within this repository. From step 03 onwards, the implementation for each step is stored as a GitHub tag:

| Step | Description | Test accuracy |
| --- | --- | --- |
| 01 | Setup repo using git and [DVC](https://realpython.com/python-data-version-control/) | NA |
| 02 | [Feature exploration](notebooks/feature_exploration.ipynb) | NA |
| 03 | Create a minimal preprocessing and feature engineering pipeline | NA |
| 04 | Implement a [baseline random forest classifier (rf)](https://github.com/dzhang32/spacetitanic/releases/tag/rf-mvp) using a DVC pipeline | 0.794 |
| 05 | Tune the rf hyperparameters using a [random grid search](https://github.com/dzhang32/spacetitanic/releases/tag/rf-random_grid) | 0.796 |
| 06 | Implement a [baseline XGBoost classifier (xgb)](https://github.com/dzhang32/spacetitanic/releases/tag/xgb-mvp) | 0.798 |
| 07 | Use the [full training dataset](https://github.com/dzhang32/spacetitanic/releases/tag/xgb-mvp_full_train) (rather than training minus validation) for training the xgb* | 0.800 |
| 08 | Use [one-hot encoding](https://github.com/dzhang32/spacetitanic/releases/tag/xgb-mvp_onehot) for the categorical features | 0.797 |
| 09 | Set [early stopping rounds](https://github.com/dzhang32/spacetitanic/releases/tag/xgb-mvp_early_stop) to avoid overfitting | 0.797 |
| 10 | [Feature selection](https://github.com/dzhang32/spacetitanic/releases/tag/xgb-mvp_feat_selection) | 0.802 |
| 11 | [Tune the xgb hyperparameters](https://github.com/dzhang32/spacetitanic/releases/tag/xgb-tuned) | 0.808 |

*This step is likely Kaggle-specific, as the labels for the test data are hidden in competitons.

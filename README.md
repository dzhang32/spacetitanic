# Space Titanic

Practicing DVC and using XGBoost with the [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic/overview) Kaggle competition.

## Objectives

The aim of this repository is to:

1. Practice using DVC to version control data.
2. Practice using DVC to create a pipeline and track metrics for each version of the model.
3. Gain a deeper intuition and practical understanding of XGboost algorithms.

## Installation

```bash
conda create
```

## Approach

The following steps were taken sequentially to develop the pipeline within this repository:

| Step | Description |
| --- | --- |
| 01 | Setup repo using git and [DVC](https://realpython.com/python-data-version-control/) |
| 02 | [Feature exploration](notebooks/feature_exploration.ipynb) |
| 03 | Create a [minimal preprocessing and feature engineering pipeline](src/spacetitanic/features/) |
| 04 | Implement a [baseline random forest (rf) classifier](https://github.com/dzhang32/spacetitanic/releases/tag/rf-mvp) |
| 05 | Tune the rf using a [random grid](https://github.com/dzhang32/spacetitanic/releases/tag/rf-random_grid) |
| 06 | Implement a baseline XGboost classifier |
| 07 | Use entire train dataset (rather than train minus valisation) for training* |
| 08 | One-hot encode the categorical features |
| 09 | Feature selection |
| 10 | Hyper parameter tune the XGboost classifier |

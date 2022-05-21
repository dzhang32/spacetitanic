# Space Titanic

Practicing DVC and using XGBoost with the [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic/overview) Kaggle competition.

## Objectives

The aim of this repository is to:

1. Have some fun!
2. Practice using DVC to version control data.
3. Practice using DVC to create a pipeline and track metrics for each version of the model.
4. Gain a deeper intuition and practical understanding of XGboost algorithms.

## Installation

```bash
conda create
```

## Approach

The following steps were taken sequentially to develop the pipeline within this repository:

| N. | Description |
| --- | --- |
| 01 | Setup repo using git and [DVC](https://realpython.com/python-data-version-control/) |
| 02 | [Feature exploration](notebooks/feature_exploration.ipynb) |
| 03 | Create a minimal preprocessing and feature engineering pipeline |
| 04 | Create a MVP model (random forest), decide on the evaluation metrics |
| 05 | Hyper parameter tune the random forest |
| 06 | Implement XGboost algorithm |
| 07 | Feature selection |
| 08 | Hyper parameter tune the XGboost algorithm |

# BitePoint — Yelp Restaurant Recommender

CS179: Introduction to Graphical Models — Final Project

## Overview

Two collaborative filtering models trained on the Yelp Open Dataset to predict user restaurant ratings:

- **Matrix Factorization** — SGD baseline with user/business bias terms
- **Bayesian Latent Factor Model** — mean-field variational inference with Gaussian priors, trained by maximizing the ELBO

## Structure

```
BitePoint/
├── src/
│   ├── config.py        # hyperparameters and paths
│   ├── data_loader.py   # load and preprocess Yelp data
│   ├── model.py         # MF and Bayesian VI models
│   ├── evaluation.py    # MAE, RMSE, correlation, accuracy
│   └── main.py          # train and evaluate
├── data/
│   ├── raw/             # Yelp JSON files
│   └── processed/       # train.csv, test.csv, mappings.pkl
├── results/
│   ├── models/          # saved results JSON
│   └── figures/         # evaluation plots
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

Place Yelp dataset JSON files (`review.json`, `user.json`, `business.json`) in `data/raw/`.

## Running

Set `MODEL = 'mf'` or `MODEL = 'bayesian'` at the top of `src/main.py`, then:

```bash
python src/main.py
```

Processed data is cached to `data/processed/` on first run.

## Team

- Anthony Suh
- Alex Yin
- Angelina Wang

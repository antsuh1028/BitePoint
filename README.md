# Yelp Restaurant Recommender using Probabilistic Graphical Models

## Project Overview
This project implements a probabilistic graphical model-based restaurant recommendation system using Yelp data. The model jointly models user preferences, restaurant features, and ratings to make personalized recommendations.

## Project Structure
```
BitePoint/
├── src/
│   ├── __init__.py
│   ├── config.py           # Configuration parameters
│   ├── data_loader.py      # Load and preprocess Yelp data
│   ├── model.py            # Probabilistic graphical model implementation
│   ├── evaluation.py       # Metrics and evaluation functions
│   └── main.py             # Entry point for training and inference
├── data/
│   ├── raw/                # Original Yelp data
│   └── processed/          # Processed data for the model
├── notebooks/
│   └── exploration.ipynb   # EDA and data exploration
├── results/
│   ├── models/             # Trained model files
│   ├── predictions/        # Model predictions
│   └── plots/              # Evaluation plots
├── report/
│   └── project_report.pdf  # Final 4-page report
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Key Components

### 1. Data (Yelp Dataset)
- User data (user IDs, review counts, ratings)
- Business/Restaurant data (categories, ratings, location)
- Review data (ratings, text, helpful votes)
- Consider downloading from: https://www.yelp.com/dataset

### 2. Probabilistic Graphical Model
Potential approaches:
- **Latent Factor Model**: Model user preferences and restaurant attributes as latent variables
- **Bayesian Network**: Model dependencies between user type, restaurant features, and ratings
- **Markov Random Field**: Model interactions between users and restaurants
- **Topic Model**: Extract review topics and model how they influence ratings

### 3. Evaluation Metrics
- **Prediction Accuracy**: MAE/RMSE on held-out ratings
- **Ranking Metrics**: NDCG, MRR, Precision@K
- **Coverage**: Percentage of restaurants/users covered
- **Performance vs. Data Size**: How accuracy improves with training data
- **Performance vs. Model Complexity**: Trade-offs in model parameters

## Getting Started

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Data Preparation
1. Download Yelp dataset from https://www.yelp.com/dataset
2. Place in `data/raw/`
3. Run preprocessing: `python src/data_loader.py`

### Run the Project
```bash
python src/main.py --model [model_type] --epochs 100 --batch_size 32
```

### Generate Report
See `notebooks/exploration.ipynb` and results in `results/` for evaluation plots.

## Team Members
- [Member 1]
- [Member 2]
- [Member 3]

## References
- Yelp Dataset: https://www.yelp.com/dataset
- Pyro: https://pyro.ai/
- Stan: https://mc-stan.org/
- Relevant papers: [Add your references]

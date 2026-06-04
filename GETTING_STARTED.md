# Getting Started with BitePoint

## Quick Start Guide

### 1. Setup Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Yelp Dataset

1. Visit https://www.yelp.com/dataset
2. Download the dataset (requires academic verification)
3. Extract the JSON files to `data/raw/`:
   - `review.json`
   - `user.json`
   - `business.json`
   - (optional: `checkin.json`, `photo.json`, `tip.json`)

### 3. Preprocess Data

```bash
cd src
python data_loader.py
```

This will:
- Load raw Yelp JSON files
- Filter by minimum reviews
- Create train/test split
- Save processed data to `data/processed/`

### 4. Run the Model

#### Option A: Quick Experimentation (Jupyter Notebook)
```bash
jupyter notebook notebooks/exploration.ipynb
```

Run cells sequentially to:
- Load and explore data
- Train matrix factorization model
- Evaluate performance
- Visualize results

#### Option B: Command Line
```bash
cd src
python main.py --model mf --epochs 100 --latent_dim 20
```

**Available arguments:**
- `--model`: `mf` (matrix factorization) or `bayesian` (full Bayesian)
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size (default: 32)
- `--latent_dim`: Latent factor dimension (default: 20)
- `--learning_rate`: Learning rate (default: 0.001)
- `--regularization`: L2 regularization (default: 0.01)

### 5. Generate Report

Results will be saved to `results/`:
- `models/mf_results.json` - Performance metrics
- Plots and visualizations

Use these in your 4-page project report.

## Project Structure

```
BitePoint/
├── src/
│   ├── config.py              # Configuration parameters
│   ├── data_loader.py         # Data loading & preprocessing
│   ├── model.py               # Model implementations
│   ├── evaluation.py          # Evaluation metrics
│   └── main.py                # Entry point
├── data/
│   ├── raw/                   # Original Yelp JSON files (download here)
│   └── processed/             # Processed data (created by data_loader.py)
├── notebooks/
│   └── exploration.ipynb      # Interactive development notebook
├── results/
│   ├── models/                # Trained model files & results
│   ├── predictions/           # Model predictions
│   └── plots/                 # Evaluation plots
├── report/
│   └── project_report.pdf     # Final 4-page report (to be written)
├── requirements.txt           # Python dependencies
├── README.md                  # Project overview
└── GETTING_STARTED.md         # This file
```

## Model Overview

### Latent Factor Model
- **Users** have latent preference vectors
- **Restaurants** have latent feature vectors
- **Ratings** predicted as dot product of these vectors
- **Probabilistic** approach with explicit uncertainty

### Key Features
✓ Collaborative filtering without explicit item features  
✓ Scalable to large datasets  
✓ Interpretable learned representations  
✓ Extends to Bayesian inference with Pyro  

## Tips for Success

1. **Start small**: Run on a subset of data first to debug
2. **Experiment**: Try different latent dimensions
3. **Ablation study**: Test impact of model changes
4. **Metrics matter**: Measure multiple metrics (MAE, RMSE, ranking)
5. **Document**: Save results and plots for your report

## Troubleshooting

### Data loading issues
- Ensure Yelp JSON files are in `data/raw/`
- Check that JSON files are not corrupted
- Verify required columns: `user_id`, `business_id`, `stars`

### Memory issues
- Reduce `max_users` or `max_businesses` in `config.py`
- Use smaller `batch_size`
- Try reducing `latent_dim`

### Slow training
- Check that GPU is available (if using PyTorch)
- Reduce number of epochs for testing
- Use smaller dataset for quick iterations

## Resources

- **Yelp Dataset**: https://www.yelp.com/dataset
- **Pyro (Probabilistic Programming)**: https://pyro.ai/
- **Collaborative Filtering**: https://en.wikipedia.org/wiki/Collaborative_filtering
- **Latent Factor Models**: https://www.youtube.com/results?search_query=latent+factor+models

## Questions?

Refer to:
- `README.md` for project overview
- `config.py` for configuration options
- `notebooks/exploration.ipynb` for interactive examples
- Code comments in `src/` files

Good luck with your project! 🚀

# Data Breaches Analysis Project

## Overview
This project analyzes data breaches to understand their causes, impacts, and trends. The goal is to identify patterns in security incidents and provide insights for improving organizational cybersecurity practices.

## Objectives
- Examine historical data breach incidents
- Identify common vulnerabilities and attack vectors
- Analyze the financial and reputational impacts of breaches
- Explore trends in data breach frequency and severity

## Project Structure
```
BA798_WEEK1_PRACTICE/
├── README.md
├── data/
│   ├── raw/                    # Raw dataset
│   ├── processed/              # Train/test splits
│   └── predictions/            # Model predictions
├── src/                        # ML pipeline modules
│   ├── config.py              # Configuration & paths
│   ├── data_loader.py         # Data loading utilities
│   ├── preprocessor.py        # Feature engineering
│   ├── models.py              # RF & XGBoost models
│   ├── trainer.py             # Model training
│   ├── evaluator.py           # Evaluation & visualization
│   └── utils.py               # Helper utilities
├── notebooks/                  # Jupyter notebooks
│   └── 05_main_pipeline.ipynb # Complete ML pipeline
├── outputs/                    # Model artifacts & results
│   ├── models/                # Serialized models
│   ├── metrics/               # JSON metrics & reports
│   └── figures/               # Visualizations
├── docs/                       # Documentation
│   ├── PIPELINE_DOCUMENTATION.md
│   └── IMPLEMENTATION_LOG.md
├── requirements.txt            # Python dependencies
└── .gitignore
```

## Data Sources
- Publicly available breach databases
- Industry security reports
- Regulatory disclosure filings

## Technologies Used
- Python
- Pandas, NumPy, Scikit-learn
- XGBoost
- Matplotlib, Seaborn
- Jupyter Notebooks

## ML Pipeline

A modular machine learning pipeline that predicts regulatory action on data breaches.

### Target Variable
**`has_any_regulatory_action`** - Binary classification predicting whether a breach will result in regulatory action from FTC, FCC, or State Attorney Generals.

### Models
- **Random Forest** - 100 estimators, balanced class weights
- **XGBoost** - 100 estimators, learning_rate=0.1

### Pipeline Components
1. **Data Loading** - Stratified train/test split (80/20)
2. **Preprocessing** - Missing value handling, feature encoding, feature scaling
3. **Feature Engineering** - breach_intensity, regulatory_risk_score, attack_surface
4. **Model Training** - Train both RF and XGBoost in parallel
5. **Evaluation** - Metrics (accuracy, precision, recall, F1, ROC-AUC) and visualizations

### Quick Start
```bash
cd BA798_WEEK1_PRACTICE
jupyter notebook notebooks/05_main_pipeline.ipynb
```

See `docs/PIPELINE_DOCUMENTATION.md` for detailed information.

## Getting Started
1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the analysis notebooks in the `notebooks/` folder

## Author
BA798 - Week 1 Practice Project

## License
This project is for educational purposes only.
# Phishing URL Detection

Ensemble classification model to detect phishing URLs using text-based feature extraction. Built for the **Data Science Competition (DSCO)** hosted by BINUS University.

## Problem

Given a dataset of URLs, classify each as either legitimate or phishing. The challenge is to detect malicious URLs using only the URL string itself -- no external lookups or page content analysis.

## Approach

1. **Feature Extraction** -- Applied TF-IDF vectorization on raw URL strings to capture character/token patterns that distinguish phishing from legitimate URLs.

2. **Model Training** -- Trained three gradient boosting models independently:
   - **LightGBM (DART)** -- Dropout-based boosting for regularization
   - **XGBoost** -- Gradient boosted trees with tuned hyperparameters
   - **CatBoost** -- Handles categorical features natively with ordered boosting

3. **Ensemble** -- Combined predictions using a weighted average across all three models to reduce variance and improve generalization.

4. **Pseudo-Labeling** -- Applied semi-supervised learning by using high-confidence predictions on the test set as additional training data, then retrained the ensemble for a final performance boost.

## Tech Stack

- Python
- Scikit-learn (TF-IDF, preprocessing, metrics)
- LightGBM, XGBoost, CatBoost
- Pandas, NumPy

## Project Structure

```
lgbm_xgb_catboost_model.py      # LightGBM DART + XGBoost + CatBoost training pipelines
ensemble_pseudo_labeling.py      # Weighted ensemble + pseudo-labeling
README.md
```

## Results

Achieved competitive ranking on the DSCO leaderboard using ensemble methods combined with semi-supervised learning. The pseudo-labeling step provided a measurable improvement over the base ensemble.

## What I Learned

- TF-IDF on raw URLs is surprisingly effective -- character-level n-grams capture domain patterns, suspicious tokens, and URL structure anomalies without any manual feature engineering.
- Ensemble diversity matters more than individual model tuning. LightGBM, XGBoost, and CatBoost each have different inductive biases, which is exactly what makes the combination strong.
- Pseudo-labeling works well when the base model is already confident on a large portion of test data. Threshold selection is critical -- too low and you inject noise.

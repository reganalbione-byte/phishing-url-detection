# Phishing URL Detection — DSCO Competition (BINUS)

Binary classification model to detect phishing URLs using text-based features. Built for the Data Science Competition (DSCO) hosted by BINUS University.

## Approach

- Extracted features from raw URLs using TF-IDF vectorization
- - Trained multiple models: LightGBM (DART), XGBoost, CatBoost
  - - Built a weighted ensemble for final predictions
    - - Applied pseudo-labeling on test data to boost performance
     
      - ## Tech Stack
     
      - - Python, Scikit-learn, LightGBM, XGBoost, CatBoost
        - - Pandas, NumPy
          - - TF-IDF for text feature extraction
           
            - ## Results
           
            - Achieved competitive ranking in the DSCO leaderboard using ensemble methods and semi-supervised learning techniques.

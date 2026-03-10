# ML Intrusion Detection on NSL-KDD

## Description
This project implements machine learning models to detect network intrusions using the `NSL-KDD` dataset. The goal is to classify network connections as either **normal** or **attack**. Multiple models, including Logistic Regression and Random Forest, are trained and evaluated with preprocessing, feature scaling, and threshold tuning.

---

## Dataset
- **NSL-KDD**: a dataset designed for network intrusion detection benchmarking.
- Features: 41 numeric/categorical features representing network connection properties.
- Target: simplified to binary labels:
  - `0` = normal connection
  - `1` = attack (all attack types combined)
- Data preprocessing includes one-hot encoding of categorical features, log transformation of skewed numeric features, and standard scaling.

---

## Project Structure
```plaintext
ML-Intrusion-Detection
├── data/
|   ├── KDDTest+.txt                # Test dataset
│   └── KDDTrain+.txt               # Train dataset
├── models/
│   ├── logistic_regression.joblib  # Logistic Regression model
│   └── random_forest.joblib        # Random Forest model 
├── notebooks/
│   ├── analysis.ipynb              # Model training and evaluation notebook
│   └── eda.ipynb                   # Exploratory Data Analysis on the NSL-KDD dataset
├── src/
│   ├── __init__.py                 # included so that src can be seen as a module
│   └── preprocess.py               # functions to use in jupyter notebooks
├── requirements.txt                # python dependencies    
└── README.md                       # Project documentation
```

---

## Preprocessing
- Drop unnecessary columns (`difficulty`)  
- Encode categorical features (`protocol_type`, `service`, `flag`) with one-hot encoding  
- Separate features and target (`is_attack`)  
- Log-transform highly skewed numeric features (such as `duration`)
- Standardize numeric features with `StandardScaler`  
- Align train and test features to ensure consistent columns  

---

## Models
1. **Logistic Regression**  
   - `max_iter=1000`
   - `GridSearch` for hyperparameter `C`
   - Threshold tuning for improved attack recall  

2. **Random Forest**  
   - `n_estimators=400`  
   - `max_features=sqrt`  
   - Threshold tuning to balance precision and recall  

---

## Evaluation Metrics
- **Precision**: proportion of predicted attacks that are correct  
- **Recall**: proportion of true attacks correctly detected  
- **F1-score**: harmonic mean of precision and recall  
- **Confusion matrix**: visual representation of true vs predicted labels  

Example results (Random Forest, threshold = 0.3):

| Class   | Precision | Recall | F1-score |
|---------|-----------|--------|----------|
| Normal  | 0.72      | 0.97   | 0.83     |
| Attack  | 0.97      | 0.71   | 0.82     |
| **Accuracy** |           |        | 0.82     |

---

## How to Run
1. Clone the repository:

```bash
git clone https://github.com/antonio-ciocodeica/ML-Intrusion-Detection.git
```

2. Install the dependencies:
```bash
pip install -r requirements.txt
```

3. Open the notebooks and run the cells

---

## Conclusions
- Logistic Regression performs moderately well but struggles with minority attack types.
- Random Forest improves recall for common attacks but still misses some rare attacks.
- Threshold tuning is important to balance attack recall and precision.
- Future work: multi-class classifiers to identify specific attack types, and additional feature engineering.
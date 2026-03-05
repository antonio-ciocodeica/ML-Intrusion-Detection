import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a model and return the classification report and confusion matrix
    """
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    conf_mat = confusion_matrix(y_test, y_pred)

    return report, conf_mat

def train(X_train, y_train, save=False):
    """
    Train and (optionally) save a Logistic Regression and a Random Forest model and return them
    """
    # 1. Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)

    # 2. Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train, y_train)

    if save:
        joblib.dump(lr, '../models/logistic_regression.joblib')
        joblib.dump(lr, '../models/random_forest.joblib')

    return lr, rf
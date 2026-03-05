import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from preprocess import load_data, train_test_data

def evaluate_model(model, model_name, X_test, y_test):
    """
    Evaluate a model and generate the classification report and confusion matrix
    """
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    conf_mat = confusion_matrix(y_test, y_pred)

    print(f'Classification report for {model_name}:')
    print(report)

    sns.heatmap(conf_mat, cmap='Blues')
    plt.title(f'{model_name} Confusion matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def main():
    # Load, preprocess and split the dataset
    df_train, df_test = load_data('../data/KDDTrain+.txt', '../data/KDDTest+.txt')
    X_train, X_test, y_train, y_test = train_test_data(df_train, df_test)

    # 1. Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    evaluate_model(lr, 'Logistic Regression', X_test, y_test)

    # 2. Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train, y_train)
    evaluate_model(rf, 'Random Forest Classifier', X_test, y_test)

if __name__ == '__main__':
    main()
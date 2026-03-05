import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(train_path, test_path):
    columns = [
        "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
        "wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised",
        "root_shell","su_attempted","num_root","num_file_creations","num_shells",
        "num_access_files","num_outbound_cmds","is_host_login","is_guest_login",
        "count","srv_count","serror_rate","srv_serror_rate","rerror_rate",
        "srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate",
        "dst_host_count","dst_host_srv_count","dst_host_same_srv_rate",
        "dst_host_diff_srv_rate","dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate","dst_host_serror_rate",
        "dst_host_srv_serror_rate","dst_host_rerror_rate",
        "dst_host_srv_rerror_rate","attack","difficulty"
    ]
     
    df_train = pd.read_csv(train_path, header=None, names=columns)
    df_test = pd.read_csv(test_path, header=None, names=columns)

    return df_train, df_test

def preprocess(df):
    # Drop unnecessary columns
    df = df.drop('difficulty', axis=1)

    # Simplify target to binary: normal(0) / attack(1)
    df['is_attack'] = df['attack'].apply(lambda x: 0 if x == 'normal' else 1)
    df = df.drop('attack', axis=1)

    # Encode categorical features
    cat_cols = ['protocol_type', 'service', 'flag']
    df = pd.get_dummies(df, columns=cat_cols)

    # Scale numeric features
    scaler = StandardScaler()
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df

def split_X_y(df):
    """
    Split the data into features (X) and target (y)
    """
    X = df.drop('is_attack', axis=1)
    y = df['is_attack']
    return X, y

def train_test_data(df_train, df_test):
    """
    Preprocess the train/test data and split it into X_train/test and y_train/test 
    """
    df_train = preprocess(df_train)
    df_test = preprocess(df_test)

    X_train, y_train = split_X_y(df_train)
    X_test, y_test = split_X_y(df_test)

    return X_train, X_test, y_train, y_test
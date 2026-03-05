import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(train_path, test_path):
    """
    Load data from the train/test data files and adds the column names
    """
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
    """
    Preprocess the data by:
    - dropping unnecessary columns
    - simplyfing the target column to binary
    - encoding categorical features
    - separate the target from the features
    - scaling the numeric features
    """
    df = df.drop('difficulty', axis=1)

    df['is_attack'] = df['attack'].apply(lambda x: 0 if x == 'normal' else 1)
    df = df.drop('attack', axis=1)

    cat_cols = ['protocol_type', 'service', 'flag']
    df = pd.get_dummies(df, columns=cat_cols)

    X = df.drop('is_attack', axis=1)
    y = df['is_attack']

    scaler = StandardScaler()
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    X[num_cols] = scaler.fit_transform(X[num_cols])

    return X, y

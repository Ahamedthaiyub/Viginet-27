
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from keras.utils import to_categorical
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.utils import to_categorical
import pandas as pd

def process_dataframe(df, encoder):
   
    expected_columns = ["duration", "protocol_type", "service", "flag", "src_bytes",
                         "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
                         "num_failed_logins", "logged_in", "num_compromised",
                         "root_shell", "su_attempted", "num_root",
                         "num_file_creations", "num_shells", "num_access_files",
                         "num_outbound_cmds", "is_host_login", "is_guest_login",
                         "count", "srv_count", "serror_rate", "srv_serror_rate",
                         "rerror_rate", "srv_rerror_rate", "same_srv_rate",
                         "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
                         "dst_host_srv_count", "dst_host_same_srv_rate",
                         "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                         "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
                         "dst_host_srv_serror_rate", "dst_host_rerror_rate",
                         "dst_host_srv_rerror_rate", "label"]

  
    columns_to_use = []
    for col in expected_columns:
        if col in df.columns:
            columns_to_use.append(col)
        else:
            print(f"Column '{col}' not found in the dataset. Skipping...")

    if not columns_to_use:
        raise ValueError("None of the expected columns found in the dataset.")


    df = df[columns_to_use]

    
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    non_numeric_columns = df.select_dtypes(exclude=['number']).columns.tolist()

    if not numeric_columns:
        raise ValueError("No numeric columns found for scaling.")

   
    scaler = StandardScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

   
    if encoder == 'labelencoder':
        label_encoders = {}
        for col in non_numeric_columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
    elif encoder == 'ordinalencoder':
        ordinal_encoder = pd.get_dummies(df[non_numeric_columns], drop_first=True)
        df = pd.concat([df.drop(columns=non_numeric_columns), ordinal_encoder], axis=1)

    return df
def kdd_encoding(params):
    data_path = "D:/VIGINET360/RNN_Intrusion-Detection_Keras-master/src/"

    if params['train_data'] == 494021:
        train_data_path = data_path + "kddcup.data_10_percent.csv"
        test_data_path = data_path + "kddcup.data_10_percent.csv"
    elif params['train_data'] == 4898431:
        train_data_path = data_path + "kddcup_traindata.csv"
        test_data_path = data_path + "kddcup_testdata_corrected.csv"
    else:
        if params['train_data'] == 125973:
            train_data_path = data_path + "KDDTrain+.csv"
        elif params['train_data'] == 25191:
            train_data_path = data_path + "KDDTrain+_20Percent.csv"
        test_data_path = data_path + "KDDTest+.csv"

    full_features = ["duration", "protocol_type", "service", "flag", "src_bytes",
                     "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
                     "num_failed_logins", "logged_in", "num_compromised",
                     "root_shell", "su_attempted", "num_root",
                     "num_file_creations", "num_shells", "num_access_files",
                     "num_outbound_cmds", "is_host_login", "is_guest_login",
                     "count", "srv_count", "serror_rate", "srv_serror_rate",
                     "rerror_rate", "srv_rerror_rate", "same_srv_rate",
                     "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
                     "dst_host_srv_count", "dst_host_same_srv_rate",
                     "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                     "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
                     "dst_host_srv_serror_rate", "dst_host_rerror_rate",
                     "dst_host_srv_rerror_rate", "label"]

    train_df = pd.read_csv(train_data_path, names=full_features)
    test_df = pd.read_csv(test_data_path, names=full_features)

    x_train, y_train = process_dataframe(train_df, encoder=params['encoder'])
    x_test, y_test = process_dataframe(test_df, encoder=params['encoder'])

    return x_train, x_test, y_train, y_test



import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.utils import to_categorical

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.utils import to_categorical

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.utils import to_categorical

def kdd_encoding(params):
    data_path = "D:/VIGINET360/RNN_Intrusion-Detection_Keras-master/src/"

    if params['train_data'] == 494021:
        train_data_path = data_path + "kddcup.data_10_percent.csv"
        test_data_path = data_path + "kddcup.data_10_percent.csv"
    elif params['train_data'] == 4898431:
        train_data_path = data_path + "kddcup_traindata.csv"
        test_data_path = data_path + "kddcup_testdata_corrected.csv"
    else:
        if params['train_data'] == 125973:
            train_data_path = data_path + "KDDTrain+.csv"
        elif params['train_data'] == 25191:
            train_data_path = data_path + "KDDTrain+_20Percent.csv"
        test_data_path = data_path + "KDDTest+.csv"

    full_features = [
        "duration", "protocol_type", "service", "flag", "src_bytes","dst_bytes", "land", "wrong_fragment",
        "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted",
        "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
        "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
        "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
        "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
        "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"
    ]

    train_df = pd.read_csv(train_data_path, names=full_features)
    test_df = pd.read_csv(test_data_path, names=full_features)

  
    convert_data_types = {
        "src_bytes": int,
        "dst_bytes": int,
        
    }

    for col, dtype in convert_data_types.items():
        train_df[col] = train_df[col].apply(lambda x: int(x) if str(x).isdigit() else None).fillna(0).astype(dtype)
        test_df[col] = test_df[col].apply(lambda x: int(x) if str(x).isdigit() else None).fillna(0).astype(dtype)

    x_train, y_train = process_dataframe(train_df, encoder=params['encoder'])
    x_test, y_test = process_dataframe(test_df, encoder=params['encoder'])

    return x_train, x_test, y_train, y_test

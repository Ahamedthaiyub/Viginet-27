from sklearn.preprocessing import (StandardScaler, OrdinalEncoder,
                                   LabelEncoder, MinMaxScaler, OneHotEncoder)
from keras.utils import to_categorical
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  


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


four_features = ['service', 'src_bytes', 'dst_host_diff_srv_rate',
                 'dst_host_rerror_rate', 'label']


eight_features = ['service', 'src_bytes', 'dst_host_diff_srv_rate',
                  'dst_host_rerror_rate', 'dst_bytes', 'hot',
                  'num_failed_logins', 'dst_host_srv_count', 'label']


entry_type = {'normal': 'normal',
              'probe': ['ipsweep.', 'nmap.', 'portsweep.',
                        'satan.', 'saint.', 'mscan.'],
              'dos': ['back.', 'land.', 'neptune.', 'pod.', 'smurf.',
                      'teardrop.', 'apache2.', 'udpstorm.', 'processtable.',
                      'mailbomb.'],
              'u2r': ['buffer_overflow.', 'loadmodule.', 'perl.', 'rootkit.',
                      'xterm.', 'ps.', 'sqlattack.'],
              'r2l': ['ftp_write.', 'guess_passwd.', 'imap.', 'multihop.',
                      'phf.', 'spy.', 'warezclient.', 'warezmaster.',
                      'snmpgetattack.', 'named.', 'xlock.', 'xsnoop.',
                      'sendmail.', 'httptunnel.', 'worm.', 'snmpguess.']}


service_values = ['http', 'smtp', 'finger', 'domain_u', 'auth', 'telnet',
                  'ftp', 'eco_i', 'ntp_u', 'ecr_i', 'other', 'private',
                  'pop_3', 'ftp_data', 'rje', 'time', 'mtp', 'link',
                  'remote_job', 'gopher', 'ssh', 'name', 'whois', 'domain',
                  'login', 'imap4', 'daytime', 'ctf', 'nntp', 'shell', 'IRC',
                  'nnsp', 'http_443', 'exec', 'printer', 'efs', 'courier',
                  'uucp', 'klogin', 'kshell', 'echo', 'discard', 'systat',
                  'supdup', 'iso_tsap', 'hostnames', 'csnet_ns', 'pop_2',
                  'sunrpc', 'uucp_path', 'netbios_ns', 'netbios_ssn',
                  'netbios_dgm', 'sql_net', 'vmnet', 'bgp', 'Z39_50', 'ldap',
                  'netstat', 'urh_i', 'X11', 'urp_i', 'pm_dump', 'tftp_u',
                  'tim_i', 'red_i', 'icmp', 'http_2784', 'harvest', 'aol',
                  'http_8001']

flag_values = ['OTH', 'RSTOS0', 'SF', 'SH',
               'RSTO', 'S2', 'S1', 'REJ', 'S3', 'RSTR', 'S0']

protocol_type_values = ['tcp', 'udp', 'icmp']


def kdd_encoding(params):
   
    data_path = "./data/"
   
    if params['train_data'] == 494021:
        train_data_path = data_path+"kddcup_traindata_10_percent.csv"
        test_data_path = data_path + "kddcup_testdata_corrected.csv"

    elif params['train_data'] == 4898431:
        train_data_path = data_path+"kddcup_traindata.csv"
        test_data_path = data_path+"kddcup_testdata_corrected.csv"
    else:
      
        if params['train_data'] == 125973:
            train_data_path = data_path+"KDDTrain+.csv"
      
        elif params['train_data'] == 25191:
            train_data_path = data_path+"KDDTrain+_20Percent.csv"
        test_data_path = data_path+"KDDTest+.csv"
       
        full_features.append("difficulty")

    
    train_df = pd.read_csv(train_data_path, names=full_features)
    test_df = pd.read_csv(test_data_path, names=full_features)

    def process_dataframe(df):
      
        if params['features_nb'] == 4:
            features = four_features
        elif params['features_nb'] == 8:
            features = eight_features
        else:
            features = full_features

        df = df[features]

    
        df['label'] = df['label'].replace(['normal.', 'normal'], 0)
        for i in range(len(entry_type['probe'])):
            df['label'] = df['label'].replace(
                [entry_type['probe'][i], entry_type['probe'][i][:-1]], 1)
        for i in range(len(entry_type['dos'])):
            df['label'] = df['label'].replace(
                [entry_type['dos'][i], entry_type['dos'][i][:-1]], 2)
        for i in range(len(entry_type['u2r'])):
            df['label'] = df['label'].replace(
                [entry_type['u2r'][i], entry_type['u2r'][i][:-1]], 3)
        for i in range(len(entry_type['r2l'])):
            df['label'] = df['label'].replace(
                [entry_type['r2l'][i], entry_type['r2l'][i][:-1]], 4)


        if "difficulty" in df.columns:
            df = df.drop(columns='difficulty')

       
        y = df['label']
        x = df.drop(columns='label')

        
        if params['encoder'] == 'ordinalencoder':
            x = OrdinalEncoder().fit_transform(x)
      
        elif params['encoder'] == 'labelencoder':
            x = x.apply(LabelEncoder().fit_transform)
        else:
          
            if 'service' in features:
                for i in range(len(service_values)):
                    x['service'] = x['service'].replace(service_values[i], i)

            if 'protocol_type' in features:
                for i in range(len(protocol_type_values)):
                    x['protocol_type'] = x['protocol_type'].replace(
                        protocol_type_values[i], i)

            if 'flag' in features:
                for i in range(len(flag_values)):
                    x['flag'] = x['flag'].replace(flag_values[i], i)

        
            if params['encoder'] == "standardscaler":
                x = StandardScaler().fit_transform(x)
            
            elif params['encoder'] == "minmaxscaler01":
                x = MinMaxScaler(feature_range=(0, 1)).fit_transform(x)
         
            elif params['encoder'] == "minmaxscaler11":
                x = MinMaxScaler(feature_range=(-1, 1)).fit_transform(x)
        return x, y

    x_train, Y_train = process_dataframe(train_df)
    x_test, Y_test = process_dataframe(test_df)


    y_train = to_categorical(Y_train)
    y_test = to_categorical(Y_test)

    return x_train, x_test, y_train, y_test

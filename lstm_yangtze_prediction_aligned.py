# 为特征数据创建scaler
feature_scaler = MinMaxScaler()
feature_cols = ['yangtze_level', 'tide_lag_4h', 'tide_lag_5h', 'tide_lag_6h', 
                'pumping_flow', 'sluice_flow'] 
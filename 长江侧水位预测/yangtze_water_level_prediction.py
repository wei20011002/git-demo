import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

DATA_YEARS = [2022, 2023, 2024]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIRS = [os.path.join(BASE_DIR, '..', str(year)) for year in DATA_YEARS]
TIDE_FILES = [os.path.join(BASE_DIR, '..', '潮位数据处理', f'{year}年潮位数据_真正小时级.csv') for year in DATA_YEARS]
PUMP_FILES = [os.path.join(BASE_DIR, '..', 'output_images', f'{year}年抽水站流量数据.csv') for year in DATA_YEARS]
SLUICE_FILES = [os.path.join(BASE_DIR, '..', 'output_images', f'{year}年节制闸流量数据.csv') for year in DATA_YEARS]

# 数据加载

def load_water_level():
    dfs = []
    for year, data_dir in zip(DATA_YEARS, DATA_DIRS):
        for month in range(1, 13):
            if year == 2024:
                fname = f'水位{year}-{month}-1.csv'
            else:
                fname = f'{year}-{month}-1.csv'
            fpath = os.path.join(data_dir, fname)
            if not os.path.exists(fpath):
                continue
            try:
                df = pd.read_csv(fpath, encoding='gbk')
            except:
                df = pd.read_csv(fpath, encoding='utf-8')
            df = df[df['名称'] == '长江侧水位'].copy()
            df['数值'] = pd.to_numeric(df['数值'], errors='coerce')
            df = df.dropna(subset=['数值'])
            df['datetime'] = pd.to_datetime(df['日期'].astype(str) + ' ' + df['时间'].astype(str), errors='coerce')
            df = df[['datetime', '数值']].rename(columns={'数值': 'yangtze_level'})
            dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)
    all_df = all_df.drop_duplicates('datetime').set_index('datetime').sort_index()
    return all_df

def load_tide():
    dfs = []
    for f in TIDE_FILES:
        if not os.path.exists(f):
            continue
        df = pd.read_csv(f)
        if 'datetime' not in df.columns:
            df['datetime'] = pd.to_datetime(df.iloc[:,0], errors='coerce')
        else:
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        if 'tide_level' not in df.columns:
            df['tide_level'] = pd.to_numeric(df.iloc[:,1], errors='coerce')
        else:
            df['tide_level'] = pd.to_numeric(df['tide_level'], errors='coerce')
        df = df[['datetime', 'tide_level']]
        dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)
    all_df = all_df.drop_duplicates('datetime').set_index('datetime').sort_index()
    all_df = all_df.resample('1h').interpolate('linear')
    all_df = all_df.ffill()
    return all_df

def load_pump():
    dfs = []
    for f in PUMP_FILES:
        if not os.path.exists(f):
            continue
        df = pd.read_csv(f)
        df['datetime'] = pd.to_datetime(df['datetime'])
        dfs.append(df[['datetime', 'pumping_flow']])
    all_df = pd.concat(dfs, ignore_index=True)
    all_df = all_df.drop_duplicates('datetime').set_index('datetime').sort_index()
    return all_df

def load_sluice():
    dfs = []
    for f in SLUICE_FILES:
        if not os.path.exists(f):
            continue
        df = pd.read_csv(f)
        df['datetime'] = pd.to_datetime(df['datetime'])
        dfs.append(df[['datetime', 'net_flow']])
    all_df = pd.concat(dfs, ignore_index=True)
    all_df = all_df.drop_duplicates('datetime').set_index('datetime').sort_index()
    return all_df

# 特征工程

def build_feature_df(seq_len=12):
    yangtze = load_water_level()
    tide = load_tide()
    pump = load_pump()
    sluice = load_sluice()
    df = yangtze.copy()
    df['month'] = df.index.month
    
    # 优化：使用shift方法批量计算延迟潮位特征，更加高效
    # 计算影响当前水位的潮位（使用4、5、6、7小时前的潮位）
    df['tide_lag_4h'] = tide['tide_level'].shift(4)  # 4小时前的潮位
    df['tide_lag_5h'] = tide['tide_level'].shift(5)  # 5小时前的潮位
    df['tide_lag_6h'] = tide['tide_level'].shift(6)  # 6小时前的潮位
    df['tide_lag_7h'] = tide['tide_level'].shift(7)  # 7小时前的潮位
    
    # 添加流量特征
    df['pumping_flow'] = pump.reindex(df.index)['pumping_flow']
    df['sluice_flow'] = sluice.reindex(df.index)['net_flow']
    
    # 移除潮位变化率特征，只保留延迟潮位
    
    df = df.dropna()
    return df

# 构建样本：输入为历史12小时水位+当前潮位+当前节制闸/抽水站流量，输出为未来3小时水位

def create_samples(df, seq_len=12, prediction_horizon=3):
    X, y = [], []
    for i in range(seq_len, len(df)-prediction_horizon+1):
        # 获取历史12小时的水位序列
        water_seq = df.iloc[i-seq_len:i]['yangtze_level'].values  # (12,)
        
        # 获取历史12小时对应的潮位延迟特征（每个时间步使用对应时刻的特征）
        tide_4h_seq = df.iloc[i-seq_len:i]['tide_lag_4h'].values  # (12,) - 每个时间步的4小时前潮位
        tide_5h_seq = df.iloc[i-seq_len:i]['tide_lag_5h'].values  # (12,) - 每个时间步的5小时前潮位
        tide_6h_seq = df.iloc[i-seq_len:i]['tide_lag_6h'].values  # (12,) - 每个时间步的6小时前潮位
        tide_7h_seq = df.iloc[i-seq_len:i]['tide_lag_7h'].values  # (12,) - 每个时间步的7小时前潮位
        
        # 获取历史12小时对应的流量特征
        pump_seq = df.iloc[i-seq_len:i]['pumping_flow'].values  # (12,) - 每个时间步的抽水站流量
        sluice_seq = df.iloc[i-seq_len:i]['sluice_flow'].values  # (12,) - 每个时间步的节制闸流量
        
        # 构建特征向量：每个时间步使用对应时刻的特征
        # 这样更符合物理逻辑：每个时间步的水位受该时刻的潮位和流量影响
        features = np.column_stack([
            water_seq,          # (12,) - 历史12小时水位
            tide_4h_seq,        # (12,) - 历史12小时对应的4小时前潮位
            tide_5h_seq,        # (12,) - 历史12小时对应的5小时前潮位
            tide_6h_seq,        # (12,) - 历史12小时对应的6小时前潮位
            tide_7h_seq,        # (12,) - 历史12小时对应的7小时前潮位
            pump_seq,           # (12,) - 历史12小时对应的抽水站流量
            sluice_seq          # (12,) - 历史12小时对应的节制闸流量
        ])  # (12,7) - 7个特征：水位、4个潮位延迟、2个流量
        
        X.append(features)
        # 输出为未来3小时的水位
        y.append(df.iloc[i:i+prediction_horizon]['yangtze_level'].values)
    X = np.array(X)
    y = np.array(y)
    return X, y

# PyTorch LSTM模型 - 预测未来3小时水位

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.2, output_size=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.fc1 = nn.Linear(hidden_size * 2, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

# 主流程

def main():
    print('【1/6】正在加载和处理数据...')
    df = build_feature_df()
    
    # 添加数据质量检查
    print(f'原始数据形状: {df.shape}')
    print(f'潮位特征缺失值统计:')
    print(f'  tide_lag_4h: {df["tide_lag_4h"].isna().sum()}/{len(df)} ({df["tide_lag_4h"].isna().sum()/len(df)*100:.1f}%)')
    print(f'  tide_lag_5h: {df["tide_lag_5h"].isna().sum()}/{len(df)} ({df["tide_lag_5h"].isna().sum()/len(df)*100:.1f}%)')
    print(f'  tide_lag_6h: {df["tide_lag_6h"].isna().sum()}/{len(df)} ({df["tide_lag_6h"].isna().sum()/len(df)*100:.1f}%)')
    print(f'  tide_lag_7h: {df["tide_lag_7h"].isna().sum()}/{len(df)} ({df["tide_lag_7h"].isna().sum()/len(df)*100:.1f}%)')
    
    print('【2/6】正在准备特征和归一化...')
    scaler = MinMaxScaler()
    # 使用实际存在的特征列名（7个特征）
    feature_cols = ['yangtze_level', 'tide_lag_4h', 'tide_lag_5h', 'tide_lag_6h', 'tide_lag_7h', 
                    'pumping_flow', 'sluice_flow']
    X_all = df[feature_cols].values
    X_scaled = scaler.fit_transform(X_all)
    df_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=df.index)
    
    # 检查归一化后的特征分布
    print(f'归一化后特征统计:')
    for i, col in enumerate(feature_cols):
        print(f'  {col}: 均值={X_scaled[:, i].mean():.4f}, 标准差={X_scaled[:, i].std():.4f}, 范围=[{X_scaled[:, i].min():.4f}, {X_scaled[:, i].max():.4f}]')
    
    X, y = create_samples(df_scaled, seq_len=12)
    print(f'样本数据形状: X={X.shape}, y={y.shape}')
    print(f'输出y的统计: 均值={y.mean():.4f}, 标准差={y.std():.4f}')
    
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    print('【3/6】正在转换为torch张量...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)  # 移除unsqueeze(-1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).to(device)  # 移除unsqueeze(-1)
    print('【4/6】正在构建和训练LSTM模型...')
    input_dim = X_train.shape[2]
    model = LSTMModel(input_size=input_dim, num_layers=2).to(device)  # 设置num_layers=2避免dropout警告
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    num_epochs = 100
    batch_size = 64
    train_loss_list = []
    val_loss_list = []
    for epoch in range(num_epochs):
        model.train()
        permutation = torch.randperm(X_train_t.size(0))
        train_loss = 0
        for i in range(0, X_train_t.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train_t[indices], y_train_t[indices]
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
        train_loss /= X_train_t.size(0)
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test_t)
            val_loss = criterion(val_outputs, y_test_t).item()
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        if (epoch+1) % 5 == 0 or epoch == 1 or epoch == num_epochs:
            print(f'  训练进度: {epoch+1}/{num_epochs}，训练损失: {train_loss:.5f}，验证损失: {val_loss:.5f}')
    print('【5/6】正在评估模型...')
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_t).cpu().numpy()
    
    # 添加预测结果调试信息
    print(f'\n预测结果形状: y_pred={y_pred.shape}, y_test={y_test.shape}')
    print(f'预测值统计: 均值={y_pred.mean():.4f}, 标准差={y_pred.std():.4f}')
    print(f'真实值统计: 均值={y_test.mean():.4f}, 标准差={y_test.std():.4f}')
    
    # 检查每个时间步的预测分布
    for i in range(3):
        print(f'未来第{i+1}小时 - 预测值: 均值={y_pred[:, i].mean():.4f}, 标准差={y_pred[:, i].std():.4f}')
        print(f'未来第{i+1}小时 - 真实值: 均值={y_test[:, i].mean():.4f}, 标准差={y_test[:, i].std():.4f}')
    
    # 修复反归一化：保持3小时维度，分别对每个小时进行反归一化
    # 获取水位特征在特征向量中的位置（第一个特征）
    water_level_scaler = MinMaxScaler()
    water_level_scaler.fit(df[['yangtze_level']].values)
    
    # 反归一化预测值和真实值，保持3小时维度
    y_test_inv = water_level_scaler.inverse_transform(y_test)  # 形状: (样本数, 3)
    y_pred_inv = water_level_scaler.inverse_transform(y_pred)  # 形状: (样本数, 3)
    
    # 逐小时评估模型性能
    print('\n=== 逐小时模型评估结果 ===')
    hours = ['未来第1小时', '未来第2小时', '未来第3小时']
    
    for i in range(3):
        # 现在y_test_inv和y_pred_inv都是(样本数, 3)的2D数组
        y_test_hour = y_test_inv[:, i]  # 第i小时的测试集真实值
        y_pred_hour = y_pred_inv[:, i]  # 第i小时的测试集预测值
        
        mse_hour = mean_squared_error(y_test_hour, y_pred_hour)
        mae_hour = mean_absolute_error(y_test_hour, y_pred_hour)
        r2_hour = r2_score(y_test_hour, y_pred_hour)
        
        print(f'{hours[i]}:')
        print(f'  MSE: {mse_hour:.4f}')
        print(f'  MAE: {mae_hour:.4f}')
        print(f'  R²:  {r2_hour:.4f}')
        print()
    
    # 整体评估（所有3小时的平均值）
    print('=== 整体模型评估结果（3小时平均）===')
    # 将所有3小时的数据合并为一维数组进行评估
    y_test_all = y_test_inv.flatten()  # 将所有3小时的真实值合并
    y_pred_all = y_pred_inv.flatten()  # 将所有3小时的预测值合并
    
    mse = mean_squared_error(y_test_all, y_pred_all)
    mae = mean_absolute_error(y_test_all, y_pred_all)
    r2 = r2_score(y_test_all, y_pred_all)
    print(f'测试集MSE: {mse:.4f}')
    print(f'测试集MAE: {mae:.4f}')
    print(f'测试集R2: {r2:.4f}')
    print('【6/6】正在保存结果图片...')
    output_folder = os.path.join(BASE_DIR, '预测结果')
    os.makedirs(output_folder, exist_ok=True)
    # 损失曲线
    plt.figure(figsize=(8,4))
    plt.plot(train_loss_list, label='训练损失')
    plt.plot(val_loss_list, label='验证损失')
    plt.legend()
    plt.title('训练损失曲线')
    plt.xlabel('训练轮次')
    plt.ylabel('损失')
    loss_img_path = os.path.join(output_folder, 'train_loss.png')
    plt.savefig(loss_img_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'训练损失曲线已保存到 {loss_img_path}，用于观察模型训练过程的收敛情况。')
    # 预测对比 - 分别显示3小时的预测结果
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    hours = ['未来第1小时', '未来第2小时', '未来第3小时']
    
    for i in range(3):
        # 现在y_test_inv和y_pred_inv都是(样本数, 3)的2D数组
        y_test_hour = y_test_inv[:, i]  # 第i小时的测试集真实值
        y_pred_hour = y_pred_inv[:, i]  # 第i小时的测试集预测值
        
        axes[i].plot(y_test_hour, label='真实值', alpha=0.8)
        axes[i].plot(y_pred_hour, label='预测值', alpha=0.8)
        axes[i].set_title(f'{hours[i]}水位预测对比')
        axes[i].set_xlabel('样本序号')
        axes[i].set_ylabel('水位 (m)')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    pred_img_path = os.path.join(output_folder, 'test_pred_vs_true.png')
    plt.savefig(pred_img_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'测试集3小时逐小时预测对比图已保存到 {pred_img_path}，用于直观比较各时间点的预测值与真实值。')
    # 误差分析 - 分别分析3小时的误差
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    hours = ['未来第1小时', '未来第2小时', '未来第3小时']
    
    for i in range(3):
        # 现在y_test_inv和y_pred_inv都是(样本数, 3)的2D数组
        y_test_hour = y_test_inv[:, i]  # 第i小时的测试集真实值
        y_pred_hour = y_pred_inv[:, i]  # 第i小时的测试集预测值
        errors_hour = y_pred_hour - y_test_hour
        
        # 误差分布直方图
        axes[i].hist(errors_hour, bins=30, color='skyblue', edgecolor='k', alpha=0.7)
        axes[i].set_title(f'{hours[i]}预测误差分布直方图')
        axes[i].set_xlabel('预测误差（预测值-真实值）')
        axes[i].set_ylabel('频数')
        axes[i].grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    err_hist_path = os.path.join(output_folder, 'error_hist.png')
    plt.savefig(err_hist_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'3小时逐小时误差分布直方图已保存到 {err_hist_path}，用于观察各时间点的误差分布特征。')
    # 2. 误差随时间变化曲线 - 分别显示3小时
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    hours = ['未来第1小时', '未来第2小时', '未来第3小时']
    
    for i in range(3):
        # 现在y_test_inv和y_pred_inv都是(样本数, 3)的2D数组
        y_test_hour = y_test_inv[:, i]  # 第i小时的测试集真实值
        y_pred_hour = y_pred_inv[:, i]  # 第i小时的测试集预测值
        errors_hour = y_pred_hour - y_test_hour
        abs_errors_hour = np.abs(errors_hour)
        
        axes[i].plot(errors_hour, label='误差', alpha=0.8)
        axes[i].plot(abs_errors_hour, label='绝对误差', alpha=0.7)
        axes[i].set_title(f'{hours[i]}误差随时间变化曲线')
        axes[i].set_xlabel('样本序号')
        axes[i].set_ylabel('误差')
        axes[i].legend()
        axes[i].grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    err_time_path = os.path.join(output_folder, 'error_time.png')
    plt.savefig(err_time_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'3小时逐小时误差随时间变化曲线已保存到 {err_time_path}，用于分析各时间点误差的时序特征。')
    # 3. 真实值与预测值散点图 - 分别显示3小时
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    hours = ['未来第1小时', '未来第2小时', '未来第3小时']
    
    for i in range(3):
        y_test_hour = y_test_inv[:, i] if y_test_inv.ndim > 1 else y_test_inv
        y_pred_hour = y_pred_inv[:, i] if y_pred_inv.ndim > 1 else y_pred_inv
        
        axes[i].scatter(y_test_hour, y_pred_hour, alpha=0.5, s=10, label='预测点')
        minv, maxv = min(y_test_hour.min(), y_pred_hour.min()), max(y_test_hour.max(), y_pred_hour.max())
        axes[i].plot([minv, maxv], [minv, maxv], 'r--', label='y=x')
        axes[i].set_xlabel('真实值')
        axes[i].set_ylabel('预测值')
        axes[i].set_title(f'{hours[i]}真实值与预测值散点图')
        axes[i].legend()
        axes[i].grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    scatter_path = os.path.join(output_folder, 'scatter_pred_vs_true.png')
    plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'3小时逐小时真实值与预测值散点图已保存到 {scatter_path}，用于直观判断各时间点模型偏差和离群点。')
    # 相关性热力图包含所有特征
    try:
        import seaborn as sns
        corr = df[['yangtze_level', 'tide_lag_4h', 'tide_lag_5h', 'tide_lag_6h', 'tide_lag_7h', 'pumping_flow', 'sluice_flow']].corr()
        plt.figure(figsize=(8,7))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('特征相关性热力图（含4-7小时延迟潮位、节制闸与抽水站）')
        heatmap_path = os.path.join(output_folder, 'feature_corr_heatmap.png')
        plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'特征相关性热力图已保存到 {heatmap_path}，用于分析全部特征的相关性。')
    except ImportError:
        print('未安装seaborn，跳过相关性热力图。')
    # 5. 按月份分组的平均绝对误差柱状图 - 分别显示3小时
    if hasattr(df, 'index') and hasattr(df.index, 'month'):
        months = df.index[-len(y_test_inv):].month
        month_mae = {1: [], 2: [], 3: []}  # 3小时的月度MAE
        
        # 初始化所有月份的MAE为0
        for hour in range(3):
            month_mae[hour+1] = [0.0] * 12
        
        for m in range(1, 13):
            mask = (months == m)
            if np.any(mask):
                for hour in range(3):
                    y_test_hour = y_test_inv[mask, hour] if y_test_inv.ndim > 1 else y_test_inv[mask]
                    y_pred_hour = y_pred_inv[mask, hour] if y_pred_inv.ndim > 1 else y_pred_inv[mask]
                    abs_errors_hour = np.abs(y_pred_hour - y_test_hour)
                    month_mae[hour+1][m-1] = np.mean(abs_errors_hour)
        
        # 绘制3小时的月度MAE对比
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        hours = ['1小时后', '2小时后', '3小时后']
        
        for hour in range(3):
            axes[hour].bar(range(1, 13), month_mae[hour+1], color='orange', alpha=0.7)
            axes[hour].set_xlabel('月份')
            axes[hour].set_ylabel('平均绝对误差')
            axes[hour].set_title(f'{hours[hour]}按月份分组的平均绝对误差')
            axes[hour].grid(True, linestyle='--', alpha=0.5)
            axes[hour].set_xticks(range(1, 13))
        
        plt.tight_layout()
        month_mae_path = os.path.join(output_folder, 'mae_by_month.png')
        plt.savefig(month_mae_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'3小时逐小时按月份分组的平均绝对误差柱状图已保存到 {month_mae_path}，用于分析各时间点模型的季节性表现。')
    # 6. 残差自相关图 - 分别显示3小时
    try:
        from statsmodels.graphics.tsaplots import plot_acf
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        hours = ['1小时后', '2小时后', '3小时后']
        
        for i in range(3):
            y_test_hour = y_test_inv[:, i] if y_test_inv.ndim > 1 else y_test_inv
            y_pred_hour = y_pred_inv[:, i] if y_pred_inv.ndim > 1 else y_pred_inv
            errors_hour = y_pred_hour - y_test_hour
            
            plot_acf(errors_hour, lags=40, alpha=0.05, ax=axes[i])
            axes[i].set_title(f'{hours[i]}残差自相关图')
            axes[i].set_xlabel('滞后阶数')
            axes[i].set_ylabel('自相关')
        
        plt.tight_layout()
        acf_path = os.path.join(output_folder, 'residual_acf.png')
        plt.savefig(acf_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'3小时逐小时残差自相关图已保存到 {acf_path}，用于判断各时间点残差的时序相关性。')
    except ImportError:
        print('未安装statsmodels，跳过残差自相关图。')
    print('结果图片已保存到 预测结果 文件夹。')
    # 保存模型参数
    model_path = os.path.join(output_folder, 'lstm_yangtze_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f'模型参数已保存到 {model_path}')

    # 分月最佳时滞分析
    def monthly_best_lag_analysis(yangtzedf, tidedf, output_folder):
        import calendar
        max_lag = 12
        months = range(1, 13)
        best_lags = []
        best_corrs = []
        plt.figure(figsize=(16, 12))
        for i, month in enumerate(months, 1):
            y_month = yangtzedf[yangtzedf.index.month == month]
            t_month = tidedf[tidedf.index.month == month]
            corrs = []
            for lag in range(0, max_lag+1):
                tide_lagged = t_month['tide_level'].shift(lag)
                valid_idx = y_month.index.intersection(tide_lagged.index)
                if len(valid_idx) < 10:
                    corrs.append(np.nan)
                    continue
                corr = y_month.loc[valid_idx, 'yangtze_level'].corr(tide_lagged.loc[valid_idx])
                corrs.append(corr)
            best_lag = int(np.nanargmax(np.abs(corrs)))
            best_corr = corrs[best_lag]
            best_lags.append(best_lag)
            best_corrs.append(best_corr)
            plt.subplot(4, 3, i)
            plt.plot(range(0, max_lag+1), corrs, marker='o')
            plt.title(f'{month}月 最佳时滞:{best_lag}h, 相关性:{best_corr:.2f}')
            plt.xlabel('时滞（小时）')
            plt.ylabel('相关性')
            plt.tight_layout()
        lag_curve_path = os.path.join(output_folder, 'monthly_lag_correlation_curves.png')
        plt.savefig(lag_curve_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'每月相关性-时滞曲线已保存到 {lag_curve_path}')
        # 绘制最佳时滞分布
        plt.figure(figsize=(8,4))
        plt.bar(months, best_lags)
        plt.xlabel('月份')
        plt.ylabel('最佳时滞（小时）')
        plt.title('每月最佳潮位时滞分布')
        lag_bar_path = os.path.join(output_folder, 'monthly_best_lag_bar.png')
        plt.savefig(lag_bar_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'每月最佳时滞分布图已保存到 {lag_bar_path}')
        # 输出每月最佳时滞和相关性
        for m, lag, corr in zip(months, best_lags, best_corrs):
            print(f'{m}月：最佳时滞={lag}小时，相关性={corr:.2f}')

    # 保存评估文件
    eval_path = os.path.join(output_folder, '模型评估报告.txt')
    with open(eval_path, 'w', encoding='utf-8') as f:
        f.write('长江侧水位预测LSTM模型评估报告\n')
        f.write('===============================\n')
        f.write(f'训练轮次：{num_epochs}\n\n')
        
        # 逐小时评估结果
        f.write('=== 逐小时模型评估结果 ===\n')
        hours = ['1小时后', '2小时后', '3小时后']
        
        for i in range(3):
            y_test_hour = y_test_inv[:, i] if y_test_inv.ndim > 1 else y_test_inv
            y_pred_hour = y_pred_inv[:, i] if y_pred_inv.ndim > 1 else y_pred_inv
            
            mse_hour = mean_squared_error(y_test_hour, y_pred_hour)
            mae_hour = mean_absolute_error(y_test_hour, y_pred_hour)
            r2_hour = r2_score(y_test_hour, y_pred_hour)
            
            f.write(f'{hours[i]}:\n')
            f.write(f'  MSE: {mse_hour:.4f}\n')
            f.write(f'  MAE: {mae_hour:.4f}\n')
            f.write(f'  R²:  {r2_hour:.4f}\n\n')
        
        # 整体评估结果
        f.write('=== 整体模型评估结果（3小时平均）===\n')
        f.write(f'测试集MSE: {mse:.4f}\n')
        f.write(f'测试集MAE: {mae:.4f}\n')
        f.write(f'测试集R2: {r2:.4f}\n\n')
        
        # 模型表现分析
        f.write('模型表现分析：\n')
        f.write('1. 逐小时表现：\n')
        for i in range(3):
            y_test_hour = y_test_inv[:, i] if y_test_inv.ndim > 1 else y_test_inv
            y_pred_hour = y_pred_inv[:, i] if y_pred_inv.ndim > 1 else y_pred_inv
            r2_hour = r2_score(y_test_hour, y_pred_hour)
            
            if r2_hour > 0.8:
                f.write(f'   {hours[i]}: 预测效果优秀 (R²={r2_hour:.3f})\n')
            elif r2_hour > 0.6:
                f.write(f'   {hours[i]}: 预测效果良好 (R²={r2_hour:.3f})\n')
            else:
                f.write(f'   {hours[i]}: 预测效果一般 (R²={r2_hour:.3f})\n')
        
        f.write('\n2. 整体表现：\n')
        if r2 > 0.8:
            f.write('   模型整体拟合效果优秀，预测能力较强，适合实际应用。\n')
        elif r2 > 0.6:
            f.write('   模型整体拟合效果良好，预测能力一般，可用于趋势分析。\n')
        else:
            f.write('   模型整体拟合效果一般，建议进一步优化特征或模型结构。\n')
        
        f.write('\n3. 评估指标说明：\n')
        f.write('   MSE（均方误差）: 越小越好，反映预测值与真实值的平均平方差\n')
        f.write('   MAE（平均绝对误差）: 越小越好，反映预测值与真实值的平均绝对差\n')
        f.write('   R²（决定系数）: 越接近1越好，反映模型解释数据变异性的能力\n')
        
        f.write('\n4. 应用建议：\n')
        f.write('   根据逐小时评估结果，可以针对性地优化特定时间点的预测性能\n')
        f.write('   对于预测效果较好的时间点，可以增加置信度；对于效果一般的时间点，建议谨慎使用\n')
    
    print(f'模型评估报告已保存到 {eval_path}')

if __name__ == '__main__':
    main()
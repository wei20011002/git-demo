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
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 长江水位预测模型 - 基于LSTM网络架构
# ============================================================================
#
# 模型结构：
# ┌─────────────────────────────────────────────────────────────────┐
# │                        输入层 (18维)                            │
# │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐   │
# │  │长江侧12小时 │ │延迟潮位特征 │ │        流量特征         │   │
# │  │   水位     │ │ (4,5,6,7h前)│ │    (节制闸+抽水站)     │   │
# │  │   (12维)   │ │   (4维)     │ │        (2维)           │   │
# │  └─────────────┘ └─────────────┘ └─────────────────────────┘   │
# └─────────────────────────────────────────────────────────────────┘
#                                    │
#                                    ▼
# ┌─────────────────────────────────────────────────────────────────┐
# │                    LSTM层                                       │
# │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐             │
# │  │  LSTM   │ │   FC1   │ │   FC2   │ │   FC3   │             │
# │  │ 18→64   │ │ 64→32   │ │  32→16  │ │  16→3   │             │
# │  └─────────┘ └─────────┘ └─────────┘ └─────────┘             │
# └─────────────────────────────────────────────────────────────────┘
#                                    │
#                                    ▼
# ┌─────────────────────────────────────────────────────────────────┐
# │                        输出层 (3维)                            │
# │  ┌─────────┐ ┌─────────┐ ┌─────────┐                         │
# │  │ 第1小时 │ │ 第2小时 │ │ 第3小时 │                         │
# │  │长江水位 │ │长江水位 │ │长江水位 │                         │
# │  └─────────┘ └─────────┘ └─────────┘                         │
# └─────────────────────────────────────────────────────────────────┘
#
# 物理约束：
# - 潮位 → 长江水位 (因果关系链)
# - 考虑延迟效应 (4-7小时延迟)
# - 当前流量反映即时影响
#
# 输入特征：
# 1. 长江侧历史12小时水位数据 (12个特征)
# 2. 4小时前的潮位数据 (1个特征)
# 3. 5小时前的潮位数据 (1个特征)
# 4. 6小时前的潮位数据 (1个特征)
# 5. 7小时前的潮位数据 (1个特征)
# 6. 当前节制闸流量 (1个特征)
# 7. 当前抽水站流量 (1个特征)
# 总计：18个输入特征
#
# 输出：
# 长江侧未来3小时的水位预测值 (3个输出)
#
# 物理约束：
# - 潮位影响长江水位
# - 考虑延迟效应和因果关系
# - 流量特征反映人工调控影响
#
# 模型优化：
# - 引入潮位数据提升预测精度
# - LSTM网络架构捕获时序依赖关系
# - 物理约束优化延迟关系建模
#
# ============================================================================

# 数据路径配置
DATA_YEARS = [2022, 2023, 2024]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIRS = [os.path.join(BASE_DIR, '..', str(year)) for year in DATA_YEARS]
TIDE_FILES = [os.path.join(BASE_DIR, '..', '潮位数据处理', f'{year}年潮位数据_真正小时级.csv') for year in DATA_YEARS]
PUMP_FILES = [os.path.join(BASE_DIR, '..', 'output_images', f'{year}年抽水站流量数据.csv') for year in DATA_YEARS]
SLUICE_FILES = [os.path.join(BASE_DIR, '..', 'output_images', f'{year}年节制闸流量数据.csv') for year in DATA_YEARS]

def load_water_level():
    """加载长江水位数据"""
    print("正在加载长江水位数据...")
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
    
    if not dfs:
        raise ValueError("未找到任何长江水位数据")
    
    all_df = pd.concat(dfs, ignore_index=True)
    all_df = all_df.drop_duplicates('datetime').set_index('datetime').sort_index()
    all_df = all_df.dropna()
    
    print(f"长江水位数据加载完成，时间范围: {all_df.index.min()} 到 {all_df.index.max()}")
    print(f"数据点数: {len(all_df)}")
    return all_df

def load_tide():
    """加载潮位数据"""
    print("正在加载潮位数据...")
    dfs = []
    for f in TIDE_FILES:
        if not os.path.exists(f):
            print(f"警告: 潮位文件不存在: {f}")
            continue
        try:
            df = pd.read_csv(f)
            if 'datetime' not in df.columns:
                df['datetime'] = pd.to_datetime(df.iloc[:,0], errors='coerce')
            else:
                df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
            if 'tide_level' not in df.columns:
                df['tide_level'] = pd.to_numeric(df.iloc[:,1], errors='coerce')
            else:
                df['tide_level'] = pd.to_numeric(df['tide_level'], errors='coerce')
            df = df[['datetime', 'tide_level']].dropna()
            dfs.append(df)
        except Exception as e:
            print(f"加载潮位文件失败 {f}: {e}")
            continue
    
    if not dfs:
        raise ValueError("未找到任何潮位数据")
    
    all_df = pd.concat(dfs, ignore_index=True)
    all_df = all_df.drop_duplicates('datetime').set_index('datetime').sort_index()
    all_df = all_df.dropna()
    
    print(f"潮位数据加载完成，时间范围: {all_df.index.min()} 到 {all_df.index.max()}")
    print(f"数据点数: {len(all_df)}")
    return all_df

def load_pump():
    """加载抽水站流量数据"""
    print("正在加载抽水站流量数据...")
    dfs = []
    for f in PUMP_FILES:
        if not os.path.exists(f):
            print(f"警告: 抽水站文件不存在: {f}")
            continue
        try:
            df = pd.read_csv(f)
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df[['datetime', 'pumping_flow']].dropna()
            dfs.append(df)
        except Exception as e:
            print(f"加载抽水站文件失败 {f}: {e}")
            continue
    
    if not dfs:
        raise ValueError("未找到任何抽水站数据")
    
    all_df = pd.concat(dfs, ignore_index=True)
    all_df = all_df.drop_duplicates('datetime').set_index('datetime').sort_index()
    all_df = all_df.dropna()
    
    print(f"抽水站数据加载完成，时间范围: {all_df.index.min()} 到 {all_df.index.max()}")
    print(f"数据点数: {len(all_df)}")
    return all_df

def load_sluice():
    """加载节制闸流量数据"""
    print("正在加载节制闸流量数据...")
    dfs = []
    for f in SLUICE_FILES:
        if not os.path.exists(f):
            print(f"警告: 节制闸文件不存在: {f}")
            continue
        try:
            df = pd.read_csv(f)
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df[['datetime', 'net_flow']].dropna()
            dfs.append(df)
        except Exception as e:
            print(f"加载节制闸文件失败 {f}: {e}")
            continue
    
    if not dfs:
        raise ValueError("未找到任何节制闸数据")
    
    all_df = pd.concat(dfs, ignore_index=True)
    all_df = all_df.drop_duplicates('datetime').set_index('datetime').sort_index()
    all_df = all_df.dropna()
    
    print(f"节制闸数据加载完成，时间范围: {all_df.index.min()} 到 {all_df.index.max()}")
    print(f"数据点数: {len(all_df)}")
    return all_df

def verify_time_alignment(yangtze, tide, pump, sluice):
    """验证时间对齐情况"""
    print("\n=== 时间对齐验证 ===")
    print(f"长江水位数据时间范围: {yangtze.index.min()} 到 {yangtze.index.max()}")
    print(f"潮位数据时间范围: {tide.index.min()} 到 {tide.index.max()}")
    print(f"抽水站数据时间范围: {pump.index.min()} 到 {pump.index.max()}")
    print(f"节制闸数据时间范围: {sluice.index.min()} 到 {sluice.index.max()}")
    
    # 找到所有数据的共同时间范围
    common_index = yangtze.index.intersection(tide.index).intersection(pump.index).intersection(sluice.index)
    print(f"所有数据重叠时间点数: {len(common_index)}")
    if len(common_index) > 0:
        print(f"重叠时间范围: {common_index.min()} 到 {common_index.max()}")
    else:
        print("警告: 没有找到所有数据的重叠时间点!")
    
    return common_index

def build_aligned_feature_df():
    """构建对齐的特征数据框"""
    print("\n=== 构建对齐特征数据 ===")
    
    # 加载所有数据
    yangtze = load_water_level()
    tide = load_tide()
    pump = load_pump()
    sluice = load_sluice()
    
    # 验证时间对齐
    common_index = verify_time_alignment(yangtze, tide, pump, sluice)
    
    if len(common_index) == 0:
        raise ValueError("无法找到所有数据的重叠时间点，请检查数据完整性")
    
    # 使用共同时间索引构建特征
    df = yangtze.loc[common_index].copy()
    df['month'] = df.index.month
    df['hour'] = df.index.hour
    
    # 计算精确对应的潮位特征（与预测时间点直接对应）
    print("计算精确对应的潮位特征...")
    print("潮位特征设计:")
    print("  - tide_lag_4h: 4小时前潮位")
    print("  - tide_lag_5h: 5小时前潮位")
    print("  - tide_lag_6h: 6小时前潮位")
    print("  - tide_lag_7h: 7小时前潮位")
    
    # 计算延迟潮位特征（4-7小时前）
    df['tide_lag_4h'] = tide.loc[common_index]['tide_level'].shift(4)
    df['tide_lag_5h'] = tide.loc[common_index]['tide_level'].shift(5)
    df['tide_lag_6h'] = tide.loc[common_index]['tide_level'].shift(6)
    df['tide_lag_7h'] = tide.loc[common_index]['tide_level'].shift(7)
    
    # 添加流量特征
    df['pumping_flow'] = pump.loc[common_index]['pumping_flow']
    df['sluice_flow'] = sluice.loc[common_index]['net_flow']
    
    # 移除包含NaN的行
    initial_count = len(df)
    df = df.dropna()
    final_count = len(df)
    
    print(f"特征构建完成，初始数据点: {initial_count}, 最终数据点: {final_count}")
    print(f"数据时间范围: {df.index.min()} 到 {df.index.max()}")
    
    # 检查特征完整性
    print("\n特征完整性检查:")
    for col in df.columns:
        missing = df[col].isna().sum()
        print(f"  {col}: 缺失值 {missing}/{len(df)} ({missing/len(df)*100:.1f}%)")
    
    return df

def create_samples(df, seq_len=12, prediction_horizon=3):
    """创建训练样本"""
    print(f"\n=== 创建训练样本 ===")
    print(f"序列长度: {seq_len}, 预测时域: {prediction_horizon}")
    
    X, y = [], []
    valid_samples = 0
    
    for i in range(seq_len, len(df) - prediction_horizon + 1):
        # 历史12小时的长江水位序列（使用归一化后的数据）
        yangtze_history = df.iloc[i-seq_len:i]['yangtze_level_scaled'].values
        
        # 当前时刻的潮位延迟特征（4-7小时前）
        current_tide_4h = df.iloc[i]['tide_lag_4h']
        current_tide_5h = df.iloc[i]['tide_lag_5h']
        current_tide_6h = df.iloc[i]['tide_lag_6h']
        current_tide_7h = df.iloc[i]['tide_lag_7h']
        
        # 当前时刻的流量特征
        current_pump = df.iloc[i]['pumping_flow']
        current_sluice = df.iloc[i]['sluice_flow']
        
        # 检查是否有NaN值
        if (np.isnan(yangtze_history).any() or np.isnan(current_tide_4h) or 
            np.isnan(current_tide_5h) or np.isnan(current_tide_6h) or np.isnan(current_tide_7h) or 
            np.isnan(current_sluice)):
            continue
        
        # 构建特征向量：历史水位序列 + 当前时刻的其他特征
        features = np.concatenate([
            yangtze_history.flatten(),  # 12个特征：长江侧过去12小时水位
            [current_tide_4h],          # 1个特征：4小时前潮位
            [current_tide_5h],          # 1个特征：5小时前潮位
            [current_tide_6h],          # 1个特征：6小时前潮位
            [current_tide_7h],          # 1个特征：7小时前潮位
            [current_pump],             # 1个特征：抽水站流量
            [current_sluice]            # 1个特征：节制闸流量
        ])
        
        X.append(features)
        y.append(df.iloc[i:i+prediction_horizon]['yangtze_level_scaled'].values)
        valid_samples += 1
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"样本创建完成: X={X.shape}, y={y.shape}")
    print(f"有效样本数: {valid_samples}")
    print(f"输入特征维度: {X.shape[1]} (长江历史12小时水位 + 6个当前特征)")
    
    return X, y

# LSTM模型架构说明：
# 1. LSTM网络：处理时序特征输入，适合水位预测任务
# 2. 多层设计：LSTM层 + 3层全连接网络提供足够的表达能力，捕获时序依赖关系
# 3. 激活函数：ReLU激活函数引入非线性，提高模型表达能力
# 4. Dropout正则化：防止过拟合，提高模型泛化能力
# 5. 物理约束优化：输入包含12小时历史数据和延迟潮位特征，重点优化延迟关系建模

class LSTMModel(nn.Module):
    """LSTM模型定义 - 支持3小时预测"""
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.1, output_size=3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout, bidirectional=False)
        
        # 全连接层
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, output_size)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # 重塑输入以适应LSTM (batch_size, seq_len, input_size)
        # 由于我们的输入是特征向量，将其视为序列长度为1的序列
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # 添加序列维度
        
        # LSTM层
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]
        
        # 全连接层
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc3(out)
        return out

def train_model(X_train, y_train, X_val, y_val, input_size, device):
    """训练模型"""
    print("\n=== 开始训练模型 ===")
    
    model = LSTMModel(input_size=input_size, hidden_size=64, num_layers=1, dropout=0.1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    num_epochs = 100
    batch_size = 32
    train_losses = []
    val_losses = []
    
    print(f"训练参数: 轮次={num_epochs}, 批次大小={batch_size}, 学习率=0.001")
    print("模型结构: LSTM层(64) + 3层全连接网络 (64→32→16→3)")
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        num_batches = 0
        
        # 随机打乱训练数据
        indices = torch.randperm(X_train.size(0))
        for i in range(0, X_train.size(0), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_x = X_train[batch_indices]
            batch_y = y_train[batch_indices]
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
            num_batches += 1
        
        train_loss /= num_batches
        
        # 验证阶段
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val).item()
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 打印进度
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"轮次 {epoch+1:3d}/{num_epochs}: 训练损失={train_loss:.6f}, 验证损失={val_loss:.6f}")
    
    print(f"训练完成，最终训练损失: {train_loss:.6f}, 最终验证损失: {val_loss:.6f}")
    
    return model, train_losses, val_losses

def evaluate_model(model, X_test, y_test, target_scaler, device):
    """评估模型"""
    print("\n=== 模型评估 ===")
    
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).cpu().numpy()
    
    # 确保y_test也转移到CPU并转换为numpy
    y_test_np = y_test.cpu().numpy()
    
    # 反归一化 - 使用目标数据的scaler
    y_test_inv = target_scaler.inverse_transform(y_test_np)
    y_pred_inv = target_scaler.inverse_transform(y_pred)
    
    # 逐小时评估
    hours = ['1小时后', '2小时后', '3小时后']
    print("\n逐小时评估结果:")
    
    for i in range(3):
        y_test_hour = y_test_inv[:, i]
        y_pred_hour = y_pred_inv[:, i]
        
        mse = mean_squared_error(y_test_hour, y_pred_hour)
        mae = mean_absolute_error(y_test_hour, y_pred_hour)
        r2 = r2_score(y_test_hour, y_pred_hour)
        
        print(f"{hours[i]}: MSE={mse:.6f}, MAE={mae:.6f}, R²={r2:.6f}")
    
    # 整体评估
    y_test_all = y_test_inv.flatten()
    y_pred_all = y_pred_inv.flatten()
    
    mse_overall = mean_squared_error(y_test_all, y_pred_all)
    mae_overall = mean_absolute_error(y_test_all, y_pred_all)
    r2_overall = r2_score(y_test_all, y_pred_all)
    
    print(f"\n整体评估结果: MSE={mse_overall:.6f}, MAE={mae_overall:.6f}, R²={r2_overall:.6f}")
    
    return y_test_inv, y_pred_inv, {
        'mse': mse_overall,
        'mae': mae_overall,
        'r2': r2_overall
    }

def plot_results(train_losses, val_losses, y_test, y_pred, output_folder):
    """绘制结果图表"""
    print("\n=== 生成结果图表 ===")
    
    # 1. 训练损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='训练损失', alpha=0.8)
    plt.plot(val_losses, label='验证损失', alpha=0.8)
    plt.xlabel('训练轮次')
    plt.ylabel('损失')
    plt.title('LSTM模型训练损失曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'training_loss.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. 预测对比图
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    hours = ['1小时后', '2小时后', '3小时后']
    
    for i in range(3):
        axes[i].plot(y_test[:, i], label='真实值', alpha=0.8, linewidth=1)
        axes[i].plot(y_pred[:, i], label='预测值', alpha=0.8, linewidth=1)
        axes[i].set_title(f'{hours[i]}长江水位预测对比')
        axes[i].set_xlabel('样本序号')
        axes[i].set_ylabel('水位 (m)')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'prediction_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. 散点图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i in range(3):
        axes[i].scatter(y_test[:, i], y_pred[:, i], alpha=0.6, s=20)
        min_val = min(y_test[:, i].min(), y_pred[:, i].min())
        max_val = max(y_test[:, i].max(), y_pred[:, i].max())
        axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        axes[i].set_xlabel('真实值')
        axes[i].set_ylabel('预测值')
        axes[i].set_title(f'{hours[i]}真实值 vs 预测值')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'scatter_plots.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("结果图表已保存")

def save_model_and_results(model, scalers, target_scaler, metrics, output_folder):
    """保存模型和结果"""
    print("\n=== 保存模型和结果 ===")
    
    # 保存模型
    model_path = os.path.join(output_folder, 'lstm_yangtze_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存到: {model_path}")
    
    # 保存所有归一化器
    import pickle
    scalers_path = os.path.join(output_folder, 'scalers.pkl')
    with open(scalers_path, 'wb') as f:
        pickle.dump(scalers, f)
    print(f"归一化器已保存到: {scalers_path}")
    
    # 保存目标归一化器
    target_scaler_path = os.path.join(output_folder, 'target_scaler.pkl')
    with open(target_scaler_path, 'wb') as f:
        pickle.dump(target_scaler, f)
    print(f"目标归一化器已保存到: {target_scaler_path}")
    
    # 保存评估结果
    results_path = os.path.join(output_folder, 'evaluation_results.txt')
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write("长江水位预测LSTM模型评估报告\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"模型架构: LSTM层(64) + 3层全连接网络 (64→32→16→3)\n")
        f.write(f"输入特征: 历史12小时水位 + 潮位特征(4h前, 5h前, 6h前, 7h前) + 流量数据\n")
        f.write(f"输出: 1小时后、2小时后和3小时后水位预测\n\n")
        f.write("评估指标:\n")
        f.write(f"  MSE: {metrics['mse']:.6f}\n")
        f.write(f"  MAE: {metrics['mae']:.6f}\n")
        f.write(f"  R²:  {metrics['r2']:.6f}\n")
    
    print(f"评估结果已保存到: {results_path}")

def main():
    """主函数"""
    print("=" * 60)
    print("长江水位预测LSTM模型 - 数据对齐版本")
    print("=" * 60)
    
    # 创建输出文件夹
    output_folder = os.path.join(BASE_DIR, '预测结果_对齐版')
    os.makedirs(output_folder, exist_ok=True)
    
    try:
        # 1. 构建对齐的特征数据
        df = build_aligned_feature_df()
        
        # 2. 数据归一化
        print("\n=== 数据归一化 ===")
        # 为长江水位数据创建scaler
        yangtze_scaler = MinMaxScaler()
        yangtze_scaled = yangtze_scaler.fit_transform(df[['yangtze_level']])
        df['yangtze_level_scaled'] = yangtze_scaled.flatten()
        
        # 为潮位数据创建scaler
        tide_scaler = MinMaxScaler()
        tide_cols = ['tide_lag_4h', 'tide_lag_5h', 'tide_lag_6h', 'tide_lag_7h']
        tide_scaled = tide_scaler.fit_transform(df[tide_cols])
        df_tide_scaled = pd.DataFrame(tide_scaled, columns=tide_cols, index=df.index)
        
        # 为流量数据创建scaler
        flow_scaler = MinMaxScaler()
        flow_cols = ['pumping_flow', 'sluice_flow']
        flow_scaled = flow_scaler.fit_transform(df[flow_cols])
        df_flow_scaled = pd.DataFrame(flow_scaled, columns=flow_cols, index=df.index)
        
        # 合并所有归一化后的数据
        df_scaled = df[['yangtze_level_scaled']].copy()
        df_scaled = pd.concat([df_scaled, df_tide_scaled, df_flow_scaled], axis=1)
        
        # 为目标数据创建单独的scaler
        target_scaler = MinMaxScaler()
        target_cols = ['yangtze_level']  # 只对长江水位进行归一化
        y_scaled = target_scaler.fit_transform(df[target_cols])
        
        # 将归一化后的目标数据添加到df_scaled中
        df_scaled['yangtze_level_scaled'] = y_scaled.flatten()
        
        # 3. 创建训练样本
        X, y = create_samples(df_scaled, seq_len=12, prediction_horizon=3)
        
        # 4. 划分训练集和测试集
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # 进一步划分验证集
        val_split_idx = int(len(X_train) * 0.8)
        X_train, X_val = X_train[:val_split_idx], X_train[val_split_idx:]
        y_train, y_val = y_train[:val_split_idx], y_train[val_split_idx:]
        
        print(f"\n数据集划分:")
        print(f"  训练集: {X_train.shape[0]} 样本")
        print(f"  验证集: {X_val.shape[0]} 样本")
        print(f"  测试集: {X_test.shape[0]} 样本")
        
        # 5. 转换为PyTorch张量
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")
        
        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test_t = torch.tensor(y_test, dtype=torch.float32).to(device)
        
        # 6. 训练模型
        input_size = X_train.shape[1]  # 特征数量
        model, train_losses, val_losses = train_model(X_train_t, y_train_t, X_val_t, y_val_t, input_size, device)
        
        # 7. 评估模型
        y_test_inv, y_pred_inv, metrics = evaluate_model(model, X_test_t, y_test_t, target_scaler, device)
        
        # 8. 绘制结果
        plot_results(train_losses, val_losses, y_test_inv, y_pred_inv, output_folder)
        
        # 9. 保存模型和结果
        # 创建一个包含所有scaler的字典
        scalers = {
            'yangtze_scaler': yangtze_scaler,
            'tide_scaler': tide_scaler,
            'flow_scaler': flow_scaler,
            'target_scaler': target_scaler
        }
        save_model_and_results(model, scalers, target_scaler, metrics, output_folder)
        
        print(f"\n所有结果已保存到: {output_folder}")
        print("=" * 60)
        
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 
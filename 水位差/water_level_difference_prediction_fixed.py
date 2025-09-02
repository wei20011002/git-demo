# ============================================================================
# 水位差预测模型 - 修正版
# ============================================================================
#
# 模型结构更新说明：
# 
# 运河侧LSTM模型输入特征已从30维更新为30维：
# ┌─────────────────────────────────────────────────────────────────┐
# │                        输入层 (30维)                            │
# │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐   │
# │  │长江侧12小时 │ │运河侧12小时 │ │延迟潮位特征(5,6,7,8h前)│   │
# │  │   水位     │ │   水位     │ │        潮位值           │   │
# │  │   (12维)   │ │   (12维)   │ │        (4维)           │   │
# │  └─────────────┘ └─────────────┘ └─────────────────────────┘   │
# │  ┌─────────────────────────────────────────────────────────┐   │
# │  │                    流量特征 (2维)                        │   │
# │  │           节制闸流量 + 抽水站流量                        │   │
# │  └─────────────────────────────────────────────────────────┘   │
# └─────────────────────────────────────────────────────────────────┘
#
# 输入特征详细说明：
# 1. 长江侧历史12小时水位数据 (12个特征)
# 2. 运河侧历史12小时水位数据 (12个特征)  
# 3. 5小时前的潮位数据 (1个特征)
# 4. 6小时前的潮位数据 (1个特征)
# 5. 7小时前的潮位数据 (1个特征)
# 6. 8小时前的潮位数据 (1个特征)
# 7. 当前节制闸流量 (1个特征)
# 8. 当前抽水站流量 (1个特征)
# 总计：30个输入特征
#
# 物理约束：
# - 潮位 → 长江水位 → 运河水位 (因果关系链)
# - 长江侧考虑延迟效应 (4-7小时延迟) - 较早响应潮位变化
# - 运河侧考虑延迟效应 (5-8小时延迟) - 较晚响应潮位变化
# - 当前长江水位反映即时影响
# - 流量特征反映人工调节影响
#
# ============================================================================
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import warnings
warnings.filterwarnings('ignore')

# 数据路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MERGED_DATA_FILE = os.path.join(BASE_DIR, 'merged_yangtze_yunhe.csv')

# 模型路径 - 使用修改好的模型
YUNHE_MODEL_PATH = os.path.join(BASE_DIR, '..', '运河侧水位预测', '预测结果', 'lstm_yunhe_model.pth')
YANGTZE_MODEL_PATH = os.path.join(BASE_DIR, '..', '长江侧水位预测', '预测结果_对齐版', 'lstm_yangtze_model.pth')
YANGTZE_SCALERS_PATH = os.path.join(BASE_DIR, '..', '长江侧水位预测', '预测结果_对齐版', 'scalers.pkl')
YANGTZE_TARGET_SCALER_PATH = os.path.join(BASE_DIR, '..', '长江侧水位预测', '预测结果_对齐版', 'target_scaler.pkl')

# ============================================================================
# 模型定义 - 使用修改好的模型架构
# ============================================================================

class LSTMModel(nn.Module):
    """LSTM模型 - 用于运河侧水位预测"""
    def __init__(self, input_size=19, hidden_size=128, num_layers=2, output_size=3):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 单向LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.2, bidirectional=False)
        
        # 全连接层
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, output_size)
        
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # 单向LSTM层
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
        out = self.relu(out)
        
        out = self.fc4(out)
        return out

class UpdatedYangtzeModel(nn.Module):
    """修改后的长江水位预测模型 - 全连接网络"""
    def __init__(self, input_size=18, hidden_size=64, output_size=3):
        super(UpdatedYangtzeModel, self).__init__()
        # 3层全连接网络
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 32)
        self.fc3 = nn.Linear(32, output_size)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out

# ============================================================================
# 数据加载函数
# ============================================================================

def load_merged_data():
    """加载合并的数据文件"""
    print("正在加载合并数据...")
    
    if not os.path.exists(MERGED_DATA_FILE):
        raise FileNotFoundError(f"合并数据文件不存在: {MERGED_DATA_FILE}")
    
    try:
        df = pd.read_csv(MERGED_DATA_FILE)
        print(f"原始数据点数: {len(df)}")
        
        # 处理时间列
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        
        # 重命名列
        df = df.rename(columns={
            '长江侧水位': 'yangtze_level',
            '内河侧水位': 'yunhe_level',
            'pumping_flow': 'pumping_flow',
            'net_flow': 'sluice_flow'
        })
        
        # 移除包含NaN的行
        df = df.dropna()
        print(f"处理后数据点数: {len(df)}")
        
        return df
    except Exception as e:
        print(f"加载合并数据失败: {e}")
        raise

# ============================================================================
# 特征构建函数
# ============================================================================

def build_yangtze_features(merged_df):
    """构建长江水位预测特征"""
    print("构建长江水位预测特征...")
    
    print(f"合并数据点数: {len(merged_df)}")
    
    # 构建特征数据框
    df = merged_df.copy()
    
    # 计算延迟潮位特征 (4-7小时前) - 长江侧使用较早的潮位信息
    df['tide_lag_4h'] = df['tide_level'].shift(4)
    df['tide_lag_5h'] = df['tide_level'].shift(5)
    df['tide_lag_6h'] = df['tide_level'].shift(6)
    df['tide_lag_7h'] = df['tide_level'].shift(7)
    
    # 移除包含NaN的行
    df = df.dropna()
    
    print(f"长江特征构建完成，数据点: {len(df)}")
    return df

def build_yunhe_features(merged_df):
    """构建运河水位预测特征"""
    print("构建运河水位预测特征...")
    
    # 构建特征数据框 - 包含长江和运河水位
    df = merged_df.copy()
    
    # 计算延迟潮位特征 (5-8小时前) - 与运河侧水位预测模型保持一致
    df['tide_lag_5h'] = df['tide_level'].shift(5)
    df['tide_lag_6h'] = df['tide_level'].shift(6)
    df['tide_lag_7h'] = df['tide_level'].shift(7)
    df['tide_lag_8h'] = df['tide_level'].shift(8)
    
    # 移除包含NaN的行
    df = df.dropna()
    
    print(f"运河特征构建完成，数据点: {len(df)}")
    return df

# ============================================================================
# 模型加载和预测函数
# ============================================================================

def load_yangtze_model():
    """加载修改后的长江水位预测模型"""
    print("加载长江水位预测模型...")
    
    # 检查模型文件是否存在
    if not os.path.exists(YANGTZE_MODEL_PATH):
        raise FileNotFoundError(f"长江水位模型文件不存在: {YANGTZE_MODEL_PATH}")
    
    if not os.path.exists(YANGTZE_SCALERS_PATH):
        raise FileNotFoundError(f"长江水位归一化器文件不存在: {YANGTZE_SCALERS_PATH}")
    
    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UpdatedYangtzeModel(input_size=18, hidden_size=64, output_size=3).to(device)
    model.load_state_dict(torch.load(YANGTZE_MODEL_PATH, map_location=device))
    model.eval()
    
    # 加载归一化器
    with open(YANGTZE_SCALERS_PATH, 'rb') as f:
        scalers = pickle.load(f)
    
    print("长江水位预测模型加载完成")
    return model, scalers, device

def load_yunhe_model():
    """加载运河水位预测模型"""
    print("加载运河水位预测模型...")
    
    if not os.path.exists(YUNHE_MODEL_PATH):
        raise FileNotFoundError(f"运河水位模型文件不存在: {YUNHE_MODEL_PATH}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMModel(input_size=30, hidden_size=128, num_layers=2, output_size=3).to(device)
    model.load_state_dict(torch.load(YUNHE_MODEL_PATH, map_location=device))
    model.eval()
    
    print("运河水位预测模型加载完成")
    return model, device

def predict_yangtze_level(model, scalers, features_df, device):
    """使用修改后的模型预测长江水位"""
    print("预测长江水位...")
    
    # 数据预处理
    yangtze_scaler = scalers['yangtze_scaler']
    tide_scaler = scalers['tide_scaler']
    flow_scaler = scalers['flow_scaler']
    
    # 归一化特征
    yangtze_scaled = yangtze_scaler.transform(features_df[['yangtze_level']])
    tide_scaled = tide_scaler.transform(features_df[['tide_lag_4h', 'tide_lag_5h', 'tide_lag_6h', 'tide_lag_7h']])
    flow_scaled = flow_scaler.transform(features_df[['pumping_flow', 'sluice_flow']])
    
    # 构建输入特征 (历史12小时水位 + 当前特征)
    predictions = []
    prediction_times = []  # 记录预测时间
    
    for i in range(12, len(features_df) - 2):  # 需要12小时历史数据，预测3小时
        # 历史12小时水位
        history = yangtze_scaled[i-12:i].flatten()
        
        # 当前时刻的其他特征
        current_tide = tide_scaled[i]
        current_flow = flow_scaled[i]
        
        # 拼接特征
        features = np.concatenate([history, current_tide, current_flow])
        
        # 预测
        with torch.no_grad():
            input_tensor = torch.tensor(features.reshape(1, -1), dtype=torch.float32).to(device)
            prediction = model(input_tensor).cpu().numpy()
            predictions.append(prediction.flatten())
            
            # 记录预测时间（当前时刻）
            prediction_times.append(features_df.index[i])
    
    # 反归一化
    target_scaler = scalers['target_scaler']
    predictions = np.array(predictions)
    predictions_original = target_scaler.inverse_transform(predictions)
    
    print(f"长江水位预测完成，预测样本数: {len(predictions_original)}")
    return predictions_original, prediction_times

def predict_yunhe_level(model, features_df, device):
    """预测运河水位 - 修正版本，使用30维输入特征（包含流量特征）"""
    print("预测运河水位...")
    
    # 创建归一化器（与训练时保持一致）
    scaler_yangtze = MinMaxScaler()
    scaler_yunhe = MinMaxScaler()
    scaler_tide = MinMaxScaler()
    scaler_flow = MinMaxScaler()  # 新增流量归一化器
    
    # 归一化数据
    yangtze_scaled = scaler_yangtze.fit_transform(features_df[['yangtze_level']])
    yunhe_scaled = scaler_yunhe.fit_transform(features_df[['yunhe_level']])
    
    # 处理潮位特征（5、6、7、8小时前）
    tide_features = ['tide_lag_5h', 'tide_lag_6h', 'tide_lag_7h', 'tide_lag_8h']
    tide_data = features_df[tide_features].copy()
    if tide_data.isna().any().any():
        print("检测到潮位数据包含NaN值，将使用0填充")
        tide_data = tide_data.fillna(0)
    
    tide_scaled = scaler_tide.fit_transform(tide_data)
    
    # 处理流量特征
    flow_features = ['pumping_flow', 'sluice_flow']
    flow_data = features_df[flow_features].copy()
    if flow_data.isna().any().any():
        print("检测到流量数据包含NaN值，将使用0填充")
        flow_data = flow_data.fillna(0)
    
    flow_scaled = scaler_flow.fit_transform(flow_data)
    
    # 运河模型现在需要30维输入特征：
    # - 长江侧历史12小时水位 (12个特征) - 归一化后
    # - 运河侧历史12小时水位 (12个特征) - 归一化后
    # - 延迟潮位特征：5、6、7、8小时前的潮位 (4个特征) - 归一化后
    # - 流量特征：当前节制闸和抽水站流量 (2个特征) - 归一化后
    
    predictions = []
    prediction_times = []  # 记录预测时间
    
    for i in range(12, len(features_df) - 2):  # 需要12小时历史数据，预测3小时
        # 构建30维特征向量 - 使用归一化后的数据
        features = []
        
        # 长江侧历史12小时水位（归一化后）
        yangtze_history = yangtze_scaled[i-12:i].flatten()
        features.extend(yangtze_history)
        
        # 运河侧历史12小时水位（归一化后）
        yunhe_history = yunhe_scaled[i-12:i].flatten()
        features.extend(yunhe_history)
        
        # 延迟潮位特征（归一化后）
        current_tide = tide_scaled[i]
        features.extend(current_tide)
        
        # 流量特征（归一化后）
        current_flow = flow_scaled[i]
        features.extend(current_flow)
        
        # 检查是否有NaN值
        if np.isnan(features).any():
            continue
            
        # 预测
        with torch.no_grad():
            input_tensor = torch.tensor(features, dtype=torch.float32).reshape(1, 1, -1).to(device)
            prediction = model(input_tensor).cpu().numpy()
            predictions.append(prediction.flatten())
            
            # 记录预测时间（当前时刻）
            prediction_times.append(features_df.index[i])
    
    if not predictions:
        print("警告: 没有有效的运河水位预测结果")
        return None, None
    
    predictions = np.array(predictions)
    
    # 反归一化运河水位预测结果
    predictions_original = scaler_yunhe.inverse_transform(predictions)
    
    print(f"运河水位预测完成，预测样本数: {len(predictions_original)}")
    return predictions_original, prediction_times

def calculate_water_level_difference(yangtze_predictions, yunhe_predictions, yangtze_times, yunhe_times):
    """计算水位差"""
    print("计算水位差...")
    
    if yangtze_predictions is None or yunhe_predictions is None:
        print("无法计算水位差，预测数据不完整")
        return None
    
    # 检查预测结果的长度
    yangtze_len = len(yangtze_predictions)
    yunhe_len = len(yunhe_predictions)
    
    print(f"长江水位预测样本数: {yangtze_len}")
    print(f"运河水位预测样本数: {yunhe_len}")
    
    # 取较小的长度，确保两个预测结果长度一致
    min_len = min(yangtze_len, yunhe_len)
    if min_len == 0:
        print("警告: 没有有效的预测结果")
        return None
    
    # 截取相同长度的预测结果
    yangtze_pred = yangtze_predictions[:min_len]
    yunhe_pred = yunhe_predictions[:min_len]
    yangtze_times = yangtze_times[:min_len]
    yunhe_times = yunhe_times[:min_len]
    
    # 计算水位差：|长江水位 - 运河水位|
    water_level_diff = np.abs(yangtze_pred - yunhe_pred)
    
    print(f"水位差计算完成，样本数: {len(water_level_diff)}")
    return water_level_diff, yangtze_pred, yunhe_pred, yangtze_times

# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    print("=" * 60)
    print("使用修正后的模型预测水位差")
    print("=" * 60)
    print("模型输入特征说明：")
    print("运河侧LSTM模型：30维")
    print("- 长江侧历史12小时水位：12维")
    print("- 运河侧历史12小时水位：12维")
    print("- 延迟潮位特征：4维 (5、6、7、8小时前)")
    print("- 流量特征：2维 (节制闸和抽水站流量)")
    print("")
    print("长江侧全连接模型：18维")
    print("- 历史12小时水位：12维")
    print("- 延迟潮位特征：4维 (4、5、6、7小时前)")
    print("- 流量特征：2维 (节制闸和抽水站流量)")
    print("=" * 60)
    
    try:
        # 1. 加载合并数据
        merged_df = load_merged_data()
        
        # 2. 构建特征
        yangtze_features = build_yangtze_features(merged_df)
        yunhe_features = build_yunhe_features(merged_df)
        
        # 3. 加载模型
        yangtze_model, yangtze_scalers, device = load_yangtze_model()
        yunhe_model, yunhe_device = load_yunhe_model()
        
        # 4. 预测水位
        yangtze_predictions, yangtze_times = predict_yangtze_level(yangtze_model, yangtze_scalers, yangtze_features, device)
        yunhe_predictions, yunhe_times = predict_yunhe_level(yunhe_model, yunhe_features, yunhe_device)
        
        # 5. 计算水位差
        result = calculate_water_level_difference(yangtze_predictions, yunhe_predictions, yangtze_times, yunhe_times)
        
        # 6. 保存结果
        if result is not None:
            water_level_diff, yangtze_pred, yunhe_pred, prediction_times = result
            output_folder = os.path.join(BASE_DIR, '预测结果_修正版')
            os.makedirs(output_folder, exist_ok=True)
            
            # 保存预测结果
            results_df = pd.DataFrame({
                'datetime': prediction_times,
                '长江水位_1h': yangtze_pred[:, 0],
                '长江水位_2h': yangtze_pred[:, 1],
                '长江水位_3h': yangtze_pred[:, 2],
                '运河水位_1h': yunhe_pred[:, 0],
                '运河水位_2h': yunhe_pred[:, 1],
                '运河水位_3h': yunhe_pred[:, 2],
                '水位差_1h': water_level_diff[:, 0],
                '水位差_2h': water_level_diff[:, 1],
                '水位差_3h': water_level_diff[:, 2]
            })
            
            results_path = os.path.join(output_folder, '水位差预测结果.csv')
            results_df.to_csv(results_path, index=False)
            print(f"预测结果已保存到: {results_path}")
        
        print("=" * 60)
        print("水位差预测完成")
        print("=" * 60)
        
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 
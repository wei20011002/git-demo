import pandas as pd
import numpy as np
import matplotlib
# 修复matplotlib后端问题，提供多种后端选择
try:
    matplotlib.use('TkAgg')  # 优先使用TkAgg后端
except:
    try:
        matplotlib.use('Qt5Agg')  # 备选Qt5Agg后端
    except:
        matplotlib.use('Agg')  # 最后使用Agg后端
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
import os
import warnings

# 忽略matplotlib警告
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# ============================================================================
# 运河侧水位预测模型 - 基于LSTM架构
# ============================================================================
#
# 模型结构：
# ┌─────────────────────────────────────────────────────────────────┐
# │                        输入层 (30维)                            │
# │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐   │
# │  │长江侧12小时 │ │运河侧12小时 │ │延迟潮位特征(5,6,7,8h前)│   │
# │  │   水位     │ │   水位     │ │        潮位值           │   │
# │  │   (12维)   │ │   (12维)   │ │        (4维)           │   │
# │  └─────────────┘ └─────────────┘ └─────────────────────────┘   │
# └─────────────────────────────────────────────────────────────────┘
#                                    │
#                                    ▼
# ┌─────────────────────────────────────────────────────────────────┐
# │                    LSTM层 (单向LSTM)                           │
# │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐             │
# │  │   LSTM层1   │ │   LSTM层2   │ │  多层设计   │             │
# │  │   (2层)    │ │   (2层)    │ │  (128维)   │             │
# │  └─────────────┘ └─────────────┘ └─────────────┘             │
# └─────────────────────────────────────────────────────────────────┘
#                                    │
#                                    ▼
# ┌─────────────────────────────────────────────────────────────────┐
# │                    全连接层                                     │
# │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐             │
# │  │   FC1   │ │   FC2   │ │   FC3   │ │   FC4   │             │
# │  │ 128→128 │ │ 128→64  │ │  64→32  │ │  32→3   │             │
# │  └─────────┘ └─────────┘ └─────────┘ └─────────┘             │
# └─────────────────────────────────────────────────────────────────┘
#                                    │
#                                    ▼
# ┌─────────────────────────────────────────────────────────────────┐
# │                        输出层 (3维)                            │
# │  ┌─────────┐ ┌─────────┐ ┌─────────┐                         │
# │  │ 第1小时 │ │ 第2小时 │ │ 第3小时 │                         │
# │  │运河水位 │ │运河水位 │ │运河水位 │                         │
# │  └─────────┘ └─────────┘ └─────────┘                         │
# └─────────────────────────────────────────────────────────────────┘
#
# 物理约束：
# - 潮位 → 长江水位 → 运河水位 (因果关系链)
# - 考虑延迟效应 (4-7小时延迟)
# - 当前长江水位反映即时影响
#
# ============================================================================
# 
# 输入特征：
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
# 输出：
# 运河侧未来3小时的水位预测值 (3个输出)
#
# 物理约束：
# - 潮位影响长江水位
# - 长江水位影响运河水位
# - 考虑延迟效应和因果关系
#
# 模型优化：
# - 引入潮位数据提升预测精度
# - LSTM架构捕获时序依赖关系
# - 物理约束优化延迟关系建模

# 修复中文字体问题
def setup_chinese_font():
    """设置中文字体，提供多种字体选择"""
    try:
        # 尝试设置SimHei字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 设置全局图表样式
        plt.style.use('default')
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['savefig.facecolor'] = 'white'
        plt.rcParams['savefig.bbox'] = 'tight'
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['savefig.transparent'] = False
        
        # 设置颜色主题
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
        
        # 测试字体是否可用
        plt.figure(figsize=(1, 1))
        plt.text(0.5, 0.5, '测试中文', fontsize=12)
        plt.close()
        print("中文字体设置成功")
    except Exception as e:
        print(f"中文字体设置失败: {e}")
        print("将使用默认字体")
        # 使用默认字体，避免中文显示问题
        plt.rcParams['font.family'] = 'DejaVu Sans'
        # 强制设置字体大小，避免字体问题
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 10
        plt.rcParams['xtick.labelsize'] = 9
        plt.rcParams['ytick.labelsize'] = 9
        plt.rcParams['legend.fontsize'] = 9

# 初始化字体设置
setup_chinese_font()

def safe_save_plot(plt, filepath, dpi=150, bbox_inches='tight'):
    """
    安全保存图片，包含错误处理和多种保存策略
    
    Args:
        plt: matplotlib图形对象
        filepath: 目标文件路径
        dpi: 图片分辨率
        bbox_inches: 边界设置
    
    Returns:
        bool: 是否保存成功
    """
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 尝试保存
        plt.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)
        print(f"图片保存成功: {filepath}")
        return True
        
    except PermissionError:
        print(f"权限不足，无法保存到: {filepath}")
        # 尝试保存到当前目录
        try:
            filename = os.path.basename(filepath)
            current_path = os.path.join(os.getcwd(), filename)
            plt.savefig(current_path, dpi=dpi, bbox_inches=bbox_inches)
            print(f"图片已保存到当前目录: {current_path}")
            return True
        except Exception as e:
            print(f"保存到当前目录也失败: {e}")
            return False
            
    except Exception as e:
        print(f"保存图片失败: {e}")
        return False

# LSTM模型架构说明：
# 1. 单向LSTM层：处理时序信息，捕获水位变化的时序依赖关系
# 2. 多层设计：2层LSTM提供更强的表达能力，适合复杂的水位变化模式
# 3. 全连接层：将LSTM输出转换为具体的水位预测值
# 4. Dropout正则化：防止过拟合，提高模型泛化能力
# 5. 物理约束优化：输入包含12小时历史数据和延迟潮位特征，重点优化延迟关系建模
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2, output_size=3):
        super().__init__()
        # 单向LSTM核心层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout, bidirectional=False)
        
        # 全连接层处理LSTM的输出
        self.fc1 = nn.Linear(hidden_size, 128)  # LSTM输出维度是hidden_size
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, output_size)
        
    def forward(self, x):
        # LSTM处理序列
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # 取最后一个时间步的输出
        # lstm_out shape: (batch, seq_len, hidden_size)
        last_output = lstm_out[:, -1, :]  # (batch, hidden_size)
        
        # 全连接层进行预测
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

class WaterLevelPredictor:
    def __init__(self, sequence_length=12, prediction_horizon=3):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.scaler_yangtze = MinMaxScaler()
        self.scaler_yunhe = MinMaxScaler()
        self.scaler_tide = MinMaxScaler()  # 潮位数据标准化器
        self.scaler_flow = MinMaxScaler()  # 流量数据标准化器
        self.model = None  # LSTM模型

    def load_data(self):
        print("正在加载水位数据...")
        yangtze_data, yunhe_data = [], []
        script_dir = os.path.dirname(os.path.abspath(__file__))
        for year in [2022, 2023, 2024]:
            for month in range(1, 13):
                if year == 2024:
                    file_path = os.path.join(script_dir, f'../{year}/水位{year}-{month}-1.csv')
                else:
                    file_path = os.path.join(script_dir, f'../{year}/{year}-{month}-1.csv')
                try:
                    if os.path.exists(file_path):
                        try:
                            df = pd.read_csv(file_path, encoding='gbk')
                        except UnicodeDecodeError:
                            try:
                                df = pd.read_csv(file_path, encoding='utf-8')
                            except UnicodeDecodeError:
                                df = pd.read_csv(file_path, encoding='latin1')
                        df_y = df[df['名称'] == '长江侧水位'].copy()
                        df_y['数值'] = pd.to_numeric(df_y['数值'], errors='coerce')
                        df_y = df_y.dropna(subset=['数值'])
                        df_y = df_y[df_y['数值'] >= 0]
                        yangtze_data.append(df_y)
                        df_u = df[df['名称'] == '内河侧水位'].copy()
                        df_u['数值'] = pd.to_numeric(df_u['数值'], errors='coerce')
                        df_u = df_u.dropna(subset=['数值'])
                        df_u = df_u[df_u['数值'] >= 0]
                        yunhe_data.append(df_u)
                except Exception as e:
                    continue
        self.yangtze_df = pd.concat(yangtze_data, ignore_index=True)
        self.yangtze_df['datetime'] = pd.to_datetime(self.yangtze_df['日期'] + ' ' + self.yangtze_df['时间'], format='mixed')
        self.yangtze_df.set_index('datetime', inplace=True)
        self.yangtze_df = self.yangtze_df[['数值']].rename(columns={'数值': 'yangtze_level'}).resample('h').mean().dropna()
        self.yunhe_df = pd.concat(yunhe_data, ignore_index=True)
        self.yunhe_df['datetime'] = pd.to_datetime(self.yunhe_df['日期'] + ' ' + self.yunhe_df['时间'], format='mixed')
        self.yunhe_df.set_index('datetime', inplace=True)
        self.yunhe_df = self.yunhe_df[['数值']].rename(columns={'数值': 'yunhe_level'}).resample('h').mean().dropna()
        self.combined_df = pd.merge(self.yangtze_df, self.yunhe_df, left_index=True, right_index=True, how='inner')
        
        # 加载潮位数据
        print("正在加载潮位数据...")
        self.load_tide_data()
        
        # 加载流量数据
        print("正在加载流量数据...")
        self.load_flow_data()
    
    def load_tide_data(self):
        """加载潮位数据并计算延迟潮位特征"""
        tide_data = []
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 直接加载处理好的潮位数据CSV文件
        for year in [2022, 2023, 2024]:
            tide_file_csv = os.path.join(script_dir, f'../潮位数据处理/{year}年潮位数据_真正小时级.csv')
            if os.path.exists(tide_file_csv):
                try:
                    print(f"正在加载{year}年潮位数据: {tide_file_csv}")
                    df = pd.read_csv(tide_file_csv)
                    
                    # 检查列名并处理
                    if 'datetime' not in df.columns:
                        df['datetime'] = pd.to_datetime(df.iloc[:,0], errors='coerce')
                    else:
                        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
                        
                    if 'tide_level' not in df.columns:
                        df['tide_level'] = pd.to_numeric(df.iloc[:,1], errors='coerce')
                    else:
                        df['tide_level'] = pd.to_numeric(df['tide_level'], errors='coerce')
                    
                    # 清理数据
                    df = df[['datetime', 'tide_level']].dropna()
                    df = df[df['tide_level'].notna()]
                    df.set_index('datetime', inplace=True)
                    df = df.sort_index()
                    
                    print(f"{year}年潮位数据加载成功，数据点数量: {len(df)}")
                    tide_data.append(df)
                    
                except Exception as e:
                    print(f"加载{year}年潮位数据失败: {e}")
                    continue
            else:
                print(f"未找到{year}年潮位数据文件: {tide_file_csv}")
                continue
        
        if tide_data:
            # 检查是否有有效数据
            valid_data = [df for df in tide_data if len(df) > 0]
            if valid_data:
                # 确保所有DataFrame都有正确的列结构
                for df in valid_data:
                    if 'datetime' not in df.columns:
                        df.reset_index(inplace=True)
                
                self.tide_df = pd.concat(valid_data, ignore_index=True)
                self.tide_df = self.tide_df.drop_duplicates('datetime').set_index('datetime').sort_index()
                self.tide_df = self.tide_df.resample('h').interpolate('linear').ffill()
                
                print(f"成功加载潮位数据，数据点数量: {len(self.tide_df)}")
            else:
                print("潮位数据为空，将使用默认值")
        else:
            print("未找到潮位数据文件，将使用默认值")

    def load_flow_data(self):
        """加载节制闸和抽水站流量数据"""
        print("正在加载流量数据...")
        
        # 尝试从合并数据文件中加载流量数据
        script_dir = os.path.dirname(os.path.abspath(__file__))
        merged_file = os.path.join(script_dir, '..', '水位差', 'merged_yangtze_yunhe.csv')
        
        if os.path.exists(merged_file):
            try:
                print(f"从合并文件加载流量数据: {merged_file}")
                merged_df = pd.read_csv(merged_file)
                merged_df['datetime'] = pd.to_datetime(merged_df['datetime'])
                merged_df.set_index('datetime', inplace=True)
                
                # 检查是否有流量列
                if 'pumping_flow' in merged_df.columns and 'net_flow' in merged_df.columns:
                    # 重命名列以保持一致性
                    merged_df = merged_df.rename(columns={
                        'net_flow': 'sluice_flow'  # 节制闸流量
                    })
                    
                    # 选择流量数据
                    flow_data = merged_df[['pumping_flow', 'sluice_flow']].copy()
                    
                    # 处理缺失值
                    flow_data = flow_data.fillna(0)
                    
                    # 合并到主数据框
                    self.combined_df = pd.merge(self.combined_df, flow_data, 
                                              left_index=True, right_index=True, how='left')
                    
                    # 填充缺失的流量数据为0
                    self.combined_df['pumping_flow'] = self.combined_df['pumping_flow'].fillna(0)
                    self.combined_df['sluice_flow'] = self.combined_df['sluice_flow'].fillna(0)
                    
                    print(f"流量数据加载成功，数据点数量: {len(self.combined_df)}")
                    print(f"节制闸流量范围: {self.combined_df['sluice_flow'].min():.2f} - {self.combined_df['sluice_flow'].max():.2f}")
                    print(f"抽水站流量范围: {self.combined_df['pumping_flow'].min():.2f} - {self.combined_df['pumping_flow'].max():.2f}")
                    
                else:
                    print("合并文件中未找到流量列，将使用默认值")
                    self.combined_df['pumping_flow'] = 0
                    self.combined_df['sluice_flow'] = 0
                    
            except Exception as e:
                print(f"加载流量数据失败: {e}")
                print("将使用默认流量值")
                self.combined_df['pumping_flow'] = 0
                self.combined_df['sluice_flow'] = 0
        else:
            print(f"未找到合并数据文件: {merged_file}")
            print("将使用默认流量值")
            self.combined_df['pumping_flow'] = 0
            self.combined_df['sluice_flow'] = 0

    def _calculate_tide_lag_features(self):
        """计算潮位延迟特征"""
        if hasattr(self, 'tide_df') and len(self.tide_df) > 0:
            print("正在计算潮位延迟特征...")
            
            def get_tide_lag(hours):
                """获取指定小时数前的潮位，如果不存在则使用最近的有效值"""
                lag_times = self.combined_df.index - pd.Timedelta(hours=hours)
                lag_values = []
                for lag_time in lag_times:
                    if lag_time in self.tide_df.index:
                        lag_values.append(self.tide_df.loc[lag_time, 'tide_level'])
                    else:
                        # 如果不存在，找到最近的有效时间点
                        nearest_time = self.tide_df.index.get_indexer([lag_time], method='nearest')[0]
                        if nearest_time >= 0:
                            lag_values.append(self.tide_df.iloc[nearest_time]['tide_level'])
                        else:
                            lag_values.append(0)  # 如果都找不到，使用0
                return lag_values
            
            # 计算延迟潮位特征（5-8小时）
            self.combined_df['tide_lag_5h'] = get_tide_lag(5)
            self.combined_df['tide_lag_6h'] = get_tide_lag(6)
            self.combined_df['tide_lag_7h'] = get_tide_lag(7)
            self.combined_df['tide_lag_8h'] = get_tide_lag(8)
            
            print("潮位延迟特征计算完成")
        else:
            # 如果没有潮位数据，创建默认列（5-8小时）
            print("未找到潮位数据，创建默认潮位延迟特征")
            self.combined_df['tide_lag_5h'] = 0
            self.combined_df['tide_lag_6h'] = 0
            self.combined_df['tide_lag_7h'] = 0
            self.combined_df['tide_lag_8h'] = 0

    def prepare_data(self):
        """
        准备训练数据：构建输入特征和输出标签
        
        输入特征结构：
        - 长江侧历史12小时水位 (12个特征)
        - 运河侧历史12小时水位 (12个特征)  
        - 延迟潮位特征：5、6、7、8小时前的潮位 (4个特征)
        - 当前节制闸流量 (1个特征)
        - 当前抽水站流量 (1个特征)
        总计：30个输入特征
        
        输出标签：
        - 运河侧未来3小时的水位预测 (3个输出)
        """
        # 首先计算潮位延迟特征
        self._calculate_tide_lag_features()
        
        yangtze_scaled = self.scaler_yangtze.fit_transform(self.combined_df[['yangtze_level']])
        yunhe_scaled = self.scaler_yunhe.fit_transform(self.combined_df[['yunhe_level']])
        
        # 标准化潮位数据（延迟潮位特征：5、6、7、8小时前）
        tide_features = ['tide_lag_5h', 'tide_lag_6h', 'tide_lag_7h', 'tide_lag_8h']
        
        tide_data = self.combined_df[tide_features].copy()
        if tide_data.isna().any().any():
            print("检测到潮位数据包含NaN值，将使用0填充")
            tide_data = tide_data.fillna(0)
        
        tide_scaled = self.scaler_tide.fit_transform(tide_data)
        
        # 标准化流量数据
        flow_data = self.combined_df[['pumping_flow', 'sluice_flow']].copy()
        flow_scaled = self.scaler_flow.fit_transform(flow_data)
        
        X, y = [], []
        for i in range(self.sequence_length, len(self.combined_df) - self.prediction_horizon + 1):
            # 构建输入特征：长江历史水位 + 运河历史水位 + 延迟潮位特征 + 流量特征
            yangtze_history = yangtze_scaled[i-self.sequence_length:i]  # 长江侧过去12小时
            yunhe_history = yunhe_scaled[i-self.sequence_length:i]      # 运河侧过去12小时
            current_tide = tide_scaled[i]                               # 当前时刻的延迟潮位特征
            current_flow = flow_scaled[i]                               # 当前时刻的流量特征
            
            # 拼接所有特征：长江历史水位 + 运河历史水位 + 潮位延迟特征 + 流量特征
            features = np.concatenate([
                yangtze_history.flatten(),  # 12个特征：长江侧过去12小时水位
                yunhe_history.flatten(),    # 12个特征：运河侧过去12小时水位
                current_tide.flatten(),     # 4个特征：5、6、7、8小时前的潮位
                current_flow.flatten()      # 2个特征：当前节制闸和抽水站流量
            ])
            
            X.append(features)
            # 输出标签：运河侧未来3小时的水位
            y.append(yunhe_scaled[i:i+self.prediction_horizon].flatten())
        
        X = np.array(X)
        y = np.array(y)
        X = X.reshape(X.shape[0], 1, X.shape[1])
        split_idx = int(0.8 * len(X))
        self.X_train, self.X_test = X[:split_idx], X[split_idx:]
        self.y_train, self.y_test = y[:split_idx], y[split_idx:]

    def build_model(self):
        """
        构建LSTM模型
        
        输入特征维度：30维
        - 长江侧历史12小时水位：12维
        - 运河侧历史12小时水位：12维  
        - 延迟潮位特征：4维 (5、6、7、8小时前的潮位)
        - 流量特征：2维 (当前节制闸和抽水站流量)
        
        输出维度：3维 (未来3小时的运河水位预测)
        """
        input_features = self.sequence_length + self.sequence_length + 4 + 2  # 12 + 12 + 4 + 2 = 30个特征
        self.model = LSTMModel(input_size=input_features, output_size=self.prediction_horizon)

    def train_model(self, epochs=100, batch_size=32):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        X_train_t = torch.tensor(self.X_train, dtype=torch.float32).to(device)
        y_train_t = torch.tensor(self.y_train, dtype=torch.float32).to(device)
        X_test_t = torch.tensor(self.X_test, dtype=torch.float32).to(device)
        y_test_t = torch.tensor(self.y_test, dtype=torch.float32).to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        train_loss_list = []
        val_loss_list = []
        for epoch in range(epochs):
            self.model.train()
            permutation = torch.randperm(X_train_t.size(0))
            train_loss = 0
            for i in range(0, X_train_t.size(0), batch_size):
                indices = permutation[i:i+batch_size]
                batch_x, batch_y = X_train_t[indices], y_train_t[indices]
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * batch_x.size(0)
            train_loss /= X_train_t.size(0)
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_test_t)
                val_loss = criterion(val_outputs, y_test_t).item()
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            if (epoch+1) % 10 == 0 or epoch == 0:
                print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.5f}, Val Loss: {val_loss:.5f}')
        self.train_loss_list = train_loss_list
        self.val_loss_list = val_loss_list
        self.X_test_t = X_test_t
        self.y_test_t = y_test_t
        self.device = device

    def evaluate_and_plot(self):
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(self.X_test_t).cpu().numpy()
        y_test_original = self.scaler_yunhe.inverse_transform(self.y_test_t.cpu().numpy())
        y_pred_original = self.scaler_yunhe.inverse_transform(y_pred)
        
        # 计算每个时间步的指标
        mse_list = []
        mae_list = []
        r2_list = []
        
        for i in range(self.prediction_horizon):
            mse = mean_squared_error(y_test_original[:, i], y_pred_original[:, i])
            mae = mean_absolute_error(y_test_original[:, i], y_pred_original[:, i])
            r2 = r2_score(y_test_original[:, i], y_pred_original[:, i])
            mse_list.append(mse)
            mae_list.append(mae)
            r2_list.append(r2)
            print(f"第{i+1}小时预测 - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
        
        # 计算总体指标
        overall_mse = np.mean(mse_list)
        overall_mae = np.mean(mae_list)
        overall_r2 = np.mean(r2_list)
        print(f"\n总体预测性能:")
        print(f"平均MSE: {overall_mse:.4f}")
        print(f"平均MAE: {overall_mae:.4f}")
        print(f"平均R2: {overall_r2:.4f}")
        
        # 创建输出文件夹并验证权限
        output_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '预测结果')
        detailed_analysis_folder = os.path.join(output_folder, '详细分析图表')
        
        try:
            os.makedirs(output_folder, exist_ok=True)
            os.makedirs(detailed_analysis_folder, exist_ok=True)
            # 测试文件夹写入权限
            test_file = os.path.join(detailed_analysis_folder, 'test_write.txt')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            print(f"输出文件夹创建成功: {output_folder}")
            print(f"详细分析图表文件夹创建成功: {detailed_analysis_folder}")
        except Exception as e:
            print(f"创建输出文件夹失败: {e}")
            # 尝试使用当前目录
            output_folder = '预测结果'
            detailed_analysis_folder = '详细分析图表'
            os.makedirs(output_folder, exist_ok=True)
            os.makedirs(detailed_analysis_folder, exist_ok=True)
            print(f"使用当前目录作为输出文件夹: {output_folder}")
        
        # 保存损失曲线
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_loss_list, label='Training Loss', color='#1f77b4', linewidth=2.5, alpha=0.8)
        plt.plot(self.val_loss_list, label='Validation Loss', color='#ff7f0e', linewidth=2.5, alpha=0.8)
        plt.legend(fontsize=12, loc='upper right', frameon=True, fancybox=True, shadow=True)
        plt.title('LSTM Model Training Loss Curve', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Training Epochs', fontsize=14, fontweight='bold')
        plt.ylabel('Loss Value', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        
        # 设置背景样式
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_linewidth(1.5)
        plt.gca().spines['bottom'].set_linewidth(1.5)
        
        # 添加网格背景
        plt.gca().set_facecolor('#f8f9fa')
        
        plt.tight_layout()
        loss_path = os.path.join(output_folder, 'train_loss.png')
        safe_save_plot(plt, loss_path)
        plt.close()
        
        # 保存预测对比图
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        hours = ['1st Hour', '2nd Hour', '3rd Hour']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for i in range(3):
            # 绘制真实值和预测值
            axes[i].plot(y_test_original[:, i], label='True Values', color=colors[i], linewidth=2, alpha=0.8)
            axes[i].plot(y_pred_original[:, i], label='Predicted Values', color='#d62728', linewidth=2, alpha=0.8)
            
            # 设置子图样式
            axes[i].set_title(f'{hours[i]} Canal Water Level Prediction Comparison', fontsize=14, fontweight='bold', pad=15)
            axes[i].set_xlabel('Sample Index', fontsize=12, fontweight='bold')
            axes[i].set_ylabel('Water Level (m)', fontsize=12, fontweight='bold')
            axes[i].legend(fontsize=11, loc='upper right', frameon=True, fancybox=True, shadow=True)
            axes[i].grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
            axes[i].tick_params(axis='both', which='major', labelsize=10)
            
            # 设置背景样式
            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)
            axes[i].spines['left'].set_linewidth(1.2)
            axes[i].spines['bottom'].set_linewidth(1.2)
            
            # 添加网格背景
            axes[i].set_facecolor('#f8f9fa')
            
            # 添加统计信息文本框
            mse = mean_squared_error(y_test_original[:, i], y_pred_original[:, i])
            mae = mean_absolute_error(y_test_original[:, i], y_pred_original[:, i])
            r2 = r2_score(y_test_original[:, i], y_pred_original[:, i])
            
            textstr = f'MSE: {mse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}'
            props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray')
            axes[i].text(0.02, 0.98, textstr, transform=axes[i].transAxes, fontsize=10,
                        verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        pred_path = os.path.join(output_folder, 'test_pred_vs_true.png')
        safe_save_plot(plt, pred_path)
        plt.close()
        
        # 新增：预测误差分布直方图
        plt.figure(figsize=(15, 5))
        for i in range(3):
            plt.subplot(1, 3, i+1)
            errors = y_pred_original[:, i] - y_test_original[:, i]
            plt.hist(errors, bins=30, color=colors[i], alpha=0.7, edgecolor='black', linewidth=0.5)
            plt.title(f'{hours[i]} Prediction Error Distribution', fontsize=14, fontweight='bold')
            plt.xlabel('Prediction Error (m)', fontsize=12, fontweight='bold')
            plt.ylabel('Frequency', fontsize=12, fontweight='bold')
            plt.grid(True, alpha=0.3)
            
            # 添加统计信息
            mean_error = np.mean(errors)
            std_error = np.std(errors)
            plt.axvline(mean_error, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_error:.3f}')
            plt.axvline(mean_error + std_error, color='orange', linestyle=':', linewidth=2, label=f'+1σ: {mean_error + std_error:.3f}')
            plt.axvline(mean_error - std_error, color='orange', linestyle=':', linewidth=2, label=f'-1σ: {mean_error - std_error:.3f}')
            plt.legend(fontsize=10)
            
            # 设置背景样式
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().set_facecolor('#f8f9fa')
        
        plt.tight_layout()
        error_hist_path = os.path.join(detailed_analysis_folder, 'prediction_error_distribution.png')
        safe_save_plot(plt, error_hist_path)
        plt.close()
        
        # 新增：预测误差时间序列图
        plt.figure(figsize=(15, 5))
        for i in range(3):
            plt.subplot(1, 3, i+1)
            errors = y_pred_original[:, i] - y_test_original[:, i]
            plt.plot(errors, color=colors[i], linewidth=1.5, alpha=0.8)
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            plt.title(f'{hours[i]} Prediction Error Time Series', fontsize=14, fontweight='bold')
            plt.xlabel('Sample Index', fontsize=12, fontweight='bold')
            plt.ylabel('Prediction Error (m)', fontsize=12, fontweight='bold')
            plt.grid(True, alpha=0.3)
            
            # 添加误差带
            mean_error = np.mean(errors)
            std_error = np.std(errors)
            plt.fill_between(range(len(errors)), mean_error - std_error, mean_error + std_error, 
                           alpha=0.2, color=colors[i], label=f'±1σ Range')
            plt.legend(fontsize=10)
            
            # 设置背景样式
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().set_facecolor('#f8f9fa')
        
        plt.tight_layout()
        error_time_path = os.path.join(detailed_analysis_folder, 'prediction_error_timeseries.png')
        safe_save_plot(plt, error_time_path)
        plt.close()
        
        # 新增：预测值vs真实值散点图
        plt.figure(figsize=(15, 5))
        for i in range(3):
            plt.subplot(1, 3, i+1)
            plt.scatter(y_test_original[:, i], y_pred_original[:, i], alpha=0.6, color=colors[i], s=30)
            
            # 添加理想预测线
            min_val = min(y_test_original[:, i].min(), y_pred_original[:, i].min())
            max_val = max(y_test_original[:, i].max(), y_pred_original[:, i].max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.8, label='Ideal Prediction Line')
            
            plt.title(f'{hours[i]} Predicted vs True Values', fontsize=14, fontweight='bold')
            plt.xlabel('True Values (m)', fontsize=12, fontweight='bold')
            plt.ylabel('Predicted Values (m)', fontsize=12, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=10)
            
            # 添加R²值
            r2 = r2_score(y_test_original[:, i], y_pred_original[:, i])
            plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes, 
                    fontsize=12, fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # 设置背景样式
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().set_facecolor('#f8f9fa')
        
        plt.tight_layout()
        scatter_path = os.path.join(detailed_analysis_folder, 'predicted_vs_true_scatter.png')
        safe_save_plot(plt, scatter_path)
        plt.close()
        
        # 新增：月度预测效果对比图
        if hasattr(self, 'combined_df') and len(self.combined_df) > 0:
            try:
                # 获取测试数据的月份信息
                test_indices = range(self.sequence_length, len(self.combined_df) - self.prediction_horizon + 1)
                test_indices = test_indices[int(0.8 * len(test_indices)):]  # 取测试集部分
                
                if len(test_indices) > 0:
                    test_times = self.combined_df.index[test_indices]
                    test_months = [t.month for t in test_times]
                    
                    # 计算每个月的平均误差
                    monthly_errors = {}
                    for i in range(3):  # 3个预测时间步
                        errors = y_pred_original[:, i] - y_test_original[:, i]
                        for month, error in zip(test_months, errors):
                            if month not in monthly_errors:
                                monthly_errors[month] = []
                            monthly_errors[month].append(abs(error))
                    
                    # 绘制月度误差对比
                    months = sorted(monthly_errors.keys())
                    month_names = ['1月', '2月', '3月', '4月', '5月', '6月', 
                                 '7月', '8月', '9月', '10月', '11月', '12月']
                    
                    plt.figure(figsize=(12, 6))
                    x_pos = np.arange(len(months))
                    width = 0.25
                    
                    for i in range(3):
                        monthly_mae = [np.mean(monthly_errors[month]) for month in months]
                        plt.bar(x_pos + i*width, monthly_mae, width, label=f'{hours[i]}', 
                               color=colors[i], alpha=0.8)
                    
                    plt.xlabel('Month', fontsize=14, fontweight='bold')
                    plt.ylabel('Mean Absolute Error (m)', fontsize=12, fontweight='bold')
                    plt.title('Monthly Prediction Performance Comparison', fontsize=16, fontweight='bold', pad=20)
                    plt.xticks(x_pos + width, [month_names[m-1] for m in months], fontsize=12)
                    plt.legend(fontsize=12, loc='upper right')
                    plt.grid(True, alpha=0.3, axis='y')
                    
                    # 设置背景样式
                    plt.gca().spines['top'].set_visible(False)
                    plt.gca().spines['right'].set_visible(False)
                    plt.gca().set_facecolor('#f8f9fa')
                    
                    plt.tight_layout()
                    monthly_path = os.path.join(detailed_analysis_folder, 'monthly_prediction_comparison.png')
                    safe_save_plot(plt, monthly_path)
                    plt.close()
            except Exception as e:
                print(f"生成月度预测效果对比图失败: {e}")
        
        # 新增：模型性能总结图
        plt.figure(figsize=(12, 8))
        
        # 创建子图
        plt.subplot(2, 2, 1)
        plt.bar(hours, mse_list, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        plt.title('MSE Comparison by Time Step', fontsize=14, fontweight='bold')
        plt.ylabel('MSE', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.subplot(2, 2, 2)
        plt.bar(hours, mae_list, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        plt.title('MAE Comparison by Time Step', fontsize=14, fontweight='bold')
        plt.ylabel('MAE (m)', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.subplot(2, 2, 3)
        plt.bar(hours, r2_list, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        plt.title('R² Comparison by Time Step', fontsize=14, fontweight='bold')
        plt.ylabel('R²', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.subplot(2, 2, 4)
        # 绘制训练过程
        plt.plot(self.train_loss_list, label='Training Loss', color='#1f77b4', linewidth=2)
        plt.plot(self.val_loss_list, label='Validation Loss', color='#ff7f0e', linewidth=2)
        plt.title('Training Process', fontsize=14, fontweight='bold')
        plt.xlabel('Training Epochs', fontsize=12, fontweight='bold')
        plt.ylabel('Loss Value', fontsize=12, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # 设置背景样式
        for ax in plt.gcf().axes:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_facecolor('#f8f9fa')
        
        plt.tight_layout()
        summary_path = os.path.join(detailed_analysis_folder, 'model_performance_summary.png')
        safe_save_plot(plt, summary_path)
        plt.close()
        
        # 保存模型参数
        model_path = os.path.join(output_folder, 'lstm_yunhe_model.pth')
        torch.save(self.model.state_dict(), model_path)
        print(f'LSTM模型参数已保存到 {model_path}')
        
        # 保存预测结果
        results_df = pd.DataFrame({
            '真实值_第1小时': y_test_original[:, 0],
            '预测值_第1小时': y_pred_original[:, 0],
            '真实值_第2小时': y_test_original[:, 1],
            '预测值_第2小时': y_pred_original[:, 1],
            '真实值_第3小时': y_test_original[:, 2],
            '预测值_第3小时': y_pred_original[:, 2]
        })
        results_path = os.path.join(output_folder, '预测结果.csv')
        results_df.to_csv(results_path, index=False)
        print(f'预测结果已保存到 {results_path}')
        
        # 保存详细分析结果
        detailed_results_df = pd.DataFrame({
            '时间步': hours,
            'MSE': mse_list,
            'MAE': mae_list,
            'R²': r2_list
        })
        detailed_results_path = os.path.join(detailed_analysis_folder, '详细分析结果.csv')
        detailed_results_df.to_csv(detailed_results_path, index=False)
        print(f'详细分析结果已保存到 {detailed_results_path}')
        
        print(f"\n=== 图表生成完成 ===")
        print(f"基础图表保存在: {output_folder}")
        print(f"详细分析图表保存在: {detailed_analysis_folder}")
        print(f"共生成 {len([f for f in os.listdir(detailed_analysis_folder) if f.endswith('.png')])} 个详细分析图表")
    
    def predict_future(self, yangtze_history, yunhe_history, tide_lag_5h=None, tide_lag_6h=None, tide_lag_7h=None, tide_lag_8h=None, pumping_flow=0, sluice_flow=0):
        """
        预测未来3小时的运河侧水位
        
        Args:
            yangtze_history: 长江侧过去12小时的水位数据 (12个值)
            yunhe_history: 运河侧过去12小时的水位数据 (12个值)
            tide_lag_5h: 5小时前的潮位数据
            tide_lag_6h: 6小时前的潮位数据
            tide_lag_7h: 7小时前的潮位数据
            tide_lag_8h: 8小时前的潮位数据
            pumping_flow: 当前抽水站流量
            sluice_flow: 当前节制闸流量
        
        Returns:
            predicted_levels: 未来3小时的运河侧水位预测值 (3个值)
        """
        if len(yangtze_history) != self.sequence_length:
            raise ValueError(f"长江历史数据长度必须为{self.sequence_length}小时")
        if len(yunhe_history) != self.sequence_length:
            raise ValueError(f"运河历史数据长度必须为{self.sequence_length}小时")
        
        # 数据预处理
        yangtze_scaled = self.scaler_yangtze.transform(np.array(yangtze_history).reshape(-1, 1))
        yunhe_scaled = self.scaler_yunhe.transform(np.array(yunhe_history).reshape(-1, 1))
        
        # 处理潮位数据（5-8小时）
        if (tide_lag_5h is not None and tide_lag_6h is not None and 
            tide_lag_7h is not None and tide_lag_8h is not None):
            tide_data = np.array([[tide_lag_5h, tide_lag_6h, tide_lag_7h, tide_lag_8h]])
            tide_scaled = self.scaler_tide.transform(tide_data)
        else:
            tide_scaled = np.array([[0.0, 0.0, 0.0, 0.0]])
        
        # 处理流量数据
        flow_data = np.array([[pumping_flow, sluice_flow]])
        flow_scaled = self.scaler_flow.transform(flow_data)
        
        # 构建输入特征（包含潮位特征和流量特征）
        features = np.concatenate([
            yangtze_scaled.flatten(),          # 12个特征：长江侧过去12小时水位
            yunhe_scaled.flatten(),            # 12个特征：内河侧过去12小时水位
            tide_scaled.flatten(),             # 4个特征：5、6、7、8小时前的潮位
            flow_scaled.flatten()              # 2个特征：当前节制闸和抽水站流量
        ])
        
        # 预测
        self.model.eval()
        with torch.no_grad():
            input_tensor = torch.tensor(features.reshape(1, 1, -1), dtype=torch.float32).to(self.device)
            prediction = self.model(input_tensor).cpu().numpy()
        
        # 反归一化
        predicted_levels = self.scaler_yunhe.inverse_transform(prediction.reshape(-1, 1)).flatten()
        
        return predicted_levels
    
    def analyze_delay_impact(self):
        """
        分析长江水位对运河水位延迟影响的效果
        """
        print("正在分析长江水位对运河水位延迟影响...")
        
        # 计算长江水位变化与运河水位变化的相关性
        yangtze_changes = self.combined_df['yangtze_level'].diff()
        yunhe_changes = self.combined_df['yunhe_level'].diff()
        
        # 计算不同延迟时间的相关性
        delays = range(0, 7)  # 0到6小时延迟
        correlations = []
        
        for delay in delays:
            if delay == 0:
                corr = yangtze_changes.corr(yunhe_changes)
            else:
                # 长江水位变化延迟delay小时后与运河水位变化的相关性
                yangtze_delayed = yangtze_changes.shift(delay)
                corr = yangtze_delayed.corr(yunhe_changes)
            correlations.append(corr)
        
        # 找到最佳延迟时间
        best_delay = delays[np.nanargmax(np.abs(correlations))]
        best_corr = correlations[best_delay]
        
        print(f"长江水位对运河水位影响的最佳延迟时间: {best_delay}小时")
        print(f"最佳延迟时间的相关系数: {best_corr:.4f}")
        
        # 绘制延迟相关性分析图
        plt.figure(figsize=(12, 8))
        
        # 创建渐变色彩
        colors = plt.cm.viridis(np.linspace(0, 1, len(delays)))
        
        # 绘制相关性曲线
        plt.plot(delays, correlations, 'o-', linewidth=3, markersize=10, 
                color='#1f77b4', markerfacecolor='white', markeredgecolor='#1f77b4', 
                markeredgewidth=2, alpha=0.8)
        
        # 标记最佳延迟点
        plt.scatter(best_delay, best_corr, s=200, color='#d62728', 
                   edgecolors='black', linewidth=2, zorder=5, 
                   label=f'Best Delay: {best_delay}h')
        
        # 添加最佳延迟线
        plt.axvline(x=best_delay, color='#d62728', linestyle='--', 
                   linewidth=2, alpha=0.7, label=f'Best Delay Time: {best_delay}h')
        
        # 设置图表样式
        plt.xlabel('Delay Time (hours)', fontsize=14, fontweight='bold')
        plt.ylabel('Correlation Coefficient', fontsize=14, fontweight='bold')
        plt.title('Delay Impact Analysis: Yangtze Water Level on Canal Water Level', fontsize=16, fontweight='bold', pad=20)
        plt.legend(fontsize=12, loc='upper right', frameon=True, fancybox=True, shadow=True)
        plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        plt.xticks(delays, fontsize=12)
        plt.yticks(fontsize=12)
        
        # 设置背景样式
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_linewidth(1.5)
        plt.gca().spines['bottom'].set_linewidth(1.5)
        
        # 添加网格背景
        plt.gca().set_facecolor('#f8f9fa')
        
        # 添加统计信息文本框
        textstr = f'Best Delay: {best_delay}h\nCorrelation: {best_corr:.4f}'
        props = dict(boxstyle='round,pad=0.8', facecolor='white', alpha=0.9, 
                    edgecolor='#d62728', linewidth=2)
        plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=12,
                verticalalignment='top', bbox=props, fontweight='bold')
        
        plt.tight_layout()
        
        # 保存延迟影响分析图
        output_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '预测结果')
        delay_analysis_path = os.path.join(output_folder, 'yangtze_delay_impact_analysis.png')
        safe_save_plot(plt, delay_analysis_path)
        plt.close()
        
        return best_delay, best_corr

def test_plot_saving():
    """测试图片保存功能是否正常工作"""
    print("正在测试图片保存功能...")
    
    try:
        # 创建测试图片
        plt.figure(figsize=(8, 6))
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        
        # 绘制测试曲线
        plt.plot(x, y, label='Test Curve', color='#1f77b4', linewidth=3, alpha=0.8)
        
        # 设置图表样式
        plt.title('Image Saving Function Test', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('X Axis', fontsize=14, fontweight='bold')
        plt.ylabel('Y Axis', fontsize=14, fontweight='bold')
        plt.legend(fontsize=12, loc='upper right', frameon=True, fancybox=True, shadow=True)
        plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        
        # 设置背景样式
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_linewidth(1.5)
        plt.gca().spines['bottom'].set_linewidth(1.5)
        
        # 添加网格背景
        plt.gca().set_facecolor('#f8f9fa')
        
        # 设置刻度标签
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        
        plt.tight_layout()
        
        # 测试保存到不同位置
        test_paths = [
            'test_plot.png',  # 当前目录
            os.path.join('预测结果', 'test_plot.png'),  # 预测结果文件夹
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '预测结果', 'test_plot.png')  # 绝对路径
        ]
        
        success_count = 0
        for test_path in test_paths:
            if safe_save_plot(plt, test_path):
                success_count += 1
                # 清理测试文件
                try:
                    if os.path.exists(test_path):
                        os.remove(test_path)
                        print(f"测试文件已清理: {test_path}")
                except:
                    pass
        
        plt.close()
        
        if success_count > 0:
            print(f"图片保存测试成功！{success_count}/{len(test_paths)} 个位置可以正常保存")
            return True
        else:
            print("图片保存测试失败！所有位置都无法保存")
            return False
            
    except Exception as e:
        print(f"图片保存测试出错: {e}")
        return False

def main():
    # 首先测试图片保存功能
    if not test_plot_saving():
        print("警告：图片保存功能测试失败，可能影响后续图表输出")
        print("建议检查文件权限和matplotlib配置")
    
    predictor = WaterLevelPredictor(sequence_length=12, prediction_horizon=3)
    predictor.load_data()
    
    # 分析长江水位对运河水位延迟影响
    best_delay, best_corr = predictor.analyze_delay_impact()
    
    predictor.prepare_data()
    predictor.build_model()
    predictor.train_model(epochs=100, batch_size=32)
    predictor.evaluate_and_plot()
    
    # 展示模型输入输出结构
    print(f"\n=== 模型输入输出结构 ===")
    print(f"输入特征 (总计30维):")
    print(f"  - 长江侧历史12小时水位: 12维")
    print(f"  - 运河侧历史12小时水位: 12维")
    print(f"  - 延迟潮位特征: 4维 (5、6、7、8小时前的潮位)")
    print(f"  - 流量特征: 2维 (当前节制闸和抽水站流量)")
    print(f"输出: 运河侧未来3小时的水位预测 (3维)")
    
    # 展示潮位特征的使用示例
    print(f"\n=== 潮位特征使用示例 ===")
    print(f"模型现在包含以下延迟潮位特征:")
    print(f"- 5小时前的潮位值")
    print(f"- 6小时前的潮位值")
    print(f"- 7小时前的潮位值") 
    print(f"- 8小时前的潮位值")
    
    # 展示流量特征的使用示例
    print(f"\n=== 流量特征使用示例 ===")
    print(f"模型现在包含以下流量特征:")
    print(f"- 当前抽水站流量")
    print(f"- 当前节制闸流量")
    
    # 示例：使用最新数据进行预测
    if len(predictor.combined_df) >= 12:
        print(f"\n=== 使用最新数据进行预测示例 ===")
        # 获取最新的12小时数据
        latest_data = predictor.combined_df.tail(12)
        
        # 长江水位历史数据（最新12小时）
        yangtze_history = latest_data['yangtze_level'].values
        # 运河水位历史数据（最新12小时）
        yunhe_history = latest_data['yunhe_level'].values
        
        print(f"长江历史水位范围: {yangtze_history.min():.3f}m - {yangtze_history.max():.3f}m")
        print(f"运河历史水位范围: {yunhe_history.min():.3f}m - {yunhe_history.max():.3f}m")
        
        # 获取对应的潮位数据（如果有的话）
        if 'tide_lag_5h' in latest_data.columns:
            latest_tide_5h = latest_data['tide_lag_5h'].iloc[-1]
            latest_tide_6h = latest_data['tide_lag_6h'].iloc[-1]
            latest_tide_7h = latest_data['tide_lag_7h'].iloc[-1]
            latest_tide_8h = latest_data['tide_lag_8h'].iloc[-1]
            
            print(f"最新延迟潮位数据:")
            print(f"  5小时前: {latest_tide_5h:.3f}m")
            print(f"  6小时前: {latest_tide_6h:.3f}m")
            print(f"  7小时前: {latest_tide_7h:.3f}m")
            print(f"  8小时前: {latest_tide_8h:.3f}m")
            
            # 获取流量数据
            if 'pumping_flow' in latest_data.columns and 'sluice_flow' in latest_data.columns:
                latest_pumping_flow = latest_data['pumping_flow'].iloc[-1]
                latest_sluice_flow = latest_data['sluice_flow'].iloc[-1]
                
                print(f"最新流量数据:")
                print(f"  抽水站流量: {latest_pumping_flow:.3f}m³/s")
                print(f"  节制闸流量: {latest_sluice_flow:.3f}m³/s")
                
                try:
                    predicted_levels = predictor.predict_future(
                        yangtze_history, yunhe_history,
                        tide_lag_5h=latest_tide_5h, tide_lag_6h=latest_tide_6h,
                        tide_lag_7h=latest_tide_7h, tide_lag_8h=latest_tide_8h,
                        pumping_flow=latest_pumping_flow, sluice_flow=latest_sluice_flow
                    )
                    
                    print(f"\n未来3小时运河水位预测:")
                    for i, level in enumerate(predicted_levels, 1):
                        print(f"  第{i}小时: {level:.3f}m")
                        
                except Exception as e:
                    print(f"预测失败: {e}")
            else:
                print("未找到流量数据，将使用默认值进行预测")
                try:
                    predicted_levels = predictor.predict_future(
                        yangtze_history, yunhe_history,
                        tide_lag_5h=latest_tide_5h, tide_lag_6h=latest_tide_6h,
                        tide_lag_7h=latest_tide_7h, tide_lag_8h=latest_tide_8h
                    )
                    
                    print(f"\n未来3小时运河水位预测:")
                    for i, level in enumerate(predicted_levels, 1):
                        print(f"  第{i}小时: {level:.3f}m")
                        
                except Exception as e:
                    print(f"预测失败: {e}")
        else:
            print("未找到潮位数据，无法进行包含潮位特征的预测")
    
    print(f"\n=== 模型优化总结 ===")
    print(f"1. 考虑了长江水位对运河水位{best_delay}小时的延迟影响")
    print(f"2. 延迟影响相关系数: {best_corr:.4f}")
    print(f"3. 新增延迟潮位特征：5、6、7、8小时前的潮位值")
    print(f"4. 输入特征维度：{predictor.sequence_length + predictor.sequence_length + 4 + 2}维 (12+12+4+2=30维)")
    print(f"5. 输出维度：{predictor.prediction_horizon}维 (未来3小时预测)")
    print(f"6. LSTM架构能更好地捕获潮位-长江水位-运河水位的因果关系")
    print(f"7. 新增流量特征，考虑节制闸和抽水站对水位的影响")
    print(f"8. 通过延迟影响分析验证物理约束的合理性")
    print(f"9. 简化长江水位输入：只使用当前长江水位，减少模型复杂度")

if __name__ == "__main__":
    # 检查命令行参数
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--test-plot':
        # 只测试图片保存功能
        print("=== 图片保存功能测试模式 ===")
        test_plot_saving()
    else:
        # 正常运行完整程序
        main() 
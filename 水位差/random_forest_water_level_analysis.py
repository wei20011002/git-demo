# ============================================================================
# 随机森林水位差影响因素分析
# ============================================================================
#
# 功能说明：
# 使用随机森林回归模型分析影响水位差的关键因素
# 包括特征重要性分析、相关性分析、预测性能评估等
#
# 分析目标：
# 1. 识别影响水位差的关键因素
# 2. 评估各因素的相对重要性
# 3. 构建水位差预测模型
# 4. 分析特征之间的相互作用
#
# ============================================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 数据路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MERGED_DATA_FILE = os.path.join(BASE_DIR, 'merged_yangtze_yunhe.csv')
OUTPUT_FOLDER = os.path.join(BASE_DIR, '随机森林分析结果')

class WaterLevelDifferenceAnalyzer:
    """水位差影响因素分析器"""
    
    def __init__(self):
        self.data = None
        self.features = None
        self.target = None
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def load_and_prepare_data(self):
        """加载和准备数据"""
        print("正在加载数据...")
        
        if not os.path.exists(MERGED_DATA_FILE):
            raise FileNotFoundError(f"数据文件不存在: {MERGED_DATA_FILE}")
        
        # 加载数据
        self.data = pd.read_csv(MERGED_DATA_FILE)
        print(f"原始数据点数: {len(self.data)}")
        
        # 处理时间列
        self.data['datetime'] = pd.to_datetime(self.data['datetime'])
        self.data.set_index('datetime', inplace=True)
        
        # 重命名列
        self.data = self.data.rename(columns={
            '长江侧水位': 'yangtze_level',
            '内河侧水位': 'yunhe_level',
            'pumping_flow': 'pumping_flow',
            'net_flow': 'sluice_flow'
        })
        
        # 计算水位差
        self.data['water_level_diff'] = np.abs(self.data['yangtze_level'] - self.data['yunhe_level'])
        
        # 移除包含NaN的行
        self.data = self.data.dropna()
        print(f"处理后数据点数: {len(self.data)}")
        
        return self.data
    
    def create_features(self):
        """创建特征工程 - 使用与LSTM模型相同的30维输入特征"""
        print("正在创建特征...")
        
        df = self.data.copy()
        
        # 创建与LSTM模型完全相同的30维特征
        features_list = []
        
        # 1. 长江侧历史12小时水位 (12维)
        for i in range(1, 13):
            df[f'长江水位_{i}小时前'] = df['yangtze_level'].shift(i)
            features_list.append(f'长江水位_{i}小时前')
        
        # 2. 运河侧历史12小时水位 (12维)
        for i in range(1, 13):
            df[f'运河水位_{i}小时前'] = df['yunhe_level'].shift(i)
            features_list.append(f'运河水位_{i}小时前')
        
        # 3. 延迟潮位特征 (4维) - 5、6、7、8小时前的潮位
        if 'tide_level' in df.columns:
            for lag in [5, 6, 7, 8]:
                df[f'潮位_{lag}小时前'] = df['tide_level'].shift(lag)
                features_list.append(f'潮位_{lag}小时前')
        else:
            # 如果没有潮位数据，创建默认列
            for lag in [5, 6, 7, 8]:
                df[f'潮位_{lag}小时前'] = 0
                features_list.append(f'潮位_{lag}小时前')
        
        # 4. 流量特征 (2维) - 当前节制闸和抽水站流量
        df['当前节制闸流量'] = df['sluice_flow']
        df['当前抽水站流量'] = df['pumping_flow']
        features_list.extend(['当前节制闸流量', '当前抽水站流量'])
        
        # 移除包含NaN的行
        df = df.dropna()
        print(f"特征工程后数据点数: {len(df)}")
        
        # 选择特征和目标变量
        self.features = df[features_list]
        self.target = df['water_level_diff']
        self.feature_names = features_list
        
        print(f"特征数量: {len(features_list)} (与LSTM模型输入维度一致)")
        print(f"目标变量: water_level_diff")
        print(f"特征列表: {features_list}")
        
        return df
    
    def train_random_forest(self):
        """训练随机森林模型"""
        print("正在训练随机森林模型...")
        
        # 数据标准化
        X_scaled = self.scaler.fit_transform(self.features)
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, self.target, test_size=0.2, random_state=42
        )
        
        # 创建随机森林模型
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        # 训练模型
        rf_model.fit(X_train, y_train)
        
        # 预测
        y_pred_train = rf_model.predict(X_train)
        y_pred_test = rf_model.predict(X_test)
        
        # 评估模型
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        print(f"训练集 R²: {train_r2:.4f}")
        print(f"测试集 R²: {test_r2:.4f}")
        print(f"训练集 RMSE: {train_rmse:.4f}")
        print(f"测试集 RMSE: {test_rmse:.4f}")
        
        self.model = rf_model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred_test = y_pred_test
        
        return {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse
        }
    
    def analyze_feature_importance(self):
        """分析特征重要性"""
        print("正在分析特征重要性...")
        
        if self.model is None:
            print("请先训练模型")
            return
        
        # 获取特征重要性
        feature_importance = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        # 绘制特征重要性图
        plt.figure(figsize=(12, 8))
        top_features = feature_importance_df.head(20)
        
        plt.barh(range(len(top_features)), top_features['importance'], 
                color='skyblue', edgecolor='black')
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('特征重要性', fontsize=12, fontweight='bold')
        plt.title('水位差影响因素重要性排名 (Top 20)', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        
        # 设置背景样式
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().set_facecolor('#f8f9fa')
        
        plt.tight_layout()
        
        # 保存图表
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        importance_path = os.path.join(OUTPUT_FOLDER, '特征重要性分析.png')
        plt.savefig(importance_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"特征重要性图表已保存到: {importance_path}")
        
        # 打印特征重要性排名
        print("\n=== 特征重要性排名 (Top 15) ===")
        for i, row in feature_importance_df.head(15).iterrows():
            print(f"{i+1:2d}. {row['feature']:<30} {row['importance']:.4f}")
        
        return feature_importance_df
    
    def analyze_correlations(self):
        """分析特征相关性"""
        print("正在分析特征相关性...")
        
        # 创建包含目标变量的完整数据框用于相关性分析
        correlation_data = self.features.copy()
        correlation_data['water_level_diff'] = self.target
        
        # 计算相关性矩阵
        correlation_matrix = correlation_data.corr()
        
        # 绘制相关性热力图
        plt.figure(figsize=(16, 12))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='coolwarm', 
                   center=0, square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        
        plt.title('特征相关性矩阵 (包含目标变量)', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        # 保存图表
        correlation_path = os.path.join(OUTPUT_FOLDER, '特征相关性热力图.png')
        plt.savefig(correlation_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"相关性热力图已保存到: {correlation_path}")
        
        # 计算与目标变量的相关性
        target_correlations = []
        for feature in self.feature_names:
            corr = correlation_data[feature].corr(correlation_data['water_level_diff'])
            target_correlations.append({
                'feature': feature,
                'correlation': corr
            })
        
        target_corr_df = pd.DataFrame(target_correlations)
        target_corr_df = target_corr_df.sort_values('correlation', key=abs, ascending=False)
        
        # 打印相关性分析结果
        print("\n=== 与水位差相关性排名 (Top 10) ===")
        for i, row in target_corr_df.head(10).iterrows():
            print(f"{i+1:2d}. {row['feature']:<30} {row['correlation']:.4f}")
        
        return target_corr_df
    
    def plot_prediction_results(self):
        """绘制预测结果"""
        print("正在绘制预测结果...")
        
        if self.model is None:
            print("请先训练模型")
            return
        
        # 创建预测结果图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 预测值vs真实值散点图
        axes[0, 0].scatter(self.y_test, self.y_pred_test, alpha=0.6, color='blue', s=30)
        min_val = min(self.y_test.min(), self.y_pred_test.min())
        max_val = max(self.y_test.max(), self.y_pred_test.max())
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        axes[0, 0].set_xlabel('真实值 (m)', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('预测值 (m)', fontsize=12, fontweight='bold')
        axes[0, 0].set_title('预测值 vs 真实值', fontsize=14, fontweight='bold')
        axes[0, 0].legend(['完美预测'])
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 预测误差分布
        errors = self.y_pred_test - self.y_test
        axes[0, 1].hist(errors, bins=30, color='green', alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('预测误差 (m)', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('频次', fontsize=12, fontweight='bold')
        axes[0, 1].set_title('预测误差分布', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 预测误差时间序列
        axes[1, 0].plot(errors, color='red', linewidth=1, alpha=0.7)
        axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[1, 0].set_xlabel('样本索引', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('预测误差 (m)', fontsize=12, fontweight='bold')
        axes[1, 0].set_title('预测误差时间序列', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 残差图
        axes[1, 1].scatter(self.y_pred_test, errors, alpha=0.6, color='purple', s=30)
        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[1, 1].set_xlabel('预测值 (m)', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('残差 (m)', fontsize=12, fontweight='bold')
        axes[1, 1].set_title('残差图', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 设置背景样式
        for ax in axes.flat:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_facecolor('#f8f9fa')
        
        plt.tight_layout()
        
        # 保存图表
        prediction_path = os.path.join(OUTPUT_FOLDER, '预测结果分析.png')
        plt.savefig(prediction_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"预测结果图表已保存到: {prediction_path}")
    
    def generate_analysis_report(self, model_metrics, feature_importance_df, target_corr_df):
        """生成分析报告"""
        print("正在生成分析报告...")
        
        report = []
        report.append("=" * 60)
        report.append("随机森林水位差影响因素分析报告")
        report.append("=" * 60)
        report.append("")
        
        # 模型性能
        report.append("1. 模型性能评估")
        report.append("-" * 30)
        report.append(f"训练集 R²: {model_metrics['train_r2']:.4f}")
        report.append(f"测试集 R²: {model_metrics['test_r2']:.4f}")
        report.append(f"训练集 RMSE: {model_metrics['train_rmse']:.4f}")
        report.append(f"测试集 RMSE: {model_metrics['test_rmse']:.4f}")
        report.append("")
        
        # 特征重要性
        report.append("2. 特征重要性排名 (Top 15)")
        report.append("-" * 30)
        for i, row in feature_importance_df.head(15).iterrows():
            report.append(f"{i+1:2d}. {row['feature']:<30} {row['importance']:.4f}")
        report.append("")
        
        # 与目标变量相关性
        report.append("3. 与水位差相关性排名 (Top 15)")
        report.append("-" * 30)
        for i, row in target_corr_df.head(15).iterrows():
            report.append(f"{i+1:2d}. {row['feature']:<30} {row['correlation']:.4f}")
        report.append("")
        
        # 关键发现
        report.append("4. 关键发现")
        report.append("-" * 30)
        top_feature = feature_importance_df.iloc[0]
        report.append(f"最重要的影响因素: {top_feature['feature']}")
        report.append(f"重要性得分: {top_feature['importance']:.4f}")
        report.append("")
        
        # 建议
        report.append("5. 建议")
        report.append("-" * 30)
        report.append("• 重点关注排名前10的特征因素")
        report.append("• 考虑特征之间的相互作用")
        report.append("• 定期更新模型以保持预测精度")
        report.append("")
        
        report.append("=" * 60)
        
        # 保存报告
        report_text = "\n".join(report)
        report_path = os.path.join(OUTPUT_FOLDER, '随机森林分析报告.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"分析报告已保存到: {report_path}")
        
        # 保存特征重要性数据
        feature_importance_path = os.path.join(OUTPUT_FOLDER, '特征重要性排名.csv')
        feature_importance_df.to_csv(feature_importance_path, index=False, encoding='utf-8')
        print(f"特征重要性数据已保存到: {feature_importance_path}")
        
        # 保存相关性数据
        correlation_path = os.path.join(OUTPUT_FOLDER, '目标变量相关性.csv')
        target_corr_df.to_csv(correlation_path, index=False, encoding='utf-8')
        print(f"相关性数据已保存到: {correlation_path}")
        
        return report_text

def main():
    """主函数"""
    print("=" * 60)
    print("随机森林水位差影响因素分析")
    print("=" * 60)
    
    try:
        # 创建分析器
        analyzer = WaterLevelDifferenceAnalyzer()
        
        # 1. 加载和准备数据
        analyzer.load_and_prepare_data()
        
        # 2. 创建特征
        analyzer.create_features()
        
        # 3. 训练随机森林模型
        model_metrics = analyzer.train_random_forest()
        
        # 4. 分析特征重要性
        feature_importance_df = analyzer.analyze_feature_importance()
        
        # 5. 分析特征相关性
        target_corr_df = analyzer.analyze_correlations()
        
        # 6. 绘制预测结果
        analyzer.plot_prediction_results()
        
        # 7. 生成分析报告
        report = analyzer.generate_analysis_report(model_metrics, feature_importance_df, target_corr_df)
        
        print("\n" + "=" * 60)
        print("分析完成！")
        print(f"结果保存在: {OUTPUT_FOLDER}")
        print("=" * 60)
        
        # 打印关键结果
        print("\n关键结果摘要:")
        print(f"• 模型测试集 R²: {model_metrics['test_r2']:.4f}")
        print(f"• 模型测试集 RMSE: {model_metrics['test_rmse']:.4f}")
        print(f"• 最重要的特征: {feature_importance_df.iloc[0]['feature']}")
        print(f"• 与水位差最相关的特征: {target_corr_df.iloc[0]['feature']}")
        print(f"• 特征总数: {len(feature_importance_df)}")
        
        # 打印特征类型统计
        print(f"\n特征类型统计:")
        feature_types = {}
        for feature in feature_importance_df['feature']:
            if '长江水位_' in feature:
                if '长江历史水位' not in feature_types:
                    feature_types['长江历史水位'] = 0
                feature_types['长江历史水位'] += 1
            elif '运河水位_' in feature:
                if '运河历史水位' not in feature_types:
                    feature_types['运河历史水位'] = 0
                feature_types['运河历史水位'] += 1
            elif '潮位_' in feature:
                if '延迟潮位' not in feature_types:
                    feature_types['延迟潮位'] = 0
                feature_types['延迟潮位'] += 1
            elif '流量' in feature:
                if '当前流量' not in feature_types:
                    feature_types['当前流量'] = 0
                feature_types['当前流量'] += 1
            else:
                if '其他' not in feature_types:
                    feature_types['其他'] = 0
                feature_types['other'] += 1
        
        for feature_type, count in feature_types.items():
            print(f"  • {feature_type}: {count} 个特征")
        
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 
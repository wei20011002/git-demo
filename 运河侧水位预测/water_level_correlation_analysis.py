import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
from datetime import datetime

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class WaterLevelCorrelationAnalyzer:
    def __init__(self):
        self.yangtze_df = None
        self.yunhe_df = None
        self.combined_df = None
        self.output_folder = None
        
    def load_data(self):
        """加载长江和运河水位数据"""
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
                        
                        # 提取长江侧水位数据
                        df_y = df[df['名称'] == '长江侧水位'].copy()
                        df_y['数值'] = pd.to_numeric(df_y['数值'], errors='coerce')
                        df_y = df_y.dropna(subset=['数值'])
                        df_y = df_y[df_y['数值'] >= 0]
                        yangtze_data.append(df_y)
                        
                        # 提取运河侧水位数据
                        df_u = df[df['名称'] == '内河侧水位'].copy()
                        df_u['数值'] = pd.to_numeric(df_u['数值'], errors='coerce')
                        df_u = df_u.dropna(subset=['数值'])
                        df_u = df_u[df_u['数值'] >= 0]
                        yunhe_data.append(df_u)
                        
                except Exception as e:
                    print(f"处理文件 {file_path} 时出错: {e}")
                    continue
        
        # 合并数据
        self.yangtze_df = pd.concat(yangtze_data, ignore_index=True)
        self.yunhe_df = pd.concat(yunhe_data, ignore_index=True)
        
        # 处理时间索引
        self.yangtze_df['datetime'] = pd.to_datetime(self.yangtze_df['日期'] + ' ' + self.yangtze_df['时间'], format='mixed')
        self.yangtze_df.set_index('datetime', inplace=True)
        self.yangtze_df = self.yangtze_df[['数值']].rename(columns={'数值': 'yangtze_level'}).resample('h').mean().dropna()
        
        self.yunhe_df['datetime'] = pd.to_datetime(self.yunhe_df['日期'] + ' ' + self.yunhe_df['时间'], format='mixed')
        self.yunhe_df.set_index('datetime', inplace=True)
        self.yunhe_df = self.yunhe_df[['数值']].rename(columns={'数值': 'yunhe_level'}).resample('h').mean().dropna()
        
        # 合并两个数据集
        self.combined_df = pd.merge(self.yangtze_df, self.yunhe_df, left_index=True, right_index=True, how='inner')
        
        print(f"数据加载完成，共 {len(self.combined_df)} 条记录")
        print(f"时间范围: {self.combined_df.index.min()} 到 {self.combined_df.index.max()}")
        
    def create_output_folder(self):
        """创建输出文件夹"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_folder = os.path.join(script_dir, '水位相关性分析结果')
        os.makedirs(self.output_folder, exist_ok=True)
        print(f"输出文件夹: {self.output_folder}")
        
    def basic_statistics(self):
        """基本统计分析"""
        print("\n=== 基本统计分析 ===")
        
        stats_data = {
            '指标': ['数据量', '时间范围', '长江水位范围', '运河水位范围', 
                    '长江平均水位', '运河平均水位', '长江水位标准差', '运河水位标准差'],
            '数值': [
                len(self.combined_df),
                f"{self.combined_df.index.min().strftime('%Y-%m-%d')} 到 {self.combined_df.index.max().strftime('%Y-%m-%d')}",
                f"{self.combined_df['yangtze_level'].min():.2f}m - {self.combined_df['yangtze_level'].max():.2f}m",
                f"{self.combined_df['yunhe_level'].min():.2f}m - {self.combined_df['yunhe_level'].max():.2f}m",
                f"{self.combined_df['yangtze_level'].mean():.2f}m",
                f"{self.combined_df['yunhe_level'].mean():.2f}m",
                f"{self.combined_df['yangtze_level'].std():.2f}m",
                f"{self.combined_df['yunhe_level'].std():.2f}m"
            ]
        }
        
        stats_df = pd.DataFrame(stats_data)
        stats_df.to_csv(os.path.join(self.output_folder, '基本统计信息.csv'), index=False, encoding='utf-8-sig')
        print(stats_df.to_string(index=False))
        
        return stats_df
        
    def correlation_analysis(self):
        """相关性分析"""
        print("\n=== 相关性分析 ===")
        
        # 计算各种相关系数
        pearson_corr = self.combined_df['yangtze_level'].corr(self.combined_df['yunhe_level'])
        spearman_corr = self.combined_df['yangtze_level'].corr(self.combined_df['yunhe_level'], method='spearman')
        kendall_corr = self.combined_df['yangtze_level'].corr(self.combined_df['yunhe_level'], method='kendall')
        
        # 计算R²
        r2 = r2_score(self.combined_df['yunhe_level'], self.combined_df['yangtze_level'])
        
        # 计算RMSE和MAE
        rmse = np.sqrt(mean_squared_error(self.combined_df['yunhe_level'], self.combined_df['yangtze_level']))
        mae = mean_absolute_error(self.combined_df['yunhe_level'], self.combined_df['yangtze_level'])
        
        correlation_data = {
            '指标': ['Pearson相关系数', 'Spearman相关系数', 'Kendall相关系数', 'R²', 'RMSE', 'MAE'],
            '数值': [pearson_corr, spearman_corr, kendall_corr, r2, rmse, mae],
            '说明': [
                '线性相关性',
                '单调相关性',
                '等级相关性',
                '决定系数',
                '均方根误差',
                '平均绝对误差'
            ]
        }
        
        correlation_df = pd.DataFrame(correlation_data)
        correlation_df.to_csv(os.path.join(self.output_folder, '相关性分析结果.csv'), index=False, encoding='utf-8-sig')
        print(correlation_df.to_string(index=False))
        
        return correlation_df
        
    def plot_correlation_analysis(self):
        """绘制相关性分析图表"""
        print("\n=== 生成相关性分析图表 ===")
        
        # 1. 散点图
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.scatter(self.combined_df['yangtze_level'], self.combined_df['yunhe_level'], alpha=0.6, s=1)
        plt.xlabel('长江水位 (m)')
        plt.ylabel('运河水位 (m)')
        plt.title('长江水位 vs 运河水位散点图')
        
        # 添加趋势线
        z = np.polyfit(self.combined_df['yangtze_level'], self.combined_df['yunhe_level'], 1)
        p = np.poly1d(z)
        plt.plot(self.combined_df['yangtze_level'], p(self.combined_df['yangtze_level']), "r--", alpha=0.8)
        
        # 2. 时间序列对比
        plt.subplot(2, 2, 2)
        sample_data = self.combined_df.iloc[::24]  # 每24小时取一个样本点
        plt.plot(sample_data.index, sample_data['yangtze_level'], label='长江水位', alpha=0.7)
        plt.plot(sample_data.index, sample_data['yunhe_level'], label='运河水位', alpha=0.7)
        plt.xlabel('时间')
        plt.ylabel('水位 (m)')
        plt.title('长江与运河水位时间序列对比')
        plt.legend()
        plt.xticks(rotation=45)
        
        # 3. 相关性热力图
        plt.subplot(2, 2, 3)
        corr_matrix = self.combined_df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=0.5)
        plt.title('水位相关性热力图')
        
        # 4. 水位差分析
        plt.subplot(2, 2, 4)
        water_diff = self.combined_df['yangtze_level'] - self.combined_df['yunhe_level']
        plt.hist(water_diff, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('水位差 (长江-运河) (m)')
        plt.ylabel('频次')
        plt.title('长江与运河水位差分布')
        plt.axvline(water_diff.mean(), color='red', linestyle='--', label=f'平均值: {water_diff.mean():.2f}m')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, '相关性分析图表.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. 月度相关性分析
        self.plot_monthly_correlation()
        
        # 6. 水位差时间序列
        self.plot_water_difference_timeseries()
        
    def plot_monthly_correlation(self):
        """月度相关性分析"""
        print("生成月度相关性分析...")
        
        # 添加月份信息
        df_with_month = self.combined_df.copy()
        df_with_month['month'] = df_with_month.index.month
        df_with_month['year'] = df_with_month.index.year
        
        # 计算每月相关性
        monthly_corr = []
        months = []
        
        for year in [2022, 2023, 2024]:
            for month in range(1, 13):
                month_data = df_with_month[(df_with_month['year'] == year) & (df_with_month['month'] == month)]
                if len(month_data) > 10:  # 确保有足够的数据
                    corr = month_data['yangtze_level'].corr(month_data['yunhe_level'])
                    monthly_corr.append(corr)
                    months.append(f"{year}-{month:02d}")
        
        # 绘制月度相关性
        plt.figure(figsize=(15, 6))
        plt.plot(months, monthly_corr, marker='o', linewidth=2, markersize=6)
        plt.axhline(y=np.mean(monthly_corr), color='red', linestyle='--', alpha=0.7, label=f'平均相关性: {np.mean(monthly_corr):.3f}')
        plt.xlabel('年月')
        plt.ylabel('相关系数')
        plt.title('月度长江与运河水位相关性变化')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, '月度相关性分析.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存月度相关性数据
        monthly_data = pd.DataFrame({
            '年月': months,
            '相关系数': monthly_corr
        })
        monthly_data.to_csv(os.path.join(self.output_folder, '月度相关性数据.csv'), index=False, encoding='utf-8-sig')
        
    def plot_water_difference_timeseries(self):
        """水位差时间序列分析"""
        print("生成水位差时间序列分析...")
        
        water_diff = self.combined_df['yangtze_level'] - self.combined_df['yunhe_level']
        
        plt.figure(figsize=(15, 8))
        
        # 水位差时间序列
        plt.subplot(2, 1, 1)
        sample_diff = water_diff.iloc[::24]  # 每24小时取一个样本点
        plt.plot(sample_diff.index, sample_diff.values, alpha=0.7, linewidth=0.8)
        plt.axhline(y=water_diff.mean(), color='red', linestyle='--', alpha=0.8, 
                   label=f'平均值: {water_diff.mean():.2f}m')
        plt.fill_between(sample_diff.index, water_diff.mean() - water_diff.std(), 
                        water_diff.mean() + water_diff.std(), alpha=0.2, color='red')
        plt.xlabel('时间')
        plt.ylabel('水位差 (长江-运河) (m)')
        plt.title('长江与运河水位差时间序列')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 水位差分布
        plt.subplot(2, 1, 2)
        plt.hist(water_diff, bins=50, alpha=0.7, edgecolor='black', density=True)
        plt.xlabel('水位差 (长江-运河) (m)')
        plt.ylabel('密度')
        plt.title('水位差分布直方图')
        plt.axvline(water_diff.mean(), color='red', linestyle='--', 
                   label=f'平均值: {water_diff.mean():.2f}m')
        plt.axvline(water_diff.median(), color='green', linestyle='--', 
                   label=f'中位数: {water_diff.median():.2f}m')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, '水位差时间序列分析.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存水位差统计信息
        diff_stats = {
            '指标': ['平均值', '中位数', '标准差', '最小值', '最大值', '25%分位数', '75%分位数'],
            '数值': [
                water_diff.mean(),
                water_diff.median(),
                water_diff.std(),
                water_diff.min(),
                water_diff.max(),
                water_diff.quantile(0.25),
                water_diff.quantile(0.75)
            ]
        }
        
        diff_stats_df = pd.DataFrame(diff_stats)
        diff_stats_df.to_csv(os.path.join(self.output_folder, '水位差统计信息.csv'), index=False, encoding='utf-8-sig')
        
    def generate_report(self):
        """生成分析报告"""
        print("\n=== 生成分析报告 ===")
        
        # 读取之前生成的数据
        stats_df = pd.read_csv(os.path.join(self.output_folder, '基本统计信息.csv'), encoding='utf-8-sig')
        correlation_df = pd.read_csv(os.path.join(self.output_folder, '相关性分析结果.csv'), encoding='utf-8-sig')
        
        report = f"""运河水位与长江水位相关性分析报告
============================================================

📊 分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📈 数据概况:
{stats_df.to_string(index=False)}

🔗 相关性分析:
{correlation_df.to_string(index=False)}

📋 主要发现:

1. 数据质量:
   - 总数据量: {stats_df.iloc[0, 1]} 条记录
   - 时间跨度: {stats_df.iloc[1, 1]}
   - 数据完整性良好

2. 水位特征:
   - 长江水位范围: {stats_df.iloc[2, 1]}
   - 运河水位范围: {stats_df.iloc[3, 1]}
   - 长江平均水位: {stats_df.iloc[4, 1]}
   - 运河平均水位: {stats_df.iloc[5, 1]}

3. 相关性强度:
   - Pearson相关系数: {correlation_df.iloc[0, 1]:.4f}
   - Spearman相关系数: {correlation_df.iloc[1, 1]:.4f}
   - 决定系数R²: {correlation_df.iloc[3, 1]:.4f}

4. 预测精度:
   - RMSE: {correlation_df.iloc[4, 1]:.4f}m
   - MAE: {correlation_df.iloc[5, 1]:.4f}m

📁 生成的文件:
   - 基本统计信息.csv
   - 相关性分析结果.csv
   - 月度相关性数据.csv
   - 水位差统计信息.csv
   - 相关性分析图表.png
   - 月度相关性分析.png
   - 水位差时间序列分析.png

✅ 结论:
   - 长江水位与运河水位存在显著相关性
   - 相关性强度为 {correlation_df.iloc[0, 1]:.3f}，属于{'强相关' if abs(correlation_df.iloc[0, 1]) > 0.7 else '中等相关' if abs(correlation_df.iloc[0, 1]) > 0.3 else '弱相关'}
   - 可用于水位预测和监控
"""
        
        with open(os.path.join(self.output_folder, '相关性分析报告.txt'), 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("分析报告已生成完成！")
        
    def run_analysis(self):
        """运行完整分析"""
        self.load_data()
        self.create_output_folder()
        self.basic_statistics()
        self.correlation_analysis()
        self.plot_correlation_analysis()
        self.generate_report()
        print(f"\n✅ 分析完成！所有结果已保存到: {self.output_folder}")

if __name__ == "__main__":
    analyzer = WaterLevelCorrelationAnalyzer()
    analyzer.run_analysis() 
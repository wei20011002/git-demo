import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime, timedelta
import warnings
import os
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建输出图片文件夹
OUTPUT_FOLDER = 'output_images'
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
    print(f"创建输出文件夹: {OUTPUT_FOLDER}")

def analyze_missing_data_impact():
    """分析缺失时段对相关性分析的影响"""
    print("=== 分析缺失时段对相关性分析的影响 ===")
    
    # 加载2023年数据
    print("\n1. 加载2023年原始数据...")
    
    # 加载潮位数据
    tide_df = pd.read_csv('潮位数据处理/2023年潮位数据_真正小时级.csv')
    tide_df['datetime'] = pd.to_datetime(tide_df['datetime'])
    tide_df.set_index('datetime', inplace=True)
    
    # 加载水位数据
    monthly_data = []
    for month in range(1, 13):
        file_path = f'2023/2023-{month}-1.csv'
        try:
            df = pd.read_csv(file_path, encoding='gbk')
            df_filtered = df[df['名称'] == '长江侧水位'].copy()
            if len(df_filtered) > 0:
                df_filtered['数值'] = pd.to_numeric(df_filtered['数值'], errors='coerce')
                df_filtered = df_filtered.dropna(subset=['数值'])
                monthly_data.append(df_filtered)
        except Exception as e:
            continue
    
    combined_df = pd.concat(monthly_data, ignore_index=True)
    combined_df['datetime'] = pd.to_datetime(combined_df['日期'] + ' ' + combined_df['时间'])
    combined_df.set_index('datetime', inplace=True)
    combined_df = combined_df[['数值']].copy()
    combined_df.rename(columns={'数值': 'water_level'}, inplace=True)
    combined_df = combined_df.resample('H').mean()
    combined_df = combined_df.dropna()
    
    print(f"原始潮位数据点数: {len(tide_df)}")
    print(f"原始水位数据点数: {len(combined_df)}")
    
    # 分析1：包含所有数据（包括缺失时段）
    print("\n2. 分析1：包含所有数据（包括缺失时段）")
    merged_all = pd.merge(tide_df, combined_df, left_index=True, right_index=True, how='inner')
    print(f"合并后数据点数: {len(merged_all)}")
    
    # 计算不同延迟时间的相关性
    lag_correlations_all = []
    for lag in range(0, 49):
        if lag == 0:
            correlation = merged_all['tide_level'].corr(merged_all['water_level'], method='pearson')
        else:
            tide_shifted = merged_all['tide_level'].shift(lag)
            correlation = tide_shifted.corr(merged_all['water_level'], method='pearson')
        
        lag_correlations_all.append({
            'lag_hours': lag,
            'correlation': correlation
        })
    
    lag_df_all = pd.DataFrame(lag_correlations_all).dropna()
    best_lag_idx_all = lag_df_all['correlation'].abs().idxmax()
    best_lag_all = lag_df_all.loc[best_lag_idx_all, 'lag_hours']
    best_correlation_all = lag_df_all.loc[best_lag_idx_all, 'correlation']
    
    print(f"包含所有数据的最佳延迟: {best_lag_all}小时")
    print(f"包含所有数据的最佳相关系数: {best_correlation_all:.4f}")
    
    # 分析2：排除缺失时段数据
    print("\n3. 分析2：排除缺失时段数据")
    # 排除8月和9月的不完整数据
    mask = ~((merged_all.index.month == 8) | (merged_all.index.month == 9))
    merged_filtered = merged_all[mask]
    print(f"排除缺失时段后数据点数: {len(merged_filtered)}")
    
    # 计算不同延迟时间的相关性
    lag_correlations_filtered = []
    for lag in range(0, 49):
        if lag == 0:
            correlation = merged_filtered['tide_level'].corr(merged_filtered['water_level'], method='pearson')
        else:
            tide_shifted = merged_filtered['tide_level'].shift(lag)
            correlation = tide_shifted.corr(merged_filtered['water_level'], method='pearson')
        
        lag_correlations_filtered.append({
            'lag_hours': lag,
            'correlation': correlation
        })
    
    lag_df_filtered = pd.DataFrame(lag_correlations_filtered).dropna()
    best_lag_idx_filtered = lag_df_filtered['correlation'].abs().idxmax()
    best_lag_filtered = lag_df_filtered.loc[best_lag_idx_filtered, 'lag_hours']
    best_correlation_filtered = lag_df_filtered.loc[best_lag_idx_filtered, 'correlation']
    
    print(f"排除缺失时段后的最佳延迟: {best_lag_filtered}小时")
    print(f"排除缺失时段后的最佳相关系数: {best_correlation_filtered:.4f}")
    
    # 分析3：详细比较缺失时段的数据特征
    print("\n4. 分析3：缺失时段的数据特征")
    
    # 8月数据
    august_data = merged_all[merged_all.index.month == 8]
    september_data = merged_all[merged_all.index.month == 9]
    
    print(f"8月数据点数: {len(august_data)}")
    print(f"9月数据点数: {len(september_data)}")
    
    if len(august_data) > 0:
        august_corr = august_data['tide_level'].corr(august_data['water_level'], method='pearson')
        print(f"8月数据相关性: {august_corr:.4f}")
    
    if len(september_data) > 0:
        september_corr = september_data['tide_level'].corr(september_data['water_level'], method='pearson')
        print(f"9月数据相关性: {september_corr:.4f}")
    
    # 分析4：其他月份的数据特征
    print("\n5. 分析4：其他月份的数据特征")
    other_months_data = merged_all[~((merged_all.index.month == 8) | (merged_all.index.month == 9))]
    
    monthly_correlations = {}
    for month in range(1, 13):
        if month not in [8, 9]:
            month_data = merged_all[merged_all.index.month == month]
            if len(month_data) > 0:
                # 计算6小时延迟的相关性（因为这是最佳延迟）
                tide_shifted = month_data['tide_level'].shift(6)
                correlation = tide_shifted.corr(month_data['water_level'], method='pearson')
                monthly_correlations[month] = correlation
                print(f"{month}月相关性(6小时延迟): {correlation:.4f}")
    
    # 计算其他月份的平均相关性
    if monthly_correlations:
        avg_other_corr = np.mean(list(monthly_correlations.values()))
        print(f"\n其他月份平均相关性: {avg_other_corr:.4f}")
    
    # 分析5：数据质量对相关性的影响
    print("\n6. 分析5：数据质量对相关性的影响")
    print(f"包含缺失时段的相关性: {best_correlation_all:.4f}")
    print(f"排除缺失时段的相关性: {best_correlation_filtered:.4f}")
    print(f"相关性变化: {best_correlation_filtered - best_correlation_all:.4f}")
    
    if best_correlation_filtered > best_correlation_all:
        print("结论：排除缺失时段后相关性提高，说明缺失时段的数据质量较差")
    else:
        print("结论：排除缺失时段后相关性降低，说明缺失时段的数据质量较好")
    
    # 绘制对比图
    plt.figure(figsize=(15, 10))
    
    # 延迟相关性对比
    plt.subplot(2, 2, 1)
    plt.plot(lag_df_all['lag_hours'], lag_df_all['correlation'], 'b-', linewidth=2, 
             label='包含所有数据', alpha=0.7)
    plt.plot(lag_df_filtered['lag_hours'], lag_df_filtered['correlation'], 'r-', linewidth=2, 
             label='排除缺失时段', alpha=0.7)
    plt.axvline(x=best_lag_all, color='blue', linestyle='--', alpha=0.5)
    plt.axvline(x=best_lag_filtered, color='red', linestyle='--', alpha=0.5)
    plt.xlabel('延迟时间 (小时)')
    plt.ylabel('皮尔逊相关系数')
    plt.title('延迟相关性对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 月度相关性分布
    plt.subplot(2, 2, 2)
    months = list(monthly_correlations.keys())
    corrs = list(monthly_correlations.values())
    plt.bar(months, corrs, alpha=0.7, color='skyblue')
    plt.axhline(y=avg_other_corr, color='red', linestyle='--', label=f'平均值: {avg_other_corr:.3f}')
    plt.xlabel('月份')
    plt.ylabel('皮尔逊相关系数')
    plt.title('各月份相关性分布(6小时延迟)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 数据点数对比
    plt.subplot(2, 2, 3)
    categories = ['包含所有数据', '排除缺失时段']
    data_points = [len(merged_all), len(merged_filtered)]
    colors = ['lightblue', 'lightcoral']
    plt.bar(categories, data_points, color=colors, alpha=0.7)
    plt.ylabel('数据点数')
    plt.title('数据点数对比')
    for i, v in enumerate(data_points):
        plt.text(i, v + 50, str(v), ha='center')
    plt.grid(True, alpha=0.3)
    
    # 相关性强度对比
    plt.subplot(2, 2, 4)
    correlation_values = [best_correlation_all, best_correlation_filtered]
    plt.bar(categories, correlation_values, color=colors, alpha=0.7)
    plt.ylabel('最佳皮尔逊相关系数')
    plt.title('相关性强度对比')
    for i, v in enumerate(correlation_values):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, '缺失时段影响分析.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 生成详细报告
    report = "缺失时段对相关性分析的影响报告\n"
    report += "=" * 50 + "\n\n"
    
    report += f"1. 数据概况:\n"
    report += f"   原始数据点数: {len(merged_all)}\n"
    report += f"   排除缺失时段后: {len(merged_filtered)}\n"
    report += f"   排除的数据点数: {len(merged_all) - len(merged_filtered)}\n\n"
    
    report += f"2. 相关性分析结果:\n"
    report += f"   包含所有数据 - 最佳延迟: {best_lag_all}小时, 相关系数: {best_correlation_all:.4f}\n"
    report += f"   排除缺失时段 - 最佳延迟: {best_lag_filtered}小时, 相关系数: {best_correlation_filtered:.4f}\n"
    report += f"   相关性变化: {best_correlation_filtered - best_correlation_all:.4f}\n\n"
    
    report += f"3. 缺失时段数据特征:\n"
    if len(august_data) > 0:
        report += f"   8月数据点数: {len(august_data)}, 相关性: {august_corr:.4f}\n"
    if len(september_data) > 0:
        report += f"   9月数据点数: {len(september_data)}, 相关性: {september_corr:.4f}\n"
    
    report += f"   其他月份平均相关性: {avg_other_corr:.4f}\n\n"
    
    report += f"4. 结论:\n"
    if best_correlation_filtered > best_correlation_all:
        report += f"   排除缺失时段后相关性提高，说明缺失时段的数据质量较差，\n"
        report += f"   这些不完整的数据会降低整体相关性分析的准确性。\n"
    else:
        report += f"   排除缺失时段后相关性降低，说明缺失时段的数据质量较好，\n"
        report += f"   但这些数据的缺失影响了分析的完整性。\n"
    
    report += f"   因此，排除不完整数据是合理的做法，能够提高分析结果的可靠性。\n"
    
    # 保存报告
    with open('缺失时段影响分析报告.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("\n详细分析报告已保存为: 缺失时段影响分析报告.txt")
    return report

if __name__ == "__main__":
    analyze_missing_data_impact() 
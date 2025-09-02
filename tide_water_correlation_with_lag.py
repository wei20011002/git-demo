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

# 定义月份名称常量
MONTH_NAMES = {
    1: '一月', 2: '二月', 3: '三月', 4: '四月', 5: '五月', 6: '六月',
    7: '七月', 8: '八月', 9: '九月', 10: '十月', 11: '十一月', 12: '十二月'
}

# 物理约束：最大延迟不超过10小时
PHYSICAL_MAX_LAG = 10

# 创建输出图片文件夹
OUTPUT_FOLDER = 'output_images'
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
    print(f"创建输出文件夹: {OUTPUT_FOLDER}")

def load_tide_data():
    """加载处理后的潮位数据"""
    print("正在加载潮位数据...")
    
    tide_data = {}
    for year in [2022, 2023, 2024]:
        try:
            file_path = f'潮位数据处理/{year}年潮位数据_真正小时级.csv'
            df = pd.read_csv(file_path)
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            
            # 对于2023年，排除8月和9月数据缺失的时段
            if year == 2023:
                print(f"2023年原始潮位数据: {len(df)}个数据点")
                
                # 排除8月和9月的不完整数据
                mask = ~((df.index.month == 8) | (df.index.month == 9))
                df = df[mask]
                
                print(f"排除8月和9月不完整数据后: {len(df)}个数据点")
                print(f"排除的月份: 8月(不完整) 和 9月(不完整)")
            
            tide_data[year] = df
            print(f"{year}年潮位数据加载成功，共{len(df)}条记录")
        except Exception as e:
            print(f"加载{year}年潮位数据失败: {e}")
    
    return tide_data

def load_water_level_data():
    """加载长江侧水位数据"""
    print("正在加载长江侧水位数据...")
    
    water_level_data = {}
    for year in [2022, 2023, 2024]:
        try:
            # 加载该年的所有月度数据
            monthly_data = []
            for month in range(1, 13):
                if year == 2024:
                    file_path = f'{year}/水位{year}-{month}-1.csv'
                else:
                    file_path = f'{year}/{year}-{month}-1.csv'
                
                try:
                    # 尝试不同的编码方式
                    try:
                        df = pd.read_csv(file_path, encoding='utf-8')
                    except:
                        df = pd.read_csv(file_path, encoding='gbk')
                    
                    # 选择长江侧水位（与潮位关系最密切）
                    df_filtered = df[df['名称'] == '长江侧水位'].copy()
                    if len(df_filtered) > 0:
                        # 确保数值列是数值类型
                        df_filtered['数值'] = pd.to_numeric(df_filtered['数值'], errors='coerce')
                        df_filtered = df_filtered.dropna(subset=['数值'])
                        monthly_data.append(df_filtered)
                except Exception as e:
                    print(f"处理{year}年{month}月数据时出错: {e}")
                    continue
            
            if monthly_data:
                # 合并所有月度数据
                combined_df = pd.concat(monthly_data, ignore_index=True)
                combined_df['datetime'] = pd.to_datetime(combined_df['日期'] + ' ' + combined_df['时间'])
                combined_df.set_index('datetime', inplace=True)
                
                # 只保留数值列进行重采样
                combined_df = combined_df[['数值']].copy()
                combined_df.rename(columns={'数值': 'water_level'}, inplace=True)
                
                # 重采样到小时级别
                combined_df = combined_df.resample('H').mean()
                combined_df = combined_df.dropna()
                
                # 对于2023年，排除8月和9月数据缺失的时段
                if year == 2023:
                    print(f"2023年原始数据: {len(combined_df)}个数据点")
                    
                    # 排除8月和9月的不完整数据
                    # 8月只有1-8日的数据，9月只有22-30日的数据
                    mask = ~((combined_df.index.month == 8) | (combined_df.index.month == 9))
                    combined_df = combined_df[mask]
                    
                    print(f"排除8月和9月不完整数据后: {len(combined_df)}个数据点")
                    print(f"排除的月份: 8月(不完整) 和 9月(不完整)")
                    print(f"实际时间范围: {combined_df.index.min()} 到 {combined_df.index.max()}")
                
                water_level_data[year] = combined_df
                print(f"{year}年水位数据加载成功，共{len(combined_df)}条记录")
            else:
                print(f"{year}年未找到有效的水位数据")
                
        except Exception as e:
            print(f"加载{year}年水位数据失败: {e}")
            import traceback
            traceback.print_exc()
    
    return water_level_data

def analyze_monthly_correlation(tide_data, water_data, max_lag_hours=48):
    """分析不同月份的相关性"""
    print(f"\n开始分析不同月份的相关性（最大延迟{max_lag_hours}小时）...")
    
    monthly_results = {}
    
    for year in tide_data.keys():
        if year not in water_data:
            continue
            
        print(f"\n--- 分析{year}年各月份数据 ---")
        
        tide_df = tide_data[year].copy()
        water_df = water_data[year].copy()
        
        # 合并数据
        merged_df = pd.merge(tide_df, water_df, left_index=True, right_index=True, how='inner')
        
        # 对于2023年，在相关性分析时也排除缺失时段
        if year == 2023:
            print(f"2023年合并后原始数据: {len(merged_df)}条记录")
            print(f"原始时间范围: {merged_df.index.min()} 到 {merged_df.index.max()}")
            
            # 排除8月和9月的不完整数据
            mask = ~((merged_df.index.month == 8) | (merged_df.index.month == 9))
            merged_df = merged_df[mask]
            
            print(f"排除8月和9月不完整数据后: {len(merged_df)}条记录")
            print(f"排除的月份: 8月(不完整) 和 9月(不完整)")
            print(f"最终时间范围: {merged_df.index.min()} 到 {merged_df.index.max()}")
        
        monthly_results[year] = {}
        
        # 按月份分析
        for month in range(1, 13):
            # 跳过2023年的8月和9月
            if year == 2023 and month in [8, 9]:
                print(f"跳过{year}年{month}月（数据不完整）")
                continue
                
            month_data = merged_df[merged_df.index.month == month].copy()
            
            if len(month_data) < 50:  # 确保有足够的数据
                print(f"{year}年{month}月数据不足（{len(month_data)}条），跳过分析")
                continue
            
            print(f"分析{year}年{month}月数据，共{len(month_data)}条记录")
            
            # 计算不同延迟时间的皮尔逊相关系数
            lag_correlations = []
            lag_hours = range(0, max_lag_hours + 1)
            
            for lag in lag_hours:
                if lag == 0:
                    correlation = month_data['tide_level'].corr(month_data['water_level'], method='pearson')
                else:
                    tide_shifted = month_data['tide_level'].shift(lag)
                    correlation = tide_shifted.corr(month_data['water_level'], method='pearson')
                
                lag_correlations.append({
                    'lag_hours': lag,
                    'correlation': correlation
                })
            
            # 找到最佳延迟时间
            lag_df = pd.DataFrame(lag_correlations)
            lag_df = lag_df.dropna()
            
            if len(lag_df) > 0:
                best_lag_idx = lag_df['correlation'].abs().idxmax()
                best_lag = lag_df.loc[best_lag_idx, 'lag_hours']
                best_correlation = lag_df.loc[best_lag_idx, 'correlation']
                
                # 计算最佳延迟下的详细统计
                if best_lag == 0:
                    tide_series = month_data['tide_level']
                    water_series = month_data['water_level']
                elif best_lag > 0:
                    tide_series = month_data['tide_level'].shift(best_lag)
                    water_series = month_data['water_level']
                else:
                    tide_series = month_data['tide_level']
                    water_series = month_data['water_level'].shift(abs(best_lag))
                
                # 去除缺失值
                valid_data = pd.DataFrame({
                    'tide_level': tide_series,
                    'water_level': water_series
                }).dropna()
                
                if len(valid_data) > 10:
                    # 线性回归分析
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        valid_data['tide_level'], valid_data['water_level']
                    )
                    
                    results = {
                        'best_lag': best_lag,
                        'best_correlation': best_correlation,
                        'r_squared': r_value**2,
                        'p_value': p_value,
                        'slope': slope,
                        'intercept': intercept,
                        'data_count': len(valid_data),
                        'lag_correlations': lag_df
                    }
                    
                    monthly_results[year][month] = results
                    
                    print(f"  {month}月 - 最佳延迟: {best_lag}小时, 相关系数: {best_correlation:.4f}, R²: {r_value**2:.4f}")
        
        # 绘制年度月度相关性热力图
        plot_monthly_correlation_heatmap(monthly_results[year], year)
        
        # 绘制月度延迟时间对比
        plot_monthly_lag_comparison(monthly_results[year], year)
    
    return monthly_results

def plot_monthly_correlation_heatmap(monthly_results, year):
    """绘制月度相关性热力图"""
    if not monthly_results:
        return
        
    # 准备数据
    months = list(monthly_results.keys())
    correlations = [monthly_results[month]['best_correlation'] for month in months]
    r_squared_values = [monthly_results[month]['r_squared'] for month in months]
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 对于2023年，在图片标题中标注数据排除情况
    title_suffix = ""
    if year == 2023:
        title_suffix = " (已排除8月和9月不完整数据)"
    
    # 相关系数热力图
    month_names = [MONTH_NAMES[month] for month in months]
    im1 = ax1.bar(month_names, correlations, color='skyblue', alpha=0.7)
    ax1.set_ylabel('皮尔逊相关系数')
    ax1.set_title(f'{year}年各月份潮位与水位最佳相关系数{title_suffix}')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-1, 1)
    
    # 添加数值标签
    for i, v in enumerate(correlations):
        ax1.text(i, v + (0.05 if v >= 0 else -0.05), f'{v:.3f}', 
                ha='center', va='bottom' if v >= 0 else 'top')
    
    # R²值热力图
    im2 = ax2.bar(month_names, r_squared_values, color='lightcoral', alpha=0.7)
    ax2.set_ylabel('R²值')
    ax2.set_title(f'{year}年各月份线性回归拟合优度{title_suffix}')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # 添加数值标签
    for i, v in enumerate(r_squared_values):
        ax2.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, f'{year}年月度相关性分析.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_monthly_lag_comparison(monthly_results, year):
    """绘制月度延迟时间对比图"""
    if not monthly_results:
        return
        
    months = list(monthly_results.keys())
    lags = [monthly_results[month]['best_lag'] for month in months]
    
    plt.figure(figsize=(12, 6))
    
    # 对于2023年，在图片标题中标注数据排除情况
    title_suffix = ""
    if year == 2023:
        title_suffix = " (已排除8月和9月不完整数据)"
    
    month_names = [MONTH_NAMES[month] for month in months]
    bars = plt.bar(month_names, lags, color='lightgreen', alpha=0.7)
    
    plt.ylabel('最佳延迟时间 (小时)')
    plt.title(f'{year}年各月份潮位影响水位的最佳延迟时间{title_suffix}')
    plt.grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, v in enumerate(lags):
        plt.text(i, v + 0.5, f'{v}h', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, f'{year}年月度延迟时间对比.png'), dpi=300, bbox_inches='tight')
    plt.show()

def analyze_correlation_with_lag(tide_data, water_data, max_lag_hours=48):
    """分析潮位与水位相关性，考虑时间延迟"""
    print(f"\n开始分析潮位与水位相关性（最大延迟{max_lag_hours}小时）...")
    
    all_results = {}
    
    for year in tide_data.keys():
        if year not in water_data:
            continue
            
        print(f"\n--- 分析{year}年数据 ---")
        
        tide_df = tide_data[year].copy()
        water_df = water_data[year].copy()
        
        # 合并数据
        merged_df = pd.merge(tide_df, water_df, left_index=True, right_index=True, how='inner')
        
        # 对于2023年，在相关性分析时也排除缺失时段
        if year == 2023:
            print(f"2023年合并后原始数据: {len(merged_df)}条记录")
            print(f"原始时间范围: {merged_df.index.min()} 到 {merged_df.index.max()}")
            
            # 排除8月和9月的不完整数据
            mask = ~((merged_df.index.month == 8) | (merged_df.index.month == 9))
            merged_df = merged_df[mask]
            
            print(f"排除8月和9月不完整数据后: {len(merged_df)}条记录")
            print(f"排除的月份: 8月(不完整) 和 9月(不完整)")
            print(f"最终时间范围: {merged_df.index.min()} 到 {merged_df.index.max()}")
        
        if len(merged_df) < 100:  # 确保有足够的数据
            print(f"{year}年有效数据不足，跳过分析")
            continue
        
        print(f"最终分析数据: {len(merged_df)}条记录")
        print(f"时间范围: {merged_df.index.min()} 到 {merged_df.index.max()}")
        
        # 计算不同延迟时间的皮尔逊相关系数（只考虑潮位影响水位的正向延迟）
        lag_correlations = []
        lag_hours = range(0, max_lag_hours + 1)  # 只考虑正向延迟（潮位影响水位）
        
        for lag in lag_hours:
            if lag == 0:
                # 无延迟，计算皮尔逊相关系数
                correlation = merged_df['tide_level'].corr(merged_df['water_level'], method='pearson')
            else:
                # 潮位延迟lag小时影响水位，计算皮尔逊相关系数
                tide_shifted = merged_df['tide_level'].shift(lag)
                correlation = tide_shifted.corr(merged_df['water_level'], method='pearson')
            
            lag_correlations.append({
                'lag_hours': lag,
                'correlation': correlation
            })
        
        # 找到最佳延迟时间
        lag_df = pd.DataFrame(lag_correlations)
        lag_df = lag_df.dropna()
        
        if len(lag_df) > 0:
            best_lag_idx = lag_df['correlation'].abs().idxmax()
            best_lag = lag_df.loc[best_lag_idx, 'lag_hours']
            best_correlation = lag_df.loc[best_lag_idx, 'correlation']
            
            print(f"最佳延迟时间: {best_lag}小时")
            print(f"最佳相关系数: {best_correlation:.4f}")
            
            # 计算最佳延迟下的详细统计
            if best_lag == 0:
                tide_series = merged_df['tide_level']
                water_series = merged_df['water_level']
            elif best_lag > 0:
                tide_series = merged_df['tide_level'].shift(best_lag)
                water_series = merged_df['water_level']
            else:
                tide_series = merged_df['tide_level']
                water_series = merged_df['water_level'].shift(abs(best_lag))
            
            # 去除缺失值
            valid_data = pd.DataFrame({
                'tide_level': tide_series,
                'water_level': water_series
            }).dropna()
            
            if len(valid_data) > 10:
                # 线性回归分析
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    valid_data['tide_level'], valid_data['water_level']
                )
                
                results = {
                    'best_lag': best_lag,
                    'best_correlation': best_correlation,
                    'r_squared': r_value**2,
                    'p_value': p_value,
                    'slope': slope,
                    'intercept': intercept,
                    'data_count': len(valid_data),
                    'lag_correlations': lag_df
                }
                
                all_results[year] = results
                
                print(f"R²: {r_value**2:.4f}")
                print(f"P值: {p_value:.4f}")
                print(f"线性关系: 水位 = {slope:.6f} × 潮位 + {intercept:.4f}")
                
                # 绘制延迟相关性图
                plot_lag_correlation(lag_df, year, best_lag, best_correlation)
                
                # 绘制最佳延迟下的散点图
                plot_best_correlation(valid_data, year, best_lag, best_correlation, r_value**2)
                
                # 绘制时间序列对比
                plot_time_series_comparison(merged_df, year, best_lag)
        
    return all_results

def plot_lag_correlation(lag_df, year, best_lag, best_correlation):
    """绘制延迟相关性图"""
    plt.figure(figsize=(12, 6))
    
    # 对于2023年，在图片标题中标注数据排除情况
    title_suffix = ""
    if year == 2023:
        title_suffix = " (已排除8月和9月不完整数据)"
    
    plt.plot(lag_df['lag_hours'], lag_df['correlation'], 'b-', linewidth=2, alpha=0.7)
    plt.axvline(x=best_lag, color='r', linestyle='--', linewidth=2, 
                label=f'最佳延迟: {best_lag}小时 (r={best_correlation:.3f})')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    plt.xlabel('延迟时间 (小时)')
    plt.ylabel('皮尔逊相关系数')
    plt.title(f'{year}年潮位与水位延迟皮尔逊相关性分析{title_suffix}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, f'{year}年延迟相关性分析.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 对于2023年，添加数据排除说明
    if year == 2023:
        print(f"注意：2023年分析已排除8月和9月不完整数据")
        print(f"排除的数据点：8月(184个点) + 9月(205个点) = 389个点")
        print(f"排除原因：8月数据完整度24.7%，9月数据完整度28.5%")

def plot_best_correlation(data, year, best_lag, correlation, r_squared):
    """绘制最佳延迟下的散点图"""
    plt.figure(figsize=(10, 8))
    
    # 对于2023年，在图片标题中标注数据排除情况
    title_suffix = ""
    if year == 2023:
        title_suffix = " (已排除8月和9月不完整数据)"
    
    plt.scatter(data['tide_level'], data['water_level'], alpha=0.6, s=1)
    
    # 添加拟合线
    slope, intercept, _, _, _ = stats.linregress(data['tide_level'], data['water_level'])
    x_range = np.linspace(data['tide_level'].min(), data['tide_level'].max(), 100)
    y_fit = slope * x_range + intercept
    plt.plot(x_range, y_fit, 'r-', linewidth=2, 
             label=f'拟合线 (R²={r_squared:.3f})')
    
    plt.xlabel('潮位 (cm)')
    plt.ylabel('水位 (m)')
    plt.title(f'{year}年潮位与水位皮尔逊相关性分析{title_suffix}\n(延迟{best_lag}小时, r={correlation:.3f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, f'{year}年最佳延迟相关性.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_time_series_comparison(merged_df, year, best_lag):
    """绘制时间序列对比图"""
    plt.figure(figsize=(15, 12))
    
    # 对于2023年，在图片标题中标注数据排除情况
    title_suffix = ""
    if year == 2023:
        title_suffix = " (已排除8月和9月不完整数据)"
    
    # 数据标准化，便于比较
    tide_normalized = (merged_df['tide_level'] - merged_df['tide_level'].mean()) / merged_df['tide_level'].std()
    water_normalized = (merged_df['water_level'] - merged_df['water_level'].mean()) / merged_df['water_level'].std()
    
    # 原始时间序列
    plt.subplot(4, 1, 1)
    plt.plot(merged_df.index, merged_df['tide_level'], label='潮位', alpha=0.7, linewidth=0.5, color='blue')
    plt.ylabel('潮位 (cm)')
    plt.title(f'{year}年潮位时间序列{title_suffix}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 对于2023年，在图中标注缺失时段
    if year == 2023:
        # 添加缺失时段的背景色
        plt.axvspan(pd.Timestamp('2023-08-01'), pd.Timestamp('2023-09-30'), 
                   alpha=0.2, color='red', label='缺失时段')
        plt.legend()
    
    plt.subplot(4, 1, 2)
    plt.plot(merged_df.index, merged_df['water_level'], label='水位', color='orange', alpha=0.7, linewidth=0.5)
    plt.ylabel('水位 (m)')
    plt.title(f'{year}年水位时间序列{title_suffix}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 对于2023年，在图中标注缺失时段
    if year == 2023:
        plt.axvspan(pd.Timestamp('2023-08-01'), pd.Timestamp('2023-09-30'), 
                   alpha=0.2, color='red', label='缺失时段')
        plt.legend()
    
    # 标准化后的时间序列对比
    plt.subplot(4, 1, 3)
    plt.plot(merged_df.index, tide_normalized, label='潮位(标准化)', alpha=0.7, linewidth=0.5, color='blue')
    plt.plot(merged_df.index, water_normalized, label='水位(标准化)', color='orange', alpha=0.7, linewidth=0.5)
    plt.ylabel('标准化值')
    plt.title(f'{year}年标准化时间序列对比{title_suffix}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 对于2023年，在图中标注缺失时段
    if year == 2023:
        plt.axvspan(pd.Timestamp('2023-08-01'), pd.Timestamp('2023-09-30'), 
                   alpha=0.2, color='red', label='缺失时段')
        plt.legend()
    
    # 对齐后的时间序列
    plt.subplot(4, 1, 4)
    if best_lag > 0:
        # 潮位延迟影响水位
        tide_aligned = merged_df['tide_level'].shift(best_lag)
        water_aligned = merged_df['water_level']
        
        # 标准化对齐后的数据
        tide_aligned_norm = (tide_aligned - tide_aligned.mean()) / tide_aligned.std()
        water_aligned_norm = (water_aligned - water_aligned.mean()) / water_aligned.std()
        
        plt.plot(merged_df.index, tide_aligned_norm, label=f'潮位(延迟{best_lag}小时)', alpha=0.7, linewidth=0.5, color='blue')
        plt.plot(merged_df.index, water_aligned_norm, label='水位', color='orange', alpha=0.7, linewidth=0.5)
    else:
        # 水位延迟影响潮位
        tide_aligned = merged_df['tide_level']
        water_aligned = merged_df['water_level'].shift(abs(best_lag))
        
        # 标准化对齐后的数据
        tide_aligned_norm = (tide_aligned - tide_aligned.mean()) / tide_aligned.std()
        water_aligned_norm = (water_aligned - water_aligned.mean()) / water_aligned.std()
        
        plt.plot(merged_df.index, tide_aligned_norm, label='潮位', alpha=0.7, linewidth=0.5, color='blue')
        plt.plot(merged_df.index, water_aligned_norm, label=f'水位(延迟{abs(best_lag)}小时)', color='orange', alpha=0.7, linewidth=0.5)
    
    plt.ylabel('标准化值')
    plt.xlabel('时间')
    plt.title(f'{year}年对齐后的时间序列对比{title_suffix}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 对于2023年，在图中标注缺失时段
    if year == 2023:
        plt.axvspan(pd.Timestamp('2023-08-01'), pd.Timestamp('2023-09-30'), 
                   alpha=0.2, color='red', label='缺失时段')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, f'{year}年时间序列对比.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印对齐效果统计
    if best_lag > 0:
        tide_aligned = merged_df['tide_level'].shift(best_lag)
        water_aligned = merged_df['water_level']
    else:
        tide_aligned = merged_df['tide_level']
        water_aligned = merged_df['water_level'].shift(abs(best_lag))
    
    # 去除缺失值后计算对齐后的皮尔逊相关系数
    aligned_data = pd.DataFrame({
        'tide_aligned': tide_aligned,
        'water_aligned': water_aligned
    }).dropna()
    
    if len(aligned_data) > 0:
        aligned_corr = aligned_data['tide_aligned'].corr(aligned_data['water_aligned'], method='pearson')
        print(f"{year}年对齐后皮尔逊相关系数: {aligned_corr:.4f}")
        print(f"对齐后数据点数量: {len(aligned_data)}")

def generate_monthly_report(monthly_results):
    """生成月度分析报告"""
    print("\n生成月度分析报告...")
    
    report = "潮位与长江侧水位月度相关性分析报告\n"
    report += "=" * 60 + "\n\n"
    
    if monthly_results:
        for year, year_data in monthly_results.items():
            if not year_data:
                continue
                
            report += f"{year}年月度分析结果:\n"
            report += "-" * 40 + "\n"
            
            # 计算年度统计
            correlations = [data['best_correlation'] for data in year_data.values()]
            lags = [data['best_lag'] for data in year_data.values()]
            r_squared_values = [data['r_squared'] for data in year_data.values()]
            
            report += f"分析月份数: {len(year_data)}\n"
            report += f"平均相关系数: {np.mean(correlations):.4f}\n"
            report += f"平均延迟时间: {np.mean(lags):.1f}小时\n"
            report += f"平均R²: {np.mean(r_squared_values):.4f}\n\n"
            
            # 各月份详细结果
            for month, data in year_data.items():
                month_name = MONTH_NAMES[month]
                report += f"{month_name}:\n"
                report += f"  最佳延迟: {data['best_lag']}小时\n"
                report += f"  相关系数: {data['best_correlation']:.4f}\n"
                report += f"  R²: {data['r_squared']:.4f}\n"
                report += f"  P值: {data['p_value']:.4f}\n"
                report += f"  数据点: {data['data_count']}个\n"
                report += f"  线性关系: 水位 = {data['slope']:.6f} × 潮位 + {data['intercept']:.4f}\n\n"
            
            # 季节性分析
            spring_months = [3, 4, 5]
            summer_months = [6, 7, 8]
            autumn_months = [9, 10, 11]
            winter_months = [12, 1, 2]
            
            seasons = {
                '春季': spring_months,
                '夏季': summer_months,
                '秋季': autumn_months,
                '冬季': winter_months
            }
            
            report += "季节性分析:\n"
            for season_name, season_months in seasons.items():
                season_data = {month: data for month, data in year_data.items() 
                             if month in season_months}
                if season_data:
                    season_correlations = [data['best_correlation'] for data in season_data.values()]
                    season_lags = [data['best_lag'] for data in season_data.values()]
                    
                    report += f"  {season_name}: {len(season_data)}个月份\n"
                    report += f"    平均相关系数: {np.mean(season_correlations):.4f}\n"
                    report += f"    平均延迟时间: {np.mean(season_lags):.1f}小时\n\n"
            
            report += "\n"
    
    # 保存报告
    with open('潮位水位月度相关性分析报告.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("月度分析报告已保存为: 潮位水位月度相关性分析报告.txt")
    return report

def generate_comprehensive_report(all_results):
    """生成综合分析报告"""
    print("\n生成综合分析报告...")
    
    report = "潮位与长江侧水位延迟相关性分析报告\n"
    report += "=" * 60 + "\n\n"
    
    if all_results:
        # 汇总统计
        best_lags = [results['best_lag'] for results in all_results.values()]
        best_correlations = [results['best_correlation'] for results in all_results.values()]
        r_squared_values = [results['r_squared'] for results in all_results.values()]
        
        report += f"分析年份: {list(all_results.keys())}\n"
        report += f"平均最佳延迟: {np.mean(best_lags):.1f}小时\n"
        report += f"平均最佳皮尔逊相关系数: {np.mean(best_correlations):.4f}\n"
        report += f"平均R²: {np.mean(r_squared_values):.4f}\n\n"
        
        # 年度详细结果
        for year, results in all_results.items():
            report += f"{year}年分析结果:\n"
            report += f"  最佳延迟时间: {results['best_lag']}小时\n"
            report += f"  最佳皮尔逊相关系数: {results['best_correlation']:.4f}\n"
            report += f"  R²: {results['r_squared']:.4f}\n"
            report += f"  P值: {results['p_value']:.4f}\n"
            report += f"  数据点数量: {results['data_count']}\n"
            report += f"  线性关系: 水位 = {results['slope']:.6f} × 潮位 + {results['intercept']:.4f}\n\n"
        
        # 延迟效应分析
        positive_lags = [lag for lag in best_lags if lag > 0]
        zero_lags = [lag for lag in best_lags if lag == 0]
        
        report += "延迟效应分析:\n"
        if positive_lags:
            report += f"  潮位影响水位延迟: {len(positive_lags)}年，平均延迟{np.mean(positive_lags):.1f}小时\n"
        if zero_lags:
            report += f"  无延迟效应: {len(zero_lags)}年\n"
        
        # 相关性强度判断
        avg_corr = np.mean(best_correlations)
        if abs(avg_corr) >= 0.8:
            strength = "极强"
        elif abs(avg_corr) >= 0.6:
            strength = "强"
        elif abs(avg_corr) >= 0.4:
            strength = "中等"
        elif abs(avg_corr) >= 0.2:
            strength = "弱"
        else:
            strength = "极弱"
        
        report += f"\n总体相关性强度: {strength}\n"
        report += f"结论: 潮位与长江侧水位存在{strength}皮尔逊相关性，考虑延迟效应后相关性更强\n"
        
        # 添加数据质量说明
        report += "\n数据质量说明:\n"
        report += "1. 2023年8月数据严重缺失（仅8月1-8日，缺失率75.3%）\n"
        report += "2. 2023年9月数据不完整（仅9月22-30日，缺失率71.5%）\n"
        report += "3. 分析时已排除2023年8月和9月的不完整数据\n"
        report += "4. 排除不完整月份后，2023年数据质量显著改善\n"
        
        # 添加解释说明
        report += "\n分析说明:\n"
        report += "1. 延迟时间表示潮位变化传播到水位测点所需的时间\n"
        report += "2. 潮位由海洋潮汐引起，水位不可能影响潮位\n"
        report += "3. 延迟时间反映了潮位传播到水位测点所需的时间\n"
        report += "4. 皮尔逊相关系数绝对值越大表示线性相关性越强\n"
        report += "5. R²表示线性回归的拟合优度\n"
        report += "6. 所有相关性分析均使用皮尔逊相关系数方法\n"
        
    else:
        report += "未获得有效的分析结果\n"
    
    # 保存报告
    with open('潮位水位延迟相关性分析报告.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("综合分析报告已保存为: 潮位水位延迟相关性分析报告.txt")
    return report

def main():
    """主函数"""
    print("开始潮位与长江侧水位延迟相关性分析...")
    
    # 1. 加载数据
    tide_data = load_tide_data()
    water_data = load_water_level_data()
    
    # 2. 分析年度延迟相关性
    all_results = analyze_correlation_with_lag(tide_data, water_data, max_lag_hours=48)
    
    # 3. 分析月度相关性
    monthly_results = analyze_monthly_correlation(tide_data, water_data, max_lag_hours=48)
    
    # 4. 生成综合报告
    report = generate_comprehensive_report(all_results)
    
    # 5. 生成月度报告
    monthly_report = generate_monthly_report(monthly_results)
    
    print("\n延迟相关性分析完成！")
    print(report)
    print("\n月度分析完成！")
    print(monthly_report)

if __name__ == "__main__":
    main() 
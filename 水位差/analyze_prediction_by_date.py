import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def analyze_prediction_by_date():
    """按日期分析水位差预测效果"""
    print("=" * 60)
    print("按日期分析水位差预测效果")
    print("=" * 60)
    
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 加载预测结果
    pred_file = os.path.join(script_dir, "预测结果_修正版", "水位差预测结果.csv")
    print(f"尝试加载预测文件: {pred_file}")
    print(f"文件是否存在: {os.path.exists(pred_file)}")
    
    df_pred = pd.read_csv(pred_file)
    df_pred['datetime'] = pd.to_datetime(df_pred['datetime'])
    
    # 加载实际数据
    actual_file = os.path.join(script_dir, "merged_yangtze_yunhe.csv")
    print(f"尝试加载实际数据文件: {actual_file}")
    print(f"文件是否存在: {os.path.exists(actual_file)}")
    
    df_actual = pd.read_csv(actual_file)
    df_actual['datetime'] = pd.to_datetime(df_actual['datetime'])
    df_actual.set_index('datetime', inplace=True)
    
    # 重命名列
    df_actual = df_actual.rename(columns={
        '长江侧水位': 'yangtze_level',
        '内河侧水位': 'yunhe_level'
    })
    
    # 计算实际水位差
    df_actual['actual_diff'] = np.abs(df_actual['yangtze_level'] - df_actual['yunhe_level'])
    
    print(f"预测数据点数: {len(df_pred)}")
    print(f"实际数据点数: {len(df_actual)}")
    
    # 按日期对齐数据
    aligned_data = align_data_by_date(df_pred, df_actual)
    
    if aligned_data is None:
        print("数据对齐失败")
        return
    
    df_pred_aligned, df_actual_aligned = aligned_data
    
    # 分析整体预测效果
    analyze_overall_performance(df_pred_aligned, df_actual_aligned)
    
    # 按日期分析预测效果
    analyze_daily_performance(df_pred_aligned, df_actual_aligned)
    
    # 创建可视化图表
    create_date_based_visualizations(df_pred_aligned, df_actual_aligned, script_dir)
    
    print("\n" + "=" * 60)
    print("按日期分析完成")
    print("=" * 60)

def align_data_by_date(df_pred, df_actual):
    """按日期对齐预测数据和实际数据"""
    print("按日期对齐数据...")
    
    # 将预测数据的时间设为索引
    df_pred_aligned = df_pred.set_index('datetime')
    
    # 找到共同的时间范围
    pred_start = df_pred_aligned.index.min()
    pred_end = df_pred_aligned.index.max()
    actual_start = df_actual.index.min()
    actual_end = df_actual.index.max()
    
    print(f"预测数据时间范围: {pred_start} 到 {pred_end}")
    print(f"实际数据时间范围: {actual_start} 到 {actual_end}")
    
    # 找到重叠的时间范围
    overlap_start = max(pred_start, actual_start)
    overlap_end = min(pred_end, actual_end)
    
    if overlap_start >= overlap_end:
        print("错误：预测数据和实际数据没有时间重叠")
        return None
    
    print(f"重叠时间范围: {overlap_start} 到 {overlap_end}")
    
    # 截取重叠部分的数据
    df_pred_aligned = df_pred_aligned[overlap_start:overlap_end]
    df_actual_aligned = df_actual[overlap_start:overlap_end]
    
    print(f"对齐后预测数据点数: {len(df_pred_aligned)}")
    print(f"对齐后实际数据点数: {len(df_actual_aligned)}")
    
    return df_pred_aligned, df_actual_aligned

def analyze_overall_performance(df_pred, df_actual):
    """分析整体预测效果"""
    print("\n=== 整体预测效果分析 ===")
    
    time_steps = ['1h', '2h', '3h']
    
    for step in time_steps:
        pred_col = f'水位差_{step}'
        pred_values = df_pred[pred_col].values
        actual_values = df_actual['actual_diff'].values
        
        # 确保长度一致
        min_len = min(len(pred_values), len(actual_values))
        pred_values = pred_values[:min_len]
        actual_values = actual_values[:min_len]
        
        # 计算评估指标
        mse = mean_squared_error(actual_values, pred_values)
        mae = mean_absolute_error(actual_values, pred_values)
        r2 = r2_score(actual_values, pred_values)
        
        # 计算相对误差（避免除零）
        non_zero_mask = actual_values > 0.001
        if np.sum(non_zero_mask) > 0:
            relative_error = np.mean(np.abs(pred_values[non_zero_mask] - actual_values[non_zero_mask]) / actual_values[non_zero_mask]) * 100
        else:
            relative_error = np.inf
        
        print(f"\n{step}预测整体效果:")
        print(f"  均方误差 (MSE): {mse:.4f}")
        print(f"  平均绝对误差 (MAE): {mae:.4f}")
        print(f"  决定系数 (R²): {r2:.4f}")
        if relative_error != np.inf:
            print(f"  平均相对误差: {relative_error:.2f}%")
        else:
            print(f"  平均相对误差: 无法计算")

def analyze_daily_performance(df_pred, df_actual):
    """按日期分析预测效果"""
    print("\n=== 按日期分析预测效果 ===")
    
    # 按日期分组
    df_pred['date'] = df_pred.index.date
    df_actual['date'] = df_actual.index.date
    
    # 获取所有日期
    all_dates = sorted(set(df_pred['date']) & set(df_actual['date']))
    
    print(f"共有 {len(all_dates)} 个日期")
    
    # 分析每个日期的预测效果
    daily_metrics = []
    
    for date in all_dates[:10]:  # 只分析前10天
        pred_date = df_pred[df_pred['date'] == date]
        actual_date = df_actual[df_actual['date'] == date]
        
        if len(pred_date) == 0 or len(actual_date) == 0:
            continue
        
        # 计算1小时预测的MAE
        pred_1h = pred_date['水位差_1h'].values
        actual_diff = actual_date['actual_diff'].values
        
        min_len = min(len(pred_1h), len(actual_diff))
        if min_len > 0:
            mae = mean_absolute_error(actual_diff[:min_len], pred_1h[:min_len])
            daily_metrics.append({
                'date': date,
                'mae': mae,
                'data_points': min_len
            })
    
    # 显示每日预测效果
    print("\n前10天每日预测效果 (1小时预测MAE):")
    print("日期\t\t\tMAE\t\t数据点数")
    print("-" * 50)
    
    for metric in daily_metrics:
        print(f"{metric['date']}\t{metric['mae']:.4f}\t\t{metric['data_points']}")
    
    # 计算统计信息
    maes = [m['mae'] for m in daily_metrics]
    if maes:
        print(f"\n每日MAE统计:")
        print(f"  平均MAE: {np.mean(maes):.4f}")
        print(f"  最小MAE: {np.min(maes):.4f}")
        print(f"  最大MAE: {np.max(maes):.4f}")
        print(f"  MAE标准差: {np.std(maes):.4f}")

def create_date_based_visualizations(df_pred, df_actual, script_dir):
    """创建基于日期的可视化图表"""
    print("\n=== 创建基于日期的可视化图表 ===")
    
    # 1. 时间序列对比图（前7天）
    create_time_series_comparison(df_pred, df_actual, script_dir)
    
    # 2. 每日预测误差箱线图
    create_daily_error_boxplot(df_pred, df_actual, script_dir)
    
    # 3. 预测值 vs 实际值散点图
    create_prediction_vs_actual_scatter(df_pred, df_actual, script_dir)
    
    # 4. 误差时间序列图（分别绘制）
    create_error_time_series(df_pred, df_actual, script_dir)
    
    # 5. 月度预测效果对比图
    create_monthly_performance_comparison(df_pred, df_actual, script_dir)
    
    # 6. 预测误差分布直方图（分别绘制）
    create_error_distribution_histogram(df_pred, df_actual, script_dir)
    
    print("所有图表已生成完成")

def create_time_series_comparison(df_pred, df_actual, script_dir):
    """创建时间序列对比图"""
    plt.figure(figsize=(14, 8))
    
    sample_days = 7
    sample_data = df_pred.head(sample_days * 24)
    
    time_points = range(len(sample_data))
    actual_sample = df_actual['actual_diff'].iloc[:len(sample_data)].values
    
    plt.plot(time_points, actual_sample, 'k-', label='实际水位差', linewidth=2)
    plt.plot(time_points, sample_data['水位差_1h'].values, 'b-', alpha=0.8, label='1h预测', linewidth=1.5)
    plt.plot(time_points, sample_data['水位差_2h'].values, 'g-', alpha=0.8, label='2h预测', linewidth=1.5)
    plt.plot(time_points, sample_data['水位差_3h'].values, 'r-', alpha=0.8, label='3h预测', linewidth=1.5)
    
    plt.xlabel('时间点 (小时)', fontsize=12)
    plt.ylabel('水位差 (米)', fontsize=12)
    plt.title(f'前{sample_days}天水位差预测时间序列对比', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    output_file = os.path.join(script_dir, "预测结果_修正版", "1_时间序列对比图.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"时间序列对比图已保存到: {output_file}")

def create_daily_error_boxplot(df_pred, df_actual, script_dir):
    """创建每日预测误差箱线图"""
    plt.figure(figsize=(14, 8))
    
    df_pred_copy = df_pred.copy()
    df_actual_copy = df_actual.copy()
    df_pred_copy['date'] = df_pred_copy.index.date
    df_actual_copy['date'] = df_actual_copy.index.date
    
    daily_errors = []
    dates = []
    
    for date in sorted(set(df_pred_copy['date']))[:15]:  # 前15天
        pred_date = df_pred_copy[df_pred_copy['date'] == date]
        actual_date = df_actual_copy[df_actual_copy['date'] == date]
        
        if len(pred_date) > 0 and len(actual_date) > 0:
            pred_1h = pred_date['水位差_1h'].values
            actual_diff = actual_date['actual_diff'].values
            
            min_len = min(len(pred_1h), len(actual_diff))
            if min_len > 0:
                errors = pred_1h[:min_len] - actual_diff[:min_len]
                daily_errors.append(errors)
                dates.append(str(date)[5:])  # 只显示月-日
    
    if daily_errors:
        plt.boxplot(daily_errors, tick_labels=dates)
        plt.xlabel('日期 (月-日)', fontsize=12)
        plt.ylabel('预测误差 (米)', fontsize=12)
        plt.title('每日预测误差分布箱线图', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        output_file = os.path.join(script_dir, "预测结果_修正版", "2_每日预测误差箱线图.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"每日预测误差箱线图已保存到: {output_file}")

def create_prediction_vs_actual_scatter(df_pred, df_actual, script_dir):
    """创建预测值 vs 实际值散点图（分别绘制）"""
    time_steps = ['1h', '2h', '3h']
    colors = ['blue', 'green', 'red']
    
    for i, step in enumerate(time_steps):
        plt.figure(figsize=(10, 8))
        
        pred_col = f'水位差_{step}'
        pred_values = df_pred[pred_col].values
        actual_values = df_actual['actual_diff'].values
        
        min_len = min(len(pred_values), len(actual_values))
        
        # 绘制散点图
        plt.scatter(actual_values[:min_len], pred_values[:min_len], alpha=0.6, 
                   color=colors[i], s=20)
        
        # 添加对角线
        min_val = 0
        max_val = max(actual_values.max(), pred_values.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, linewidth=2, label='理想线')
        
        # 计算R²值
        r2 = r2_score(actual_values[:min_len], pred_values[:min_len])
        
        plt.xlabel('实际水位差 (米)', fontsize=12)
        plt.ylabel('预测水位差 (米)', fontsize=12)
        plt.title(f'{step}预测值 vs 实际值散点图 (R^2 = {r2:.4f})', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        output_file = os.path.join(script_dir, "预测结果_修正版", f"3_{step}预测值vs实际值散点图.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"{step}预测值vs实际值散点图已保存到: {output_file}")

def create_error_time_series(df_pred, df_actual, script_dir):
    """创建误差时间序列图（分别绘制）"""
    time_steps = ['1h', '2h', '3h']
    colors = ['blue', 'green', 'red']
    
    for i, step in enumerate(time_steps):
        plt.figure(figsize=(14, 8))
        
        pred_col = f'水位差_{step}'
        pred_values = df_pred[pred_col].values
        actual_values = df_actual['actual_diff'].values
        min_len = min(len(pred_values), len(actual_values))
        
        errors = pred_values[:min_len] - actual_values[:min_len]
        time_points = range(len(errors))
        
        plt.plot(time_points, errors, color=colors[i], alpha=0.7, linewidth=0.8)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.7, linewidth=2)
        plt.xlabel('时间点 (小时)', fontsize=12)
        plt.ylabel('预测误差 (米)', fontsize=12)
        plt.title(f'{step}预测误差时间序列', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 添加统计信息
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        plt.text(0.02, 0.98, f'平均误差: {mean_error:.4f}\n标准差: {std_error:.4f}', 
                 transform=plt.gca().transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        output_file = os.path.join(script_dir, "预测结果_修正版", f"4_{step}误差时间序列图.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"{step}误差时间序列图已保存到: {output_file}")

def create_monthly_performance_comparison(df_pred, df_actual, script_dir):
    """创建月度预测效果对比图"""
    plt.figure(figsize=(14, 8))
    
    df_pred_copy = df_pred.copy()
    df_actual_copy = df_actual.copy()
    df_pred_copy['month'] = df_pred_copy.index.month
    df_actual_copy['month'] = df_actual_copy.index.month
    
    monthly_mae = []
    months = []
    
    for month in range(1, 13):
        pred_month = df_pred_copy[df_pred_copy['month'] == month]
        actual_month = df_actual_copy[df_actual_copy['month'] == month]
        
        if len(pred_month) > 0 and len(actual_month) > 0:
            pred_1h = pred_month['水位差_1h'].values
            actual_diff = actual_month['actual_diff'].values
            
            min_len = min(len(pred_1h), len(actual_diff))
            if min_len > 0:
                mae = mean_absolute_error(actual_diff[:min_len], pred_1h[:min_len])
                monthly_mae.append(mae)
                months.append(month)
    
    month_names = ['1月', '2月', '3月', '4月', '5月', '6月', 
                   '7月', '8月', '9月', '10月', '11月', '12月']
    
    plt.bar(months, monthly_mae, color='skyblue', alpha=0.8, edgecolor='navy', linewidth=1)
    plt.xlabel('月份', fontsize=12)
    plt.ylabel('平均绝对误差 (MAE)', fontsize=12)
    plt.title('月度预测效果对比 (1小时预测)', fontsize=14, fontweight='bold')
    plt.xticks(months, [month_names[m-1] for m in months])
    plt.grid(True, alpha=0.3, axis='y')
    
    output_file = os.path.join(script_dir, "预测结果_修正版", "5_月度预测效果对比图.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"月度预测效果对比图已保存到: {output_file}")

def create_error_distribution_histogram(df_pred, df_actual, script_dir):
    """创建预测误差分布直方图（分别绘制）"""
    time_steps = ['1h', '2h', '3h']
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    
    for i, step in enumerate(time_steps):
        plt.figure(figsize=(14, 8))
        
        pred_col = f'水位差_{step}'
        pred_values = df_pred[pred_col].values
        actual_values = df_actual['actual_diff'].values
        min_len = min(len(pred_values), len(actual_values))
        
        errors = pred_values[:min_len] - actual_values[:min_len]
        
        plt.hist(errors, bins=50, alpha=0.7, color=colors[i], edgecolor='black', linewidth=0.5)
        plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='零误差线')
        plt.xlabel('预测误差 (米)', fontsize=12)
        plt.ylabel('频次', fontsize=12)
        plt.title(f'{step}预测误差分布直方图', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # 添加统计信息
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        plt.text(0.02, 0.98, f'平均误差: {mean_error:.4f}\n标准差: {std_error:.4f}', 
                 transform=plt.gca().transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        output_file = os.path.join(script_dir, "预测结果_修正版", f"6_{step}预测误差分布直方图.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"{step}预测误差分布直方图已保存到: {output_file}")

if __name__ == '__main__':
    analyze_prediction_by_date() 
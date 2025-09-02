import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def load_data(file_path):
    """加载CSV数据"""
    try:
        df = pd.read_csv(file_path, encoding='gbk', header=0)
    except:
        try:
            df = pd.read_csv(file_path, encoding='utf-8', header=0)
        except:
            df = pd.read_csv(file_path, encoding='latin1', header=0)
    
    # 重命名列
    df.columns = ['监测点', '日期', '时间', '水位值']
    
    # 合并日期和时间
    df['日期时间'] = pd.to_datetime(df['日期'] + ' ' + df['时间'])
    
    # 转换水位值为数值类型
    df['水位值'] = pd.to_numeric(df['水位值'], errors='coerce')
    
    return df

def get_water_levels(df):
    """获取各种水位数据"""
    # 分离不同类型的水位数据
    inland_data = df[df['监测点'].str.contains('内河侧', na=False)].copy()
    yangtze_data = df[df['监测点'].str.contains('长江侧', na=False)].copy()
    upstream_in_data = df[df['监测点'].str.contains('上游闸内', na=False)].copy()
    upstream_out_data = df[df['监测点'].str.contains('上游闸外', na=False)].copy()
    downstream_in_data = df[df['监测点'].str.contains('下游闸内', na=False)].copy()
    downstream_out_data = df[df['监测点'].str.contains('下游闸外', na=False)].copy()
    
    # 按时间排序
    inland_data = inland_data.sort_values('日期时间')
    yangtze_data = yangtze_data.sort_values('日期时间')
    upstream_in_data = upstream_in_data.sort_values('日期时间')
    upstream_out_data = upstream_out_data.sort_values('日期时间')
    downstream_in_data = downstream_in_data.sort_values('日期时间')
    downstream_out_data = downstream_out_data.sort_values('日期时间')
    
    return {
        '内河': inland_data,
        '长江': yangtze_data,
        '上游闸内': upstream_in_data,
        '上游闸外': upstream_out_data,
        '下游闸内': downstream_in_data,
        '下游闸外': downstream_out_data
    }

def interpolate_level(data, time_point):
    """对指定时间点进行水位插值"""
    if data.empty:
        return np.nan
    
    before = data[data['日期时间'] <= time_point]
    after = data[data['日期时间'] > time_point]
    
    if not before.empty and not after.empty:
        # 线性插值
        before_time = before['日期时间'].iloc[-1]
        after_time = after['日期时间'].iloc[0]
        before_level = before['水位值'].iloc[-1]
        after_level = after['水位值'].iloc[0]
        
        if pd.notna(before_level) and pd.notna(after_level):
            ratio = (time_point - before_time) / (after_time - before_time)
            return before_level + ratio * (after_level - before_level)
        else:
            return np.nan
    elif not before.empty:
        return before['水位值'].iloc[-1]
    elif not after.empty:
        return after['水位值'].iloc[0]
    else:
        return np.nan

def calculate_hourly_data(df):
    """计算每小时的水位数据"""
    water_levels = get_water_levels(df)
    
    # 创建时间序列，每小时一个数据点
    time_range = pd.date_range(
        start=df['日期时间'].min(),
        end=df['日期时间'].max(),
        freq='H'
    )
    
    hourly_data = []
    
    for time_point in time_range:
        # 获取各监测点的水位值
        inland_level = interpolate_level(water_levels['内河'], time_point)
        yangtze_level = interpolate_level(water_levels['长江'], time_point)
        upstream_in_level = interpolate_level(water_levels['上游闸内'], time_point)
        upstream_out_level = interpolate_level(water_levels['上游闸外'], time_point)
        downstream_in_level = interpolate_level(water_levels['下游闸内'], time_point)
        downstream_out_level = interpolate_level(water_levels['下游闸外'], time_point)
        
        # 计算水位差
        inland_yangtze_diff = abs(inland_level - yangtze_level) if pd.notna(inland_level) and pd.notna(yangtze_level) else np.nan
        upstream_diff = abs(upstream_in_level - upstream_out_level) if pd.notna(upstream_in_level) and pd.notna(upstream_out_level) else np.nan
        downstream_diff = abs(downstream_in_level - downstream_out_level) if pd.notna(downstream_in_level) and pd.notna(downstream_out_level) else np.nan
        
        hourly_data.append({
            '时间': time_point,
            '内河水位': inland_level,
            '长江水位': yangtze_level,
            '上游闸内水位': upstream_in_level,
            '上游闸外水位': upstream_out_level,
            '下游闸内水位': downstream_in_level,
            '下游闸外水位': downstream_out_level,
            '内河长江水位差': inland_yangtze_diff,
            '上游闸内外水位差': upstream_diff,
            '下游闸内外水位差': downstream_diff
        })
    
    return pd.DataFrame(hourly_data)

def check_opening_conditions(hourly_df):
    """检查船闸开通条件
    条件1：内河与长江水位差小于0.3m
    条件2：上游闸内外水位差小于0.3m
    条件3：下游闸内外水位差小于0.3m
    条件4：以上条件持续1小时
    """
    # 检查每个小时是否满足条件
    condition1 = hourly_df['内河长江水位差'] < 0.3
    condition2 = hourly_df['上游闸内外水位差'] < 0.1
    condition3 = hourly_df['下游闸内外水位差'] < 0.1
    
    # 综合条件：同时满足所有条件
    all_conditions = condition1 & condition2 & condition3
    
    # 寻找连续满足条件的时间段
    opening_periods = []
    current_start = None
    current_duration = 0
    
    for i, (time, is_safe) in enumerate(zip(hourly_df['时间'], all_conditions)):
        if is_safe and pd.notna(hourly_df['内河长江水位差'].iloc[i]) and pd.notna(hourly_df['上游闸内外水位差'].iloc[i]) and pd.notna(hourly_df['下游闸内外水位差'].iloc[i]):
            if current_start is None:
                current_start = time
            current_duration += 1
        else:
            if current_start is not None and current_duration >= 1:  # 至少持续1小时
                # 计算该时间段的平均水位差
                period_data = hourly_df.iloc[i-current_duration:i]
                opening_periods.append({
                    '开始时间': current_start,
                    '结束时间': hourly_df['时间'].iloc[i-1],
                    '持续时间(小时)': current_duration,
                    '平均内河长江水位差': period_data['内河长江水位差'].mean(),
                    '平均上游闸内外水位差': period_data['上游闸内外水位差'].mean(),
                    '平均下游闸内外水位差': period_data['下游闸内外水位差'].mean()
                })
            current_start = None
            current_duration = 0
    
    # 处理最后一个时间段
    if current_start is not None and current_duration >= 1:
        period_data = hourly_df.iloc[-current_duration:]
        opening_periods.append({
            '开始时间': current_start,
            '结束时间': hourly_df['时间'].iloc[-1],
            '持续时间(小时)': current_duration,
            '平均内河长江水位差': period_data['内河长江水位差'].mean(),
            '平均上游闸内外水位差': period_data['上游闸内外水位差'].mean(),
            '平均下游闸内外水位差': period_data['下游闸内外水位差'].mean()
        })
    
    return opening_periods

def save_results(opening_periods, hourly_df, output_file="船闸开通分析结果.txt"):
    """保存分析结果"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=== 船闸开通分析结果 ===\n")
        f.write(f"分析时间：{hourly_df['时间'].min()} 到 {hourly_df['时间'].max()}\n")
        f.write(f"满足开通条件的时间段数量：{len(opening_periods)}\n\n")
        
        # 开通条件说明
        f.write("=== 开通条件 ===\n")
        f.write("1. 内河与长江水位差小于0.3m\n")
        f.write("2. 上游闸内外水位差小于0.1m\n")
        f.write("3. 下游闸内外水位差小于0.1m\n")
        f.write("4. 以上条件持续至少1小时\n\n")
        
        if opening_periods:
            f.write("满足开通条件的时间段：\n")
            f.write("=" * 80 + "\n")
            for i, period in enumerate(opening_periods, 1):
                f.write(f"\n时间段 {i}:\n")
                f.write(f"  开始时间：{period['开始时间']}\n")
                f.write(f"  结束时间：{period['结束时间']}\n")
                f.write(f"  持续时间：{period['持续时间(小时)']} 小时\n")
                f.write(f"  平均内河长江水位差：{period['平均内河长江水位差']:.3f} 米\n")
                f.write(f"  平均上游闸内外水位差：{period['平均上游闸内外水位差']:.3f} 米\n")
                f.write(f"  平均下游闸内外水位差：{period['平均下游闸内外水位差']:.3f} 米\n")
                f.write("-" * 50 + "\n")
        else:
            f.write("没有满足开通条件的时间段\n")
        
        # 统计信息
        valid_data = hourly_df.dropna(subset=['内河长江水位差', '上游闸内外水位差', '下游闸内外水位差'])
        total_hours = len(valid_data)
        
        f.write(f"\n=== 统计信息 ===\n")
        f.write(f"总分析小时数：{len(hourly_df)}\n")
        f.write(f"有效数据小时数：{total_hours}\n")
        
        if total_hours > 0:
            condition1_hours = len(valid_data[valid_data['内河长江水位差'] < 0.3])
            condition2_hours = len(valid_data[valid_data['上游闸内外水位差'] < 0.1])
            condition3_hours = len(valid_data[valid_data['下游闸内外水位差'] < 0.1])
            all_conditions_hours = len(valid_data[(valid_data['内河长江水位差'] < 0.3) & 
                                                 (valid_data['上游闸内外水位差'] < 0.1) & 
                                                 (valid_data['下游闸内外水位差'] < 0.1)])
            
            f.write(f"满足条件1（内河长江水位差<0.3m）的时间比例：{condition1_hours/total_hours*100:.1f}% ({condition1_hours}/{total_hours} 小时)\n")
            f.write(f"满足条件2（上游闸内外水位差<0.1m）的时间比例：{condition2_hours/total_hours*100:.1f}% ({condition2_hours}/{total_hours} 小时)\n")
            f.write(f"满足条件3（下游闸内外水位差<0.1m）的时间比例：{condition3_hours/total_hours*100:.1f}% ({condition3_hours}/{total_hours} 小时)\n")
            f.write(f"同时满足所有条件的时间比例：{all_conditions_hours/total_hours*100:.1f}% ({all_conditions_hours}/{total_hours} 小时)\n")
        else:
            f.write("没有有效数据，无法计算统计信息\n")
    
    print(f"分析结果已保存到文件：{output_file}")

def save_periods_csv(opening_periods, output_file="船闸开通时间段.csv"):
    """将时间段保存为CSV格式"""
    if opening_periods:
        periods_df = pd.DataFrame(opening_periods)
        periods_df['开始时间'] = periods_df['开始时间'].dt.strftime('%Y-%m-%d %H:%M:%S')
        periods_df['结束时间'] = periods_df['结束时间'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # 如果有月份信息，添加到CSV中
        if '月份' in periods_df.columns:
            # 重新排列列的顺序，让月份在前面
            columns = ['月份', '开始时间', '结束时间', '持续时间(小时)', 
                      '平均内河长江水位差', '平均上游闸内外水位差', '平均下游闸内外水位差']
            periods_df = periods_df[columns]
        
        periods_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"时间段数据已保存到CSV文件：{output_file}")
    else:
        print("没有满足条件的时间段，未创建CSV文件")

def analyze_ship_lock(file_path):
    """分析单个文件的船闸数据"""
    if not os.path.exists(file_path):
        print(f"错误：找不到文件 {file_path}")
        return None, None
    
    print(f"正在加载船闸数据：{file_path}")
    df = load_data(file_path)
    
    print(f"数据加载完成，共 {len(df)} 条记录")
    print(f"数据时间范围：{df['日期时间'].min()} 到 {df['日期时间'].max()}")
    
    print("正在计算每小时水位数据...")
    hourly_df = calculate_hourly_data(df)
    
    print("正在检查船闸开通条件...")
    opening_periods = check_opening_conditions(hourly_df)
    
    return opening_periods, hourly_df

def analyze_full_year_2024():
    """分析2024年全年数据"""
    print("=== 2024年全年船闸开通分析 ===")
    
    # 定义所有月份的文件路径
    month_files = [
        "2023/2023-1-1.csv",
        "2023/2023-2-1.csv",
        "2023/2023-3-1.csv",
        "2023/2023-4-1.csv",
        "2023/2023-5-1.csv",
        "2023/2023-6-1.csv",
        "2023/2023-7-1.csv",
        "2023/2023-8-1.csv",
        "2023/2023-9-1.csv",
        "2023/2023-10-1.csv",
        "2023/2023-11-1.csv",
        "2023/2023-12-1.csv"
    ]
    
    all_opening_periods = []
    all_hourly_data = []
    
    # 逐月分析
    for i, file_path in enumerate(month_files, 1):
        print(f"\n--- 分析第{i}个月数据 ---")
        
        opening_periods, hourly_df = analyze_ship_lock(file_path)
        
        if opening_periods is not None and hourly_df is not None:
            # 为每个时间段添加月份信息
            for period in opening_periods:
                period['月份'] = i
            
            all_opening_periods.extend(opening_periods)
            all_hourly_data.append(hourly_df)
            
            print(f"第{i}个月找到 {len(opening_periods)} 个满足条件的时间段")
        else:
            print(f"第{i}个月数据加载失败")
    
    # 合并所有小时数据
    if all_hourly_data:
        combined_hourly_df = pd.concat(all_hourly_data, ignore_index=True)
        combined_hourly_df = combined_hourly_df.sort_values('时间')
        
        # 输出全年汇总结果
        print(f"\n=== 2024年全年船闸开通分析结果 ===")
        print(f"总分析时间：{combined_hourly_df['时间'].min()} 到 {combined_hourly_df['时间'].max()}")
        print(f"全年满足开通条件的时间段数量：{len(all_opening_periods)}")
        
        if all_opening_periods:
            print("\n全年满足开通条件的时间段（按时间顺序）：")
            for i, period in enumerate(all_opening_periods, 1):
                print(f"\n时间段 {i} (第{period['月份']}个月):")
                print(f"  开始时间：{period['开始时间']}")
                print(f"  结束时间：{period['结束时间']}")
                print(f"  持续时间：{period['持续时间(小时)']} 小时")
                print(f"  平均内河长江水位差：{period['平均内河长江水位差']:.3f} 米")
                print(f"  平均上游闸内外水位差：{period['平均上游闸内外水位差']:.3f} 米")
                print(f"  平均下游闸内外水位差：{period['平均下游闸内外水位差']:.3f} 米")
        else:
            print("\n全年没有满足开通条件的时间段")
        
        # 全年统计信息
        valid_data = combined_hourly_df.dropna(subset=['内河长江水位差', '上游闸内外水位差', '下游闸内外水位差'])
        total_hours = len(valid_data)
        
        print(f"\n=== 全年统计信息 ===")
        print(f"总分析小时数：{len(combined_hourly_df)}")
        print(f"有效数据小时数：{total_hours}")
        
        if total_hours > 0:
            condition1_hours = len(valid_data[valid_data['内河长江水位差'] < 0.3])
            condition2_hours = len(valid_data[valid_data['上游闸内外水位差'] < 0.1])
            condition3_hours = len(valid_data[valid_data['下游闸内外水位差'] < 0.1])
            all_conditions_hours = len(valid_data[(valid_data['内河长江水位差'] < 0.3) & 
                                                 (valid_data['上游闸内外水位差'] < 0.1) & 
                                                 (valid_data['下游闸内外水位差'] < 0.1)])
            
            print(f"满足条件1（内河长江水位差<0.3m）的时间比例：{condition1_hours/total_hours*100:.1f}% ({condition1_hours}/{total_hours} 小时)")
            print(f"满足条件2（上游闸内外水位差<0.1m）的时间比例：{condition2_hours/total_hours*100:.1f}% ({condition2_hours}/{total_hours} 小时)")
            print(f"满足条件3（下游闸内外水位差<0.1m）的时间比例：{condition3_hours/total_hours*100:.1f}% ({condition3_hours}/{total_hours} 小时)")
            print(f"同时满足所有条件的时间比例：{all_conditions_hours/total_hours*100:.1f}% ({all_conditions_hours}/{total_hours} 小时)")
        else:
            print("没有有效数据，无法计算统计信息")
        
        # 保存全年结果
        print("\n正在保存全年分析结果...")
        save_results(all_opening_periods, combined_hourly_df, "2023年全年船闸开通分析结果.txt")
        save_periods_csv(all_opening_periods, "2023年全年船闸开通时间段.csv")
        
        return all_opening_periods, combined_hourly_df
    else:
        print("没有成功加载任何数据")
        return None, None

if __name__ == "__main__":
    # 分析2024年全年数据
    analyze_full_year_2024() 
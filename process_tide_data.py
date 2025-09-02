import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def process_tide_data():
    """处理三年潮位数据，只保留每天各时间段的潮位数据"""
    
    # 创建输出文件夹
    output_dir = "潮位数据处理"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for year in [2022, 2023, 2024]:
        print(f"\n=== 处理{year}年潮位数据 ===")
        
        try:
            # 读取原始潮位数据
            file_path = f'潮位{year}.xls'
            df = pd.read_excel(file_path)
            
            print(f"原始数据形状: {df.shape}")
            print(f"列名: {list(df.columns)}")
            
            # 重新整理数据
            processed_data = []
            
            # 遍历数据行
            for i, row in df.iterrows():
                # 第一列是时间
                time_val = row.iloc[0]
                
                # 检查其他列是否有潮位数据
                for col_idx in [1, 2, 3]:  # 检查第2、3、4列
                    if col_idx >= len(row):
                        continue
                    
                    col_val = row.iloc[col_idx]
                    
                    # 跳过空值
                    if pd.isna(col_val):
                        continue
                    
                    # 检查是否是时间格式（跳过）
                    if isinstance(col_val, str):
                        if ':' in col_val and len(col_val) <= 10:
                            continue  # 跳过时间格式
                        if 'cm' in col_val:
                            # 这是潮位值
                            try:
                                tide_level = float(col_val.replace('cm', ''))
                                processed_data.append({
                                    'datetime': time_val,
                                    'tide_level': tide_level
                                })
                                break  # 找到潮位值后跳出列循环
                            except:
                                continue
                    elif isinstance(col_val, (int, float)) and col_val > 0:
                        # 这可能是潮位值
                        processed_data.append({
                            'datetime': time_val,
                            'tide_level': col_val
                        })
                        break  # 找到潮位值后跳出列循环
            
            if processed_data:
                # 创建处理后的数据框
                processed_df = pd.DataFrame(processed_data)
                processed_df['datetime'] = pd.to_datetime(processed_df['datetime'])
                
                # 按时间排序
                processed_df = processed_df.sort_values('datetime')
                
                # 去除重复的时间点
                processed_df = processed_df.drop_duplicates(subset=['datetime'])
                
                print(f"处理后数据: {len(processed_df)}条记录")
                print(f"时间范围: {processed_df['datetime'].min()} 到 {processed_df['datetime'].max()}")
                print(f"潮位范围: {processed_df['tide_level'].min():.1f} - {processed_df['tide_level'].max():.1f} cm")
                
                # 保存处理后的数据
                output_file = os.path.join(output_dir, f'{year}年潮位数据_处理后.csv')
                processed_df.to_csv(output_file, index=False, encoding='utf-8-sig')
                print(f"数据已保存到: {output_file}")
                
                # 显示前10行数据
                print(f"\n前10行数据:")
                print(processed_df.head(10))
                
                # 统计每天的数据点数量
                daily_counts = processed_df.groupby(processed_df['datetime'].dt.date).size()
                print(f"\n每天数据点统计:")
                print(f"平均每天数据点: {daily_counts.mean():.1f}")
                print(f"最多每天数据点: {daily_counts.max()}")
                print(f"最少每天数据点: {daily_counts.min()}")
                
            else:
                print(f"{year}年未找到有效的潮位数据")
                
        except Exception as e:
            print(f"处理{year}年数据失败: {e}")
            import traceback
            traceback.print_exc()

def create_hourly_tide_data():
    """创建按小时整理的潮位数据"""
    print("\n=== 创建按小时整理的潮位数据 ===")
    
    output_dir = "潮位数据处理"
    
    for year in [2022, 2023, 2024]:
        try:
            # 读取处理后的数据
            input_file = os.path.join(output_dir, f'{year}年潮位数据_处理后.csv')
            if not os.path.exists(input_file):
                print(f"未找到{year}年处理后的数据文件")
                continue
                
            df = pd.read_csv(input_file)
            df['datetime'] = pd.to_datetime(df['datetime'])
            
            # 设置时间索引
            df.set_index('datetime', inplace=True)
            
            # 重采样到小时级别
            hourly_df = df.resample('H').mean()
            
            # 去除缺失值过多的行
            hourly_df = hourly_df.dropna()
            
            print(f"{year}年小时级数据: {len(hourly_df)}条记录")
            
            # 保存小时级数据
            hourly_file = os.path.join(output_dir, f'{year}年潮位数据_小时级.csv')
            hourly_df.to_csv(hourly_file, encoding='utf-8-sig')
            print(f"小时级数据已保存到: {hourly_file}")
            
        except Exception as e:
            print(f"创建{year}年小时级数据失败: {e}")

def generate_summary_report():
    """生成数据汇总报告"""
    print("\n=== 生成数据汇总报告 ===")
    
    output_dir = "潮位数据处理"
    report = "潮位数据处理汇总报告\n"
    report += "=" * 50 + "\n\n"
    
    for year in [2022, 2023, 2024]:
        try:
            # 读取处理后的数据
            input_file = os.path.join(output_dir, f'{year}年潮位数据_处理后.csv')
            if not os.path.exists(input_file):
                report += f"{year}年: 未找到处理后的数据文件\n\n"
                continue
                
            df = pd.read_csv(input_file)
            df['datetime'] = pd.to_datetime(df['datetime'])
            
            report += f"{year}年潮位数据统计:\n"
            report += f"  总记录数: {len(df)}\n"
            report += f"  时间范围: {df['datetime'].min()} 到 {df['datetime'].max()}\n"
            report += f"  潮位范围: {df['tide_level'].min():.1f} - {df['tide_level'].max():.1f} cm\n"
            report += f"  平均潮位: {df['tide_level'].mean():.1f} cm\n"
            report += f"  标准差: {df['tide_level'].std():.1f} cm\n"
            
            # 按月份统计
            monthly_stats = df.groupby(df['datetime'].dt.month)['tide_level'].agg(['mean', 'std', 'count'])
            report += f"  按月份统计:\n"
            for month, stats in monthly_stats.iterrows():
                report += f"    {month}月: 平均{stats['mean']:.1f}cm, 标准差{stats['std']:.1f}cm, {stats['count']}条记录\n"
            
            report += "\n"
            
        except Exception as e:
            report += f"{year}年: 处理失败 - {e}\n\n"
    
    # 保存报告
    report_file = os.path.join(output_dir, "数据处理汇总报告.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"汇总报告已保存到: {report_file}")
    print(report)

def main():
    """主函数"""
    print("开始处理三年潮位数据...")
    
    # 1. 处理原始潮位数据
    process_tide_data()
    
    # 2. 创建小时级数据
    create_hourly_tide_data()
    
    # 3. 生成汇总报告
    generate_summary_report()
    
    print("\n潮位数据处理完成！")
    print("处理后的数据保存在 '潮位数据处理' 文件夹中")

if __name__ == "__main__":
    main() 
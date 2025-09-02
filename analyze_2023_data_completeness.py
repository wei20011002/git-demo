import pandas as pd
import os
from datetime import datetime, timedelta

def analyze_2023_data_completeness():
    """分析2023年水位数据的完整性"""
    print("=== 2023年水位数据完整性分析 ===\n")
    
    # 检查每个月份的数据
    for month in range(1, 13):
        file_path = f'2023/2023-{month}-1.csv'
        
        if not os.path.exists(file_path):
            print(f"❌ {month}月: 文件不存在")
            continue
            
        try:
            # 尝试不同的编码
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
            except:
                df = pd.read_csv(file_path, encoding='gbk')
            
            # 检查数据基本信息
            print(f"📊 {month}月数据:")
            print(f"   - 总行数: {len(df)}")
            print(f"   - 列名: {df.columns.tolist()}")
            
            # 检查名称列的唯一值
            if '名称' in df.columns:
                unique_names = df['名称'].unique()
                print(f"   - 水位类型: {unique_names}")
                
                # 统计每种水位类型的数据量
                for name in unique_names:
                    count = len(df[df['名称'] == name])
                    print(f"     * {name}: {count}条记录")
            
            # 检查时间范围
            if '日期' in df.columns and '时间' in df.columns:
                # 合并日期和时间
                df['datetime'] = pd.to_datetime(df['日期'] + ' ' + df['时间'])
                
                if len(df) > 0:
                    start_time = df['datetime'].min()
                    end_time = df['datetime'].max()
                    print(f"   - 时间范围: {start_time} 到 {end_time}")
                    
                    # 计算应该有的小时数
                    expected_hours = (end_time - start_time).total_seconds() / 3600 + 1
                    actual_hours = len(df['datetime'].dt.floor('H').unique())
                    print(f"   - 应有小时数: {expected_hours:.0f}")
                    print(f"   - 实际小时数: {actual_hours}")
                    print(f"   - 数据完整度: {actual_hours/expected_hours*100:.1f}%")
            
            print()
            
        except Exception as e:
            print(f"❌ {month}月: 读取失败 - {e}")
            print()

def check_specific_month_data(month):
    """检查特定月份的数据详情"""
    print(f"=== 2023年{month}月数据详情 ===")
    
    file_path = f'2023/2023-{month}-1.csv'
    
    try:
        df = pd.read_csv(file_path, encoding='gbk')
        
        # 检查长江侧水位数据
        if '名称' in df.columns:
            yangtze_data = df[df['名称'] == '长江侧水位'].copy()
            
            if len(yangtze_data) > 0:
                print(f"长江侧水位数据: {len(yangtze_data)}条记录")
                
                # 合并日期和时间
                yangtze_data['datetime'] = pd.to_datetime(yangtze_data['日期'] + ' ' + yangtze_data['时间'])
                yangtze_data = yangtze_data.sort_values('datetime')
                
                # 检查时间间隔
                time_diffs = yangtze_data['datetime'].diff()
                print(f"时间间隔统计:")
                print(f"  最小间隔: {time_diffs.min()}")
                print(f"  最大间隔: {time_diffs.max()}")
                print(f"  平均间隔: {time_diffs.mean()}")
                
                # 检查是否有缺失的小时
                yangtze_data['hour'] = yangtze_data['datetime'].dt.floor('H')
                unique_hours = yangtze_data['hour'].unique()
                print(f"唯一小时数: {len(unique_hours)}")
                
                # 显示前10条记录
                print("\n前10条长江侧水位记录:")
                for i, row in yangtze_data.head(10).iterrows():
                    print(f"  {row['datetime']}: {row['数值']}")
                    
            else:
                print("❌ 没有找到长江侧水位数据")
                
    except Exception as e:
        print(f"❌ 读取失败: {e}")

if __name__ == "__main__":
    # 分析所有月份
    analyze_2023_data_completeness()
    
    # 重点检查8月和9月
    print("\n" + "="*50)
    check_specific_month_data(8)
    
    print("\n" + "="*50)
    check_specific_month_data(9) 
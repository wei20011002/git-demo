import pandas as pd
import os

def detailed_2023_check():
    """详细检查2023年数据分布"""
    print("=== 详细检查2023年数据分布 ===")
    
    # 加载2023年水位数据
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
            print(f"处理{month}月数据时出错: {e}")
            continue
    
    if monthly_data:
        combined_df = pd.concat(monthly_data, ignore_index=True)
        combined_df['datetime'] = pd.to_datetime(combined_df['日期'] + ' ' + combined_df['时间'])
        combined_df.set_index('datetime', inplace=True)
        
        combined_df = combined_df[['数值']].copy()
        combined_df.rename(columns={'数值': 'water_level'}, inplace=True)
        
        combined_df = combined_df.resample('H').mean()
        combined_df = combined_df.dropna()
        
        print(f"原始数据时间范围: {combined_df.index.min()} 到 {combined_df.index.max()}")
        print(f"原始数据点数: {len(combined_df)}")
        
        # 按月份检查数据分布
        for month in range(1, 13):
            month_data = combined_df[combined_df.index.month == month]
            if len(month_data) > 0:
                print(f"\n{month}月数据:")
                print(f"  时间范围: {month_data.index.min()} 到 {month_data.index.max()}")
                print(f"  数据点数: {len(month_data)}")
                print(f"  应有小时数: {pd.Timestamp(2023, month, 1).days_in_month * 24}")
            else:
                print(f"\n{month}月数据: 无数据")
        
        # 检查8月和9月的具体情况
        august_data = combined_df[combined_df.index.month == 8]
        september_data = combined_df[combined_df.index.month == 9]
        
        print(f"\n=== 8月详细分析 ===")
        if len(august_data) > 0:
            print(f"8月数据时间范围: {august_data.index.min()} 到 {august_data.index.max()}")
            print(f"8月数据点数: {len(august_data)}")
            print(f"8月应有小时数: 744 (31天 × 24小时)")
            print(f"8月数据完整度: {len(august_data)/744*100:.1f}%")
        
        print(f"\n=== 9月详细分析 ===")
        if len(september_data) > 0:
            print(f"9月数据时间范围: {september_data.index.min()} 到 {september_data.index.max()}")
            print(f"9月数据点数: {len(september_data)}")
            print(f"9月应有小时数: 720 (30天 × 24小时)")
            print(f"9月数据完整度: {len(september_data)/720*100:.1f}%")
        
        # 模拟排除缺失时段的效果
        print(f"\n=== 排除缺失时段效果 ===")
        # 排除8月8日之后到9月22日之前的数据
        mask = ~((combined_df.index >= '2023-08-09') & (combined_df.index < '2023-09-22'))
        filtered_df = combined_df[mask]
        
        print(f"排除前数据点数: {len(combined_df)}")
        print(f"排除后数据点数: {len(filtered_df)}")
        print(f"排除的数据点数: {len(combined_df) - len(filtered_df)}")
        
        if len(filtered_df) > 0:
            print(f"排除后时间范围: {filtered_df.index.min()} 到 {filtered_df.index.max()}")
        
        # 检查排除的数据
        excluded_data = combined_df[~mask]
        if len(excluded_data) > 0:
            print(f"被排除的数据时间范围: {excluded_data.index.min()} 到 {excluded_data.index.max()}")
            print(f"被排除的数据点数: {len(excluded_data)}")

if __name__ == "__main__":
    detailed_2023_check() 
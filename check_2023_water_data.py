import pandas as pd
import os

def check_2023_water_data():
    """检查2023年水位数据的实际时间范围"""
    print("=== 检查2023年水位数据时间范围 ===")
    
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
        
        # 检查8月和9月的数据
        august_data = combined_df[combined_df.index.month == 8]
        september_data = combined_df[combined_df.index.month == 9]
        
        print(f"\n8月数据:")
        print(f"  时间范围: {august_data.index.min()} 到 {august_data.index.max()}")
        print(f"  数据点数: {len(august_data)}")
        
        print(f"\n9月数据:")
        print(f"  时间范围: {september_data.index.min()} 到 {september_data.index.max()}")
        print(f"  数据点数: {len(september_data)}")
        
        # 排除缺失时段
        mask = ~((combined_df.index >= '2023-08-09') & (combined_df.index < '2023-09-22'))
        filtered_df = combined_df[mask]
        
        print(f"\n排除缺失时段后:")
        print(f"  时间范围: {filtered_df.index.min()} 到 {filtered_df.index.max()}")
        print(f"  数据点数: {len(filtered_df)}")
        
        # 检查排除的数据
        excluded_data = combined_df[~mask]
        print(f"\n被排除的数据:")
        print(f"  时间范围: {excluded_data.index.min()} 到 {excluded_data.index.max()}")
        print(f"  数据点数: {len(excluded_data)}")

if __name__ == "__main__":
    check_2023_water_data() 
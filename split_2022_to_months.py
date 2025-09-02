import os
import pandas as pd

# 输入文件
input_file = 'D:/船闸项目/2022.csv'
# 输出文件夹
output_dir = '2022'

# 创建输出文件夹
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 读取数据，自动识别编码
try:
    df = pd.read_csv(input_file, encoding='gbk')
except:
    try:
        df = pd.read_csv(input_file, encoding='utf-8')
    except:
        df = pd.read_csv(input_file, encoding='latin1')

# 确保有日期列
if '日期' not in df.columns:
    # 尝试自动识别
    for col in df.columns:
        if '日期' in col:
            df.rename(columns={col: '日期'}, inplace=True)
            break

# 按月份拆分
# 假设日期格式为 2022/1/1 或 2022-01-01
if not pd.api.types.is_datetime64_any_dtype(df['日期']):
    df['日期'] = pd.to_datetime(df['日期'], errors='coerce')

for month in range(1, 13):
    month_df = df[df['日期'].dt.month == month]
    if not month_df.empty:
        out_file = os.path.join(output_dir, f'2022-{month}-1.csv')
        month_df.to_csv(out_file, index=False, encoding='utf-8-sig')
        print(f'已保存: {out_file} ({len(month_df)} 条记录)')
    else:
        print(f'第{month}月无数据')
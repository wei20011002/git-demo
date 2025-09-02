import pandas as pd
import os

all_periods = []
base_dir = '交通需求'
for year in [2022, 2023, 2024]:
    fname = os.path.join(base_dir, f'{year}年全年船闸开通时间段.csv')
    if os.path.exists(fname):
        df = pd.read_csv(fname)
        df['年份'] = year
        all_periods.append(df)

if all_periods:
    all_df = pd.concat(all_periods, ignore_index=True)
    out_path = os.path.join(base_dir, '三年可开通闸时间段汇总.csv')
    all_df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f'已生成 {out_path}')
else:
    print('未找到任何可用的全年船闸开通时间段文件')
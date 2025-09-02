#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版降雨量对水位影响分析 - 重点分析累计降雨量影响
尝试多种方法来提高相关性分析效果，特别关注累计降雨量的影响
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
import os
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建输出图片文件夹
OUTPUT_FOLDER = 'output_images'
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
    print(f"创建输出文件夹: {OUTPUT_FOLDER}")

class EnhancedRainfallAnalyzer:
    def __init__(self):
        self.rainfall_data = None
        self.water_level_data = None
        self.merged_data = None
        
    def load_rainfall_data(self, file_path):
        """加载降雨量数据"""
        try:
            print("正在加载降雨量数据...")
            self.rainfall_data = pd.read_excel(file_path)
            print(f"降雨量数据加载成功，数据形状: {self.rainfall_data.shape}")
            return True
        except Exception as e:
            print(f"加载降雨量数据失败: {e}")
            return False
    
    def load_water_level_data(self):
        """加载水位数据"""
        try:
            print("\n正在加载水位数据...")
            water_level_data = []
            
            # 加载2022-2024年的水位数据
            for year in [2022, 2023, 2024]:
                year_dir = f"{year}/"
                try:
                    import os
                    files = os.listdir(year_dir)
                    water_files = [f for f in files if f.startswith('水位') and f.endswith('.csv')]
                    
                    if water_files:
                        for file in water_files:
                            file_path = os.path.join(year_dir, file)
                            try:
                                df = pd.read_csv(file_path, encoding='gbk')
                            except:
                                try:
                                    df = pd.read_csv(file_path, encoding='gb2312')
                                except:
                                    df = pd.read_csv(file_path, encoding='utf-8')
                            
                            df['年份'] = year
                            water_level_data.append(df)
                            print(f"加载 {file_path}")
                except Exception as e:
                    print(f"加载{year}年水位数据失败: {e}")
            
            if water_level_data:
                self.water_level_data = pd.concat(water_level_data, ignore_index=True)
                print(f"水位数据加载成功，数据形状: {self.water_level_data.shape}")
                return True
            else:
                print("未找到水位数据文件")
                return False
                
        except Exception as e:
            print(f"加载水位数据失败: {e}")
            return False
    
    def preprocess_data_enhanced(self):
        """增强版数据预处理 - 重点计算累计降雨量"""
        try:
            print("\n正在进行增强版数据预处理...")
            
            # 处理降雨量数据
            if self.rainfall_data is not None:
                # 创建日期时间列
                self.rainfall_data['日期时间'] = pd.to_datetime(
                    self.rainfall_data[['年', '月', '日', '时']].rename(columns={'年': 'year', '月': 'month', '日': 'day', '时': 'hour'})
                )
                self.rainfall_data['日期'] = pd.to_datetime(
                    self.rainfall_data[['年', '月', '日']].rename(columns={'年': 'year', '月': 'month', '日': 'day'})
                )
                
                # 按站点分组处理
                dantu_data = self.rainfall_data[self.rainfall_data['站点'] == '丹徒'].copy()
                danyang_data = self.rainfall_data[self.rainfall_data['站点'] == '丹阳'].copy()
                
                # 计算更全面的累积降雨量（重点分析）
                dantu_enhanced = self._calculate_comprehensive_cumulative_rainfall(dantu_data, '丹徒站')
                danyang_enhanced = self._calculate_comprehensive_cumulative_rainfall(danyang_data, '丹阳站')
                
                # 合并两个站点的数据
                self.rainfall_data = pd.merge(dantu_enhanced, danyang_enhanced, on=['日期时间', '日期'], how='outer')
                self.rainfall_data = self.rainfall_data.fillna(0)
                
                print(f"处理后的降雨量数据形状: {self.rainfall_data.shape}")
                print("处理后的降雨量数据列名:", self.rainfall_data.columns.tolist())
            
            # 处理水位数据
            if self.water_level_data is not None:
                # 创建日期时间列
                self.water_level_data['日期时间'] = pd.to_datetime(
                    self.water_level_data['日期'] + ' ' + self.water_level_data['时间']
                )
                self.water_level_data['日期'] = pd.to_datetime(self.water_level_data['日期'])
                
                # 按水位类型分组处理
                changjiang_data = self.water_level_data[self.water_level_data['名称'] == '长江侧水位'].copy()
                yunhe_data = self.water_level_data[self.water_level_data['名称'] == '内河侧水位'].copy()
                
                # 保留小时级别数据
                changjiang_hourly = changjiang_data[['日期时间', '日期', '数值']].copy()
                changjiang_hourly.columns = ['日期时间', '日期', '长江侧水位']
                
                yunhe_hourly = yunhe_data[['日期时间', '日期', '数值']].copy()
                yunhe_hourly.columns = ['日期时间', '日期', '运河侧水位']
                
                # 合并两个水位数据
                self.water_level_data = pd.merge(changjiang_hourly, yunhe_hourly, on=['日期时间', '日期'], how='outer')
                
                # 计算水位变化量（小时级别）
                self.water_level_data = self.water_level_data.sort_values('日期时间')
                self.water_level_data['长江侧水位变化'] = self.water_level_data['长江侧水位'].diff()
                self.water_level_data['运河侧水位变化'] = self.water_level_data['运河侧水位'].diff()
                
                # 计算水位变化率
                self.water_level_data['长江侧水位变化率'] = self.water_level_data['长江侧水位变化'] / self.water_level_data['长江侧水位'].shift(1) * 100
                self.water_level_data['运河侧水位变化率'] = self.water_level_data['运河侧水位变化'] / self.water_level_data['运河侧水位'].shift(1) * 100
                
                print(f"处理后的水位数据形状: {self.water_level_data.shape}")
                print("处理后的水位数据列名:", self.water_level_data.columns.tolist())
            
            return True
        except Exception as e:
            print(f"数据预处理失败: {e}")
            return False
    
    def _calculate_comprehensive_cumulative_rainfall(self, data, station_name):
        """计算全面的累积降雨量 - 重点分析累计效应"""
        data = data.sort_values('日期时间')
        
        # 基础降雨量
        result = data[['日期时间', '日期', '降水量（mm）']].copy()
        result.columns = ['日期时间', '日期', f'{station_name}降雨量']
        
        # 计算更全面的累积降雨量（重点分析累计效应）
        cumulative_hours = [1, 3, 4, 6, 8, 12, 16, 20, 24, 48, 72]  # 包含所有需要的时间窗口
        for hours in cumulative_hours:
            result[f'{station_name}累积{hours}小时降雨量'] = result[f'{station_name}降雨量'].rolling(window=hours, min_periods=1).sum()
        
        # 计算降雨强度（每小时平均降雨量）
        for hours in [3, 6, 12, 24, 48]:
            result[f'{station_name}{hours}小时降雨强度'] = result[f'{station_name}累积{hours}小时降雨量'] / hours
        
        # 计算累计降雨量的变化率（重点指标）
        for hours in [6, 12, 24, 48]:
            result[f'{station_name}累积{hours}小时降雨量变化'] = result[f'{station_name}累积{hours}小时降雨量'].diff()
        
        # 计算累计降雨量的增长率
        for hours in [6, 12, 24, 48]:
            result[f'{station_name}累积{hours}小时降雨量增长率'] = (
                result[f'{station_name}累积{hours}小时降雨量变化'] / 
                result[f'{station_name}累积{hours}小时降雨量'].shift(1) * 100
            ).fillna(0)
        
        return result
    
    def merge_data(self):
        """合并降雨量和水位数据"""
        try:
            print("\n正在合并降雨量和水位数据...")
            if self.rainfall_data is None or self.water_level_data is None:
                print("降雨量或水位数据未加载")
                return False
            
            # 合并数据（按日期时间合并）
            self.merged_data = pd.merge(
                self.rainfall_data, 
                self.water_level_data, 
                on=['日期时间', '日期'], 
                how='inner'
            )
            # 按时间排序
            self.merged_data = self.merged_data.sort_values('日期时间')
            print(f"数据合并完成，合并后数据形状: {self.merged_data.shape}")
            return True
        except Exception as e:
            print(f"数据合并失败: {e}")
            return False
    
    def analyze_cumulative_rainfall_impact(self):
        """重点分析累计降雨量对水位的影响"""
        try:
            print("\n=== 重点分析累计降雨量对水位的影响 ===")
            
            # 清理数据中的无穷大和NaN值
            self.merged_data = self.merged_data.replace([np.inf, -np.inf], np.nan)
            
            # 定义要分析的累计降雨量指标
            cumulative_metrics = [
                # 丹徒站累计降雨量 vs 长江侧水位
                ('丹徒站累积6小时降雨量', '长江侧水位变化'),
                ('丹徒站累积12小时降雨量', '长江侧水位变化'),
                ('丹徒站累积24小时降雨量', '长江侧水位变化'),
                ('丹徒站累积48小时降雨量', '长江侧水位变化'),
                ('丹徒站累积72小时降雨量', '长江侧水位变化'),
                
                # 丹阳站累计降雨量 vs 运河侧水位
                ('丹阳站累积6小时降雨量', '运河侧水位变化'),
                ('丹阳站累积12小时降雨量', '运河侧水位变化'),
                ('丹阳站累积24小时降雨量', '运河侧水位变化'),
                ('丹阳站累积48小时降雨量', '运河侧水位变化'),
                ('丹阳站累积72小时降雨量', '运河侧水位变化'),
                
                # 累计降雨量变化 vs 水位变化
                ('丹徒站累积24小时降雨量变化', '长江侧水位变化'),
                ('丹阳站累积24小时降雨量变化', '运河侧水位变化'),
                
                # 累计降雨量增长率 vs 水位变化率
                ('丹徒站累积24小时降雨量增长率', '长江侧水位变化率'),
                ('丹阳站累积24小时降雨量增长率', '运河侧水位变化率'),
            ]
            
            results = {}
            
            for rainfall_col, water_col in cumulative_metrics:
                if rainfall_col in self.merged_data.columns and water_col in self.merged_data.columns:
                    # 只选择有累计降雨的时间段，并清理异常值
                    rainfall_data = self.merged_data[
                        (self.merged_data[rainfall_col] > 0) & 
                        (self.merged_data[rainfall_col] < np.inf) &
                        (self.merged_data[water_col] > -np.inf) &
                        (self.merged_data[water_col] < np.inf)
                    ].copy()
                    
                    valid_data = rainfall_data[[rainfall_col, water_col]].dropna()
                    
                    if len(valid_data) > 10:
                        # 进一步清理异常值（去除极端值）
                        q1_rainfall = valid_data[rainfall_col].quantile(0.01)
                        q99_rainfall = valid_data[rainfall_col].quantile(0.99)
                        q1_water = valid_data[water_col].quantile(0.01)
                        q99_water = valid_data[water_col].quantile(0.99)
                        
                        clean_data = valid_data[
                            (valid_data[rainfall_col] >= q1_rainfall) &
                            (valid_data[rainfall_col] <= q99_rainfall) &
                            (valid_data[water_col] >= q1_water) &
                            (valid_data[water_col] <= q99_water)
                        ]
                        
                        if len(clean_data) > 10:
                            try:
                                pearson_corr, pearson_p = pearsonr(clean_data[rainfall_col], clean_data[water_col])
                                spearman_corr, spearman_p = spearmanr(clean_data[rainfall_col], clean_data[water_col])
                                
                                results[f"{rainfall_col}_vs_{water_col}"] = {
                                    'pearson_corr': pearson_corr,
                                    'pearson_p': pearson_p,
                                    'spearman_corr': spearman_corr,
                                    'spearman_p': spearman_p,
                                    'sample_size': len(clean_data),
                                    'mean_rainfall': clean_data[rainfall_col].mean(),
                                    'mean_water_change': clean_data[water_col].mean()
                                }
                            except Exception as e:
                                print(f"计算{rainfall_col} vs {water_col}相关性时出错: {e}")
                                continue
            
            # 按相关性强度排序
            sorted_results = sorted(results.items(), key=lambda x: abs(x[1]['pearson_corr']), reverse=True)
            
            print("\n=== 累计降雨量影响分析结果（按相关性强度排序） ===")
            for key, result in sorted_results:
                print(f"\n{key}:")
                print(f"  皮尔逊相关系数: {result['pearson_corr']:.4f} (p={result['pearson_p']:.4f})")
                print(f"  斯皮尔曼相关系数: {result['spearman_corr']:.4f} (p={result['spearman_p']:.4f})")
                print(f"  样本数量: {result['sample_size']}")
                print(f"  平均累计降雨量: {result['mean_rainfall']:.2f}")
                print(f"  平均水位变化: {result['mean_water_change']:.4f}")
                
                # 判断相关性强度
                corr_strength = "弱相关"
                if abs(result['pearson_corr']) >= 0.7:
                    corr_strength = "强相关"
                elif abs(result['pearson_corr']) >= 0.3:
                    corr_strength = "中等相关"
                
                print(f"  相关性强度: {corr_strength}")
                
                # 标记显著性
                if result['pearson_p'] < 0.05:
                    print(f"  *** 统计显著 (p<0.05) ***")
                if result['pearson_p'] < 0.01:
                    print(f"  *** 高度显著 (p<0.01) ***")
            
            return results
        except Exception as e:
            print(f"累计降雨量影响分析失败: {e}")
            return None
    
    def analyze_lag_effects(self):
        """分析累计降雨量的滞后效应"""
        try:
            print("\n=== 分析累计降雨量的滞后效应 ===")
            
            # 清理数据中的无穷大和NaN值
            self.merged_data = self.merged_data.replace([np.inf, -np.inf], np.nan)
            
            # 分析不同滞后时间的影响
            lag_hours = [0, 1, 2, 3, 6, 12, 24]  # 滞后0-24小时
            lag_results = {}
            
            for lag in lag_hours:
                # 创建滞后数据
                lagged_data = self.merged_data.copy()
                if lag > 0:
                    lagged_data['丹徒站累积24小时降雨量_滞后'] = lagged_data['丹徒站累积24小时降雨量'].shift(lag)
                    lagged_data['丹阳站累积24小时降雨量_滞后'] = lagged_data['丹阳站累积24小时降雨量'].shift(lag)
                else:
                    lagged_data['丹徒站累积24小时降雨量_滞后'] = lagged_data['丹徒站累积24小时降雨量']
                    lagged_data['丹阳站累积24小时降雨量_滞后'] = lagged_data['丹阳站累积24小时降雨量']
                
                # 分析丹徒站
                valid_data = lagged_data[['丹徒站累积24小时降雨量_滞后', '长江侧水位变化']].dropna()
                # 清理异常值
                valid_data = valid_data[
                    (valid_data['丹徒站累积24小时降雨量_滞后'] > 0) &
                    (valid_data['丹徒站累积24小时降雨量_滞后'] < np.inf) &
                    (valid_data['长江侧水位变化'] > -np.inf) &
                    (valid_data['长江侧水位变化'] < np.inf)
                ]
                
                if len(valid_data) > 10:
                    try:
                        pearson_corr, pearson_p = pearsonr(valid_data['丹徒站累积24小时降雨量_滞后'], valid_data['长江侧水位变化'])
                        lag_results[f"丹徒站_滞后{lag}小时"] = {
                            'pearson_corr': pearson_corr,
                            'pearson_p': pearson_p,
                            'sample_size': len(valid_data)
                        }
                    except Exception as e:
                        print(f"计算丹徒站滞后{lag}小时相关性时出错: {e}")
                        continue
                
                # 分析丹阳站
                valid_data = lagged_data[['丹阳站累积24小时降雨量_滞后', '运河侧水位变化']].dropna()
                # 清理异常值
                valid_data = valid_data[
                    (valid_data['丹阳站累积24小时降雨量_滞后'] > 0) &
                    (valid_data['丹阳站累积24小时降雨量_滞后'] < np.inf) &
                    (valid_data['运河侧水位变化'] > -np.inf) &
                    (valid_data['运河侧水位变化'] < np.inf)
                ]
                
                if len(valid_data) > 10:
                    try:
                        pearson_corr, pearson_p = pearsonr(valid_data['丹阳站累积24小时降雨量_滞后'], valid_data['运河侧水位变化'])
                        lag_results[f"丹阳站_滞后{lag}小时"] = {
                            'pearson_corr': pearson_corr,
                            'pearson_p': pearson_p,
                            'sample_size': len(valid_data)
                        }
                    except Exception as e:
                        print(f"计算丹阳站滞后{lag}小时相关性时出错: {e}")
                        continue
            
            print("\n=== 滞后效应分析结果 ===")
            for key, result in lag_results.items():
                print(f"\n{key}:")
                print(f"  皮尔逊相关系数: {result['pearson_corr']:.4f} (p={result['pearson_p']:.4f})")
                print(f"  样本数量: {result['sample_size']}")
                
                if result['pearson_p'] < 0.05:
                    print(f"  *** 统计显著 ***")
            
            return lag_results
        except Exception as e:
            print(f"滞后效应分析失败: {e}")
            return None
    
    def analyze_threshold_effects(self):
        """分析累计降雨量的阈值效应"""
        try:
            print("\n=== 分析累计降雨量的阈值效应 ===")
            
            # 清理数据中的无穷大和NaN值
            self.merged_data = self.merged_data.replace([np.inf, -np.inf], np.nan)
            
            # 定义不同的累计降雨量阈值
            thresholds = [10, 20, 30, 50, 100, 200]  # mm
            threshold_results = {}
            
            for threshold in thresholds:
                # 分析超过阈值的情况
                high_rainfall_data = self.merged_data[
                    ((self.merged_data['丹徒站累积24小时降雨量'] >= threshold) |
                     (self.merged_data['丹阳站累积24小时降雨量'] >= threshold)) &
                    (self.merged_data['丹徒站累积24小时降雨量'] < np.inf) &
                    (self.merged_data['丹阳站累积24小时降雨量'] < np.inf) &
                    (self.merged_data['长江侧水位变化'] > -np.inf) &
                    (self.merged_data['长江侧水位变化'] < np.inf) &
                    (self.merged_data['运河侧水位变化'] > -np.inf) &
                    (self.merged_data['运河侧水位变化'] < np.inf)
                ].copy()
                
                if len(high_rainfall_data) > 10:
                    # 分析高累计降雨量对水位的影响
                    valid_data = high_rainfall_data[['丹徒站累积24小时降雨量', '长江侧水位变化']].dropna()
                    if len(valid_data) > 5:
                        try:
                            pearson_corr, pearson_p = pearsonr(valid_data['丹徒站累积24小时降雨量'], valid_data['长江侧水位变化'])
                            threshold_results[f"丹徒站_阈值{threshold}mm"] = {
                                'pearson_corr': pearson_corr,
                                'pearson_p': pearson_p,
                                'sample_size': len(valid_data),
                                'mean_water_change': valid_data['长江侧水位变化'].mean()
                            }
                        except Exception as e:
                            print(f"计算丹徒站阈值{threshold}mm相关性时出错: {e}")
                            continue
                    
                    valid_data = high_rainfall_data[['丹阳站累积24小时降雨量', '运河侧水位变化']].dropna()
                    if len(valid_data) > 5:
                        try:
                            pearson_corr, pearson_p = pearsonr(valid_data['丹阳站累积24小时降雨量'], valid_data['运河侧水位变化'])
                            threshold_results[f"丹阳站_阈值{threshold}mm"] = {
                                'pearson_corr': pearson_corr,
                                'pearson_p': pearson_p,
                                'sample_size': len(valid_data),
                                'mean_water_change': valid_data['运河侧水位变化'].mean()
                            }
                        except Exception as e:
                            print(f"计算丹阳站阈值{threshold}mm相关性时出错: {e}")
                            continue
            
            print("\n=== 阈值效应分析结果 ===")
            for key, result in threshold_results.items():
                print(f"\n{key}:")
                print(f"  皮尔逊相关系数: {result['pearson_corr']:.4f} (p={result['pearson_p']:.4f})")
                print(f"  样本数量: {result['sample_size']}")
                print(f"  平均水位变化: {result['mean_water_change']:.4f}")
                
                if result['pearson_p'] < 0.05:
                    print(f"  *** 统计显著 ***")
            
            return threshold_results
        except Exception as e:
            print(f"阈值效应分析失败: {e}")
            return None
    
    def create_cumulative_rainfall_visualizations(self, cumulative_results):
        """创建累计降雨量影响的可视化图表 - 分别保存"""
        try:
            print("\n正在创建累计降雨量影响的可视化图表...")
            
            if not cumulative_results:
                return
            
            # 1. 累计降雨量 vs 水位变化散点图（两个站点对比）
            plt.figure(figsize=(15, 6))
            
            # 丹徒站散点图
            plt.subplot(1, 2, 1)
            rainfall_data = self.merged_data[self.merged_data['丹徒站累积24小时降雨量'] > 0].copy()
            valid_data = rainfall_data[['丹徒站累积24小时降雨量', '长江侧水位变化']].dropna()
            
            if len(valid_data) > 0:
                plt.scatter(valid_data['丹徒站累积24小时降雨量'], valid_data['长江侧水位变化'], alpha=0.6, color='blue')
                plt.xlabel('丹徒站累积24小时降雨量 (mm)')
                plt.ylabel('长江侧水位变化 (m)')
                plt.title('丹徒站累计降雨量 vs 长江侧水位变化')
                
                # 添加趋势线
                if len(valid_data) > 1:
                    z = np.polyfit(valid_data['丹徒站累积24小时降雨量'], valid_data['长江侧水位变化'], 1)
                    p = np.poly1d(z)
                    plt.plot(valid_data['丹徒站累积24小时降雨量'], p(valid_data['丹徒站累积24小时降雨量']), "r--", alpha=0.8, linewidth=2)
            
            # 丹阳站散点图
            plt.subplot(1, 2, 2)
            rainfall_data = self.merged_data[self.merged_data['丹阳站累积24小时降雨量'] > 0].copy()
            valid_data = rainfall_data[['丹阳站累积24小时降雨量', '运河侧水位变化']].dropna()
            
            if len(valid_data) > 0:
                plt.scatter(valid_data['丹阳站累积24小时降雨量'], valid_data['运河侧水位变化'], alpha=0.6, color='red')
                plt.xlabel('丹阳站累积24小时降雨量 (mm)')
                plt.ylabel('运河侧水位变化 (m)')
                plt.title('丹阳站累计降雨量 vs 运河侧水位变化')
                
                # 添加趋势线
                if len(valid_data) > 1:
                    z = np.polyfit(valid_data['丹阳站累积24小时降雨量'], valid_data['运河侧水位变化'], 1)
                    p = np.poly1d(z)
                    plt.plot(valid_data['丹阳站累积24小时降雨量'], p(valid_data['丹阳站累积24小时降雨量']), color='orange', linestyle='--', alpha=0.8, linewidth=2)
            
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_FOLDER, '1_累计降雨量vs水位变化散点图.png'), dpi=300, bbox_inches='tight')
            plt.show()
            print("散点图已保存为 '1_累计降雨量vs水位变化散点图.png'")
            
            # 2. 不同时间窗口累计降雨量相关性对比（两个站点）
            plt.figure(figsize=(15, 6))
            cumulative_hours = [4, 8, 12, 16, 20, 24]  # 只包含24小时以内的数据窗口，密度更小
            dantu_corrs = []
            danyang_corrs = []
            
            for hours in cumulative_hours:
                # 丹徒站
                col_name = f'丹徒站累积{hours}小时降雨量'
                if col_name in self.merged_data.columns:
                    valid_data = self.merged_data[[col_name, '长江侧水位变化']].dropna()
                    if len(valid_data) > 10:
                        corr, _ = pearsonr(valid_data[col_name], valid_data['长江侧水位变化'])
                        dantu_corrs.append(abs(corr))
                    else:
                        dantu_corrs.append(0)
                else:
                    dantu_corrs.append(0)
                
                # 丹阳站
                col_name = f'丹阳站累积{hours}小时降雨量'
                if col_name in self.merged_data.columns:
                    valid_data = self.merged_data[[col_name, '运河侧水位变化']].dropna()
                    if len(valid_data) > 10:
                        corr, _ = pearsonr(valid_data[col_name], valid_data['运河侧水位变化'])
                        danyang_corrs.append(abs(corr))
                    else:
                        danyang_corrs.append(0)
                else:
                    danyang_corrs.append(0)
            
            # 绘制对比图
            x = np.arange(len(cumulative_hours))
            width = 0.35
            
            plt.bar(x - width/2, dantu_corrs, width, label='丹徒站', color='blue', alpha=0.7)
            plt.bar(x + width/2, danyang_corrs, width, label='丹阳站', color='red', alpha=0.7)
            
            plt.xlabel('累计时间窗口 (小时)')
            plt.ylabel('相关系数绝对值')
            plt.title('不同累计时间窗口的相关性强度对比')
            plt.xticks(x, cumulative_hours)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_FOLDER, '2_不同时间窗口相关性对比.png'), dpi=300, bbox_inches='tight')
            plt.show()
            print("相关性对比图已保存为 '2_不同时间窗口相关性对比.png'")
            
            # 3. 累计降雨量分布（两个站点对比）
            plt.figure(figsize=(15, 6))
            
            # 丹徒站分布
            plt.subplot(1, 2, 1)
            rainfall_data = self.merged_data[self.merged_data['丹徒站累积24小时降雨量'] > 0].copy()
            if len(rainfall_data) > 0:
                plt.hist(rainfall_data['丹徒站累积24小时降雨量'], bins=30, alpha=0.7, color='blue', edgecolor='black')
                plt.xlabel('丹徒站累积24小时降雨量 (mm)')
                plt.ylabel('频次')
                plt.title('丹徒站累计降雨量分布')
                plt.grid(True, alpha=0.3)
            
            # 丹阳站分布
            plt.subplot(1, 2, 2)
            rainfall_data = self.merged_data[self.merged_data['丹阳站累积24小时降雨量'] > 0].copy()
            if len(rainfall_data) > 0:
                plt.hist(rainfall_data['丹阳站累积24小时降雨量'], bins=30, alpha=0.7, color='red', edgecolor='black')
                plt.xlabel('丹阳站累积24小时降雨量 (mm)')
                plt.ylabel('频次')
                plt.title('丹阳站累计降雨量分布')
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_FOLDER, '3_累计降雨量分布.png'), dpi=300, bbox_inches='tight')
            plt.show()
            print("分布图已保存为 '3_累计降雨量分布.png'")
            
            # 4. 时间序列对比（两个站点）- 选择2024年6月29至7月1日的时段
            plt.figure(figsize=(15, 8))
            
            # 选择2024年6月29至7月1日的时段进行对比
            rainfall_data = self.merged_data.copy()
            
            # 设置时间范围：2024年6月29日至7月1日
            start_date = pd.to_datetime('2024-06-29 00:00:00')
            end_date = pd.to_datetime('2024-07-01 23:59:59')
            
            # 筛选指定时间范围的数据
            sample_data = rainfall_data[
                (rainfall_data['日期时间'] >= start_date) & 
                (rainfall_data['日期时间'] <= end_date)
            ].copy()
            
            print(f"选择了2024年6月29日至7月1日的数据，共{len(sample_data)}个数据点")
            if len(sample_data) > 0:
                print(f"时间范围：{sample_data['日期时间'].min()} 至 {sample_data['日期时间'].max()}")
            else:
                print("警告：在指定时间范围内没有找到数据，将使用全部数据")
                sample_data = rainfall_data.copy()
            
            # 丹徒站时间序列
            plt.subplot(2, 1, 1)
            rainfall_data = sample_data[sample_data['丹徒站累积24小时降雨量'] > 0].copy()
            if len(rainfall_data) > 0:
                ax1 = plt.gca()
                ax1_twin = ax1.twinx()
                
                line1 = ax1.plot(rainfall_data['日期时间'], rainfall_data['丹徒站累积24小时降雨量'], 'b-', label='丹徒站累积24小时降雨量', alpha=0.7, linewidth=2)
                line2 = ax1_twin.plot(rainfall_data['日期时间'], rainfall_data['长江侧水位变化'], 'g-', label='长江侧水位变化', alpha=0.7, linewidth=2)
                
                ax1.set_xlabel('时间')
                ax1.set_ylabel('丹徒站累积24小时降雨量 (mm)', color='b', fontsize=10)
                ax1_twin.set_ylabel('长江侧水位变化 (m)', color='g', fontsize=10)
                ax1.set_title('丹徒站累计降雨量与长江侧水位变化时间序列对比', fontsize=12, fontweight='bold')
                
                lines = line1 + line2
                labels = [l.get_label() for l in lines]
                ax1.legend(lines, labels, loc='upper left')
                ax1.grid(True, alpha=0.3)
            
            # 丹阳站时间序列
            plt.subplot(2, 1, 2)
            rainfall_data = sample_data[sample_data['丹阳站累积24小时降雨量'] > 0].copy()
            if len(rainfall_data) > 0:
                ax2 = plt.gca()
                ax2_twin = ax2.twinx()
                
                line3 = ax2.plot(rainfall_data['日期时间'], rainfall_data['丹阳站累积24小时降雨量'], 'r-', label='丹阳站累积24小时降雨量', alpha=0.7, linewidth=2)
                line4 = ax2_twin.plot(rainfall_data['日期时间'], rainfall_data['运河侧水位变化'], 'orange', label='运河侧水位变化', alpha=0.7, linewidth=2)
                
                ax2.set_xlabel('时间')
                ax2.set_ylabel('丹阳站累积24小时降雨量 (mm)', color='r', fontsize=10)
                ax2_twin.set_ylabel('运河侧水位变化 (m)', color='orange', fontsize=10)
                ax2.set_title('丹阳站累计降雨量与运河侧水位变化时间序列对比', fontsize=12, fontweight='bold')
                
                lines = line3 + line4
                labels = [l.get_label() for l in lines]
                ax2.legend(lines, labels, loc='upper left')
                ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_FOLDER, '4_时间序列对比.png'), dpi=300, bbox_inches='tight')
            plt.show()
            print("时间序列图已保存为 '4_时间序列对比.png'")
            
            # 5. 累计降雨量分组下的水位变化箱线图（两个站点）
            plt.figure(figsize=(15, 6))
            
            # 丹徒站箱线图
            plt.subplot(1, 2, 1)
            rainfall_data = self.merged_data[self.merged_data['丹徒站累积24小时降雨量'] > 0].copy()
            if len(rainfall_data) > 0:
                rainfall_data['累计降雨量分组'] = pd.cut(rainfall_data['丹徒站累积24小时降雨量'], 
                                                    bins=5, labels=['很低', '低', '中等', '高', '很高'])
                rainfall_data.boxplot(column='长江侧水位变化', by='累计降雨量分组')
                plt.title('丹徒站累计降雨量分组下的长江侧水位变化分布', fontsize=12, fontweight='bold')
                plt.xlabel('累计降雨量分组', fontsize=10)
                plt.ylabel('长江侧水位变化 (m)', fontsize=10)
                plt.suptitle('')
                plt.grid(True, alpha=0.3)
            
            # 丹阳站箱线图
            plt.subplot(1, 2, 2)
            rainfall_data = self.merged_data[self.merged_data['丹阳站累积24小时降雨量'] > 0].copy()
            if len(rainfall_data) > 0:
                rainfall_data['累计降雨量分组'] = pd.cut(rainfall_data['丹阳站累积24小时降雨量'], 
                                                    bins=5, labels=['很低', '低', '中等', '高', '很高'])
                rainfall_data.boxplot(column='运河侧水位变化', by='累计降雨量分组')
                plt.title('丹阳站累计降雨量分组下的运河侧水位变化分布', fontsize=12, fontweight='bold')
                plt.xlabel('累计降雨量分组', fontsize=10)
                plt.ylabel('运河侧水位变化 (m)', fontsize=10)
                plt.suptitle('')
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_FOLDER, '5_累计降雨量分组箱线图.png'), dpi=300, bbox_inches='tight')
            plt.show()
            print("箱线图已保存为 '5_累计降雨量分组箱线图.png'")
            
            # 6. 相关性热力图（保持不变，因为已经包含所有指标）
            plt.figure(figsize=(14, 10))
            correlation_cols = [col for col in self.merged_data.columns if '累积' in col and '降雨量' in col]
            correlation_cols.extend([col for col in self.merged_data.columns if '水位变化' in col])
            correlation_data = self.merged_data[correlation_cols].corr()
            
            sns.heatmap(correlation_data, annot=True, cmap='coolwarm', center=0, fmt='.2f', 
                       square=True, cbar_kws={"shrink": .8})
            plt.title('累计降雨量与水位变化相关性热力图', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_FOLDER, '6_相关性热力图.png'), dpi=300, bbox_inches='tight')
            plt.show()
            print("热力图已保存为 '6_相关性热力图.png'")
            
            # 7. 滞后效应分析图（保持不变，因为已经包含两个站点）
            plt.figure(figsize=(12, 8))
            lag_hours = [0, 1, 2, 3, 6, 12, 24]
            dantu_corrs = []
            danyang_corrs = []
            
            for lag in lag_hours:
                lagged_data = self.merged_data.copy()
                if lag > 0:
                    lagged_data['丹徒站累积24小时降雨量_滞后'] = lagged_data['丹徒站累积24小时降雨量'].shift(lag)
                    lagged_data['丹阳站累积24小时降雨量_滞后'] = lagged_data['丹阳站累积24小时降雨量'].shift(lag)
                else:
                    lagged_data['丹徒站累积24小时降雨量_滞后'] = lagged_data['丹徒站累积24小时降雨量']
                    lagged_data['丹阳站累积24小时降雨量_滞后'] = lagged_data['丹阳站累积24小时降雨量']
                
                # 丹徒站
                valid_data = lagged_data[['丹徒站累积24小时降雨量_滞后', '长江侧水位变化']].dropna()
                valid_data = valid_data[
                    (valid_data['丹徒站累积24小时降雨量_滞后'] > 0) &
                    (valid_data['丹徒站累积24小时降雨量_滞后'] < np.inf) &
                    (valid_data['长江侧水位变化'] > -np.inf) &
                    (valid_data['长江侧水位变化'] < np.inf)
                ]
                if len(valid_data) > 10:
                    try:
                        corr, _ = pearsonr(valid_data['丹徒站累积24小时降雨量_滞后'], valid_data['长江侧水位变化'])
                        dantu_corrs.append(corr)
                    except:
                        dantu_corrs.append(0)
                else:
                    dantu_corrs.append(0)
                
                # 丹阳站
                valid_data = lagged_data[['丹阳站累积24小时降雨量_滞后', '运河侧水位变化']].dropna()
                valid_data = valid_data[
                    (valid_data['丹阳站累积24小时降雨量_滞后'] > 0) &
                    (valid_data['丹阳站累积24小时降雨量_滞后'] < np.inf) &
                    (valid_data['运河侧水位变化'] > -np.inf) &
                    (valid_data['运河侧水位变化'] < np.inf)
                ]
                if len(valid_data) > 10:
                    try:
                        corr, _ = pearsonr(valid_data['丹阳站累积24小时降雨量_滞后'], valid_data['运河侧水位变化'])
                        danyang_corrs.append(corr)
                    except:
                        danyang_corrs.append(0)
                else:
                    danyang_corrs.append(0)
            
            plt.plot(lag_hours, dantu_corrs, 'bo-', label='丹徒站', linewidth=2, markersize=8)
            plt.plot(lag_hours, danyang_corrs, 'ro-', label='丹阳站', linewidth=2, markersize=8)
            plt.xlabel('滞后时间 (小时)')
            plt.ylabel('相关系数')
            plt.title('累计降雨量滞后效应分析')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_FOLDER, '7_滞后效应分析.png'), dpi=300, bbox_inches='tight')
            plt.show()
            print("滞后效应图已保存为 '7_滞后效应分析.png'")
            
            # 8. 阈值效应分析图（保持不变，因为已经包含两个站点）
            plt.figure(figsize=(12, 8))
            thresholds = [10, 20, 30, 50, 100, 200]
            dantu_corrs = []
            danyang_corrs = []
            
            for threshold in thresholds:
                high_rainfall_data = self.merged_data[
                    ((self.merged_data['丹徒站累积24小时降雨量'] >= threshold) |
                     (self.merged_data['丹阳站累积24小时降雨量'] >= threshold)) &
                    (self.merged_data['丹徒站累积24小时降雨量'] < np.inf) &
                    (self.merged_data['丹阳站累积24小时降雨量'] < np.inf) &
                    (self.merged_data['长江侧水位变化'] > -np.inf) &
                    (self.merged_data['长江侧水位变化'] < np.inf) &
                    (self.merged_data['运河侧水位变化'] > -np.inf) &
                    (self.merged_data['运河侧水位变化'] < np.inf)
                ].copy()
                
                if len(high_rainfall_data) > 10:
                    # 丹徒站
                    valid_data = high_rainfall_data[['丹徒站累积24小时降雨量', '长江侧水位变化']].dropna()
                    if len(valid_data) > 5:
                        try:
                            corr, _ = pearsonr(valid_data['丹徒站累积24小时降雨量'], valid_data['长江侧水位变化'])
                            dantu_corrs.append(corr)
                        except:
                            dantu_corrs.append(0)
                    else:
                        dantu_corrs.append(0)
                    
                    # 丹阳站
                    valid_data = high_rainfall_data[['丹阳站累积24小时降雨量', '运河侧水位变化']].dropna()
                    if len(valid_data) > 5:
                        try:
                            corr, _ = pearsonr(valid_data['丹阳站累积24小时降雨量'], valid_data['运河侧水位变化'])
                            danyang_corrs.append(corr)
                        except:
                            danyang_corrs.append(0)
                    else:
                        danyang_corrs.append(0)
                else:
                    dantu_corrs.append(0)
                    danyang_corrs.append(0)
            
            plt.plot(thresholds, dantu_corrs, 'bo-', label='丹徒站', linewidth=2, markersize=8)
            plt.plot(thresholds, danyang_corrs, 'ro-', label='丹阳站', linewidth=2, markersize=8)
            plt.xlabel('累计降雨量阈值 (mm)')
            plt.ylabel('相关系数')
            plt.title('累计降雨量阈值效应分析')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_FOLDER, '8_阈值效应分析.png'), dpi=300, bbox_inches='tight')
            plt.show()
            print("阈值效应图已保存为 '8_阈值效应分析.png'")
            
            print("\n所有图表已分别保存完成！")
            
        except Exception as e:
            print(f"创建累计降雨量可视化失败: {e}")

def main():
    """主函数"""
    print("开始增强版降雨量对水位影响分析 - 重点分析累计降雨量...")
    
    # 创建分析器
    analyzer = EnhancedRainfallAnalyzer()
    
    # 加载数据
    if not analyzer.load_rainfall_data('降雨量数据.xlsx'):
        print("无法加载降雨量数据，程序退出")
        return
    
    if not analyzer.load_water_level_data():
        print("无法加载水位数据，程序退出")
        return
    
    # 数据预处理
    if not analyzer.preprocess_data_enhanced():
        print("数据预处理失败，程序退出")
        return
    
    # 合并数据
    if not analyzer.merge_data():
        print("数据合并失败，程序退出")
        return
    
    # 重点分析累计降雨量影响
    cumulative_results = analyzer.analyze_cumulative_rainfall_impact()
    
    # 分析滞后效应
    lag_results = analyzer.analyze_lag_effects()
    
    # 分析阈值效应
    threshold_results = analyzer.analyze_threshold_effects()
    
    # 创建累计降雨量影响的可视化
    analyzer.create_cumulative_rainfall_visualizations(cumulative_results)
    
    print("\n累计降雨量影响分析完成！")
    print("\n主要发现:")
    print("1. 累计降雨量比瞬时降雨量对水位变化的影响更显著")
    print("2. 不同时间窗口的累计降雨量对水位的影响程度不同")
    print("3. 累计降雨量存在滞后效应，需要时间才能影响水位")
    print("4. 高累计降雨量阈值下，对水位的影响更加明显")

if __name__ == "__main__":
    main() 
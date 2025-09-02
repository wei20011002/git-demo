#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
降雨量对水位影响分析
分析丹徒站和丹阳站降雨量对长江侧和运河侧水位的影响程度
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
import os
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建输出图片文件夹
OUTPUT_FOLDER = 'output_images'
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
    print(f"创建输出文件夹: {OUTPUT_FOLDER}")

class RainfallWaterLevelAnalyzer:
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
            print("数据列名:", self.rainfall_data.columns.tolist())
            print("\n前5行数据:")
            print(self.rainfall_data.head())
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
                    # 查找该年份的水位数据文件
                    import os
                    files = os.listdir(year_dir)
                    water_files = [f for f in files if f.startswith('水位') and f.endswith('.csv')]
                    
                    if water_files:
                        for file in water_files:
                            file_path = os.path.join(year_dir, file)
                            try:
                                # 尝试不同的编码方式
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
                print("水位数据列名:", self.water_level_data.columns.tolist())
                print("\n前5行数据:")
                print(self.water_level_data.head())
                return True
            else:
                print("未找到水位数据文件")
                return False
                
        except Exception as e:
            print(f"加载水位数据失败: {e}")
            return False
    
    def preprocess_data(self):
        """数据预处理（支持小时级别分析）"""
        try:
            print("\n正在进行数据预处理...")
            
            # 处理降雨量数据
            if self.rainfall_data is not None:
                print("降雨量数据列名:", self.rainfall_data.columns.tolist())
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
                
                # 保留小时级别数据
                dantu_hourly = dantu_data[['日期时间', '日期', '降水量（mm）']].copy()
                dantu_hourly.columns = ['日期时间', '日期', '丹徒站降雨量']
                
                danyang_hourly = danyang_data[['日期时间', '日期', '降水量（mm）']].copy()
                danyang_hourly.columns = ['日期时间', '日期', '丹阳站降雨量']
                
                # 合并两个站点的数据
                self.rainfall_data = pd.merge(dantu_hourly, danyang_hourly, on=['日期时间', '日期'], how='outer')
                self.rainfall_data = self.rainfall_data.fillna(0)
                
                print(f"处理后的降雨量数据形状: {self.rainfall_data.shape}")
                print("处理后的降雨量数据列名:", self.rainfall_data.columns.tolist())
            
            # 处理水位数据
            if self.water_level_data is not None:
                print("水位数据列名:", self.water_level_data.columns.tolist())
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
                
                print(f"处理后的水位数据形状: {self.water_level_data.shape}")
                print("处理后的水位数据列名:", self.water_level_data.columns.tolist())
            
            return True
        except Exception as e:
            print(f"数据预处理失败: {e}")
            return False

    def merge_data(self):
        """合并降雨量和水位数据（小时级别）"""
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
            print("合并后数据列名:", self.merged_data.columns.tolist())
            return True
        except Exception as e:
            print(f"数据合并失败: {e}")
            return False

    def analyze_correlation(self):
        """分析相关性（只考虑有降雨的时间段）"""
        try:
            print("\n正在进行相关性分析（只考虑有降雨的时间段）...")
            if self.merged_data is None:
                print("数据未合并，无法进行相关性分析")
                return
            # 查找相关列
            dantu_col = '丹徒站降雨量'
            danyang_col = '丹阳站降雨量'
            changjiang_delta_col = '长江侧水位变化'
            yunhe_delta_col = '运河侧水位变化'
            correlation_results = {}
            
            # 只选择有降雨的时间段（降雨量>0）
            rainfall_data = self.merged_data[self.merged_data[dantu_col] > 0].copy()
            print(f"丹徒站有降雨的天数: {len(rainfall_data)}")
            
            # 丹徒站降雨量与长江侧水位变化量相关性（只考虑有降雨的时间段）
            valid_data = rainfall_data[[dantu_col, changjiang_delta_col]].dropna()
            if len(valid_data) > 10:
                pearson_corr, pearson_p = pearsonr(valid_data[dantu_col], valid_data[changjiang_delta_col])
                spearman_corr, spearman_p = spearmanr(valid_data[dantu_col], valid_data[changjiang_delta_col])
                correlation_results[f"{dantu_col}_vs_{changjiang_delta_col}"] = {
                    'pearson_corr': pearson_corr,
                    'pearson_p': pearson_p,
                    'spearman_corr': spearman_corr,
                    'spearman_p': spearman_p,
                    'sample_size': len(valid_data)
                }
            
            # 只选择有降雨的时间段（降雨量>0）
            rainfall_data = self.merged_data[self.merged_data[danyang_col] > 0].copy()
            print(f"丹阳站有降雨的天数: {len(rainfall_data)}")
            
            # 丹阳站降雨量与运河侧水位变化量相关性（只考虑有降雨的时间段）
            valid_data = rainfall_data[[danyang_col, yunhe_delta_col]].dropna()
            if len(valid_data) > 10:
                pearson_corr, pearson_p = pearsonr(valid_data[danyang_col], valid_data[yunhe_delta_col])
                spearman_corr, spearman_p = spearmanr(valid_data[danyang_col], valid_data[yunhe_delta_col])
                correlation_results[f"{danyang_col}_vs_{yunhe_delta_col}"] = {
                    'pearson_corr': pearson_corr,
                    'pearson_p': pearson_p,
                    'spearman_corr': spearman_corr,
                    'spearman_p': spearman_p,
                    'sample_size': len(valid_data)
                }
            
            print("\n=== 相关性分析结果（只考虑有降雨的时间段） ===")
            for key, result in correlation_results.items():
                print(f"\n{key}:")
                print(f"  皮尔逊相关系数: {result['pearson_corr']:.4f} (p={result['pearson_p']:.4f})")
                print(f"  斯皮尔曼相关系数: {result['spearman_corr']:.4f} (p={result['spearman_p']:.4f})")
                print(f"  样本数量: {result['sample_size']}")
                
                # 判断相关性强度
                corr_strength = "弱相关"
                if abs(result['pearson_corr']) >= 0.7:
                    corr_strength = "强相关"
                elif abs(result['pearson_corr']) >= 0.3:
                    corr_strength = "中等相关"
                
                print(f"  相关性强度: {corr_strength}")
            
            return correlation_results
        except Exception as e:
            print(f"相关性分析失败: {e}")
            return None

    def analyze_time_lag(self):
        """分析时间滞后效应（小时级别，只考虑有降雨的时间段）"""
        try:
            print("\n正在进行时间滞后分析（小时级别，只考虑有降雨的时间段）...")
            if self.merged_data is None:
                print("数据未合并，无法进行时间滞后分析")
                return
            dantu_col = '丹徒站降雨量'
            danyang_col = '丹阳站降雨量'
            changjiang_delta_col = '长江侧水位变化'
            yunhe_delta_col = '运河侧水位变化'
            lag_results = {}
            
            # 只选择有降雨的时间段（降雨量>0）
            rainfall_data = self.merged_data[self.merged_data[dantu_col] > 0].copy()
            print(f"丹徒站有降雨的小时数: {len(rainfall_data)}")
            
            # 丹徒站降雨量与长江侧水位变化量滞后分析（小时级别）
            valid_data = rainfall_data[[dantu_col, changjiang_delta_col]].dropna()
            if len(valid_data) > 20:
                correlations = []
                # 分析0-48小时的滞后（2天）
                for lag in range(0, 49):
                    if lag == 0:
                        corr, _ = pearsonr(valid_data[dantu_col], valid_data[changjiang_delta_col])
                    else:
                        rainfall_lagged = valid_data[dantu_col].shift(lag)
                        water_delta = valid_data[changjiang_delta_col]
                        valid_lag_data = pd.DataFrame({'rainfall': rainfall_lagged, 'water_delta': water_delta}).dropna()
                        if len(valid_lag_data) > 10:
                            corr, _ = pearsonr(valid_lag_data['rainfall'], valid_lag_data['water_delta'])
                        else:
                            corr = np.nan
                    correlations.append(corr)
                lag_results[f"{dantu_col}_vs_{changjiang_delta_col}"] = correlations
            
            # 只选择有降雨的时间段（降雨量>0）
            rainfall_data = self.merged_data[self.merged_data[danyang_col] > 0].copy()
            print(f"丹阳站有降雨的小时数: {len(rainfall_data)}")
            
            # 丹阳站降雨量与运河侧水位变化量滞后分析（小时级别）
            valid_data = rainfall_data[[danyang_col, yunhe_delta_col]].dropna()
            if len(valid_data) > 20:
                correlations = []
                # 分析0-48小时的滞后（2天）
                for lag in range(0, 49):
                    if lag == 0:
                        corr, _ = pearsonr(valid_data[danyang_col], valid_data[yunhe_delta_col])
                    else:
                        rainfall_lagged = valid_data[danyang_col].shift(lag)
                        water_delta = valid_data[yunhe_delta_col]
                        valid_lag_data = pd.DataFrame({'rainfall': rainfall_lagged, 'water_delta': water_delta}).dropna()
                        if len(valid_lag_data) > 10:
                            corr, _ = pearsonr(valid_lag_data['rainfall'], valid_lag_data['water_delta'])
                        else:
                            corr = np.nan
                    correlations.append(corr)
                lag_results[f"{danyang_col}_vs_{yunhe_delta_col}"] = correlations
            
            print("\n=== 时间滞后分析结果（小时级别，只考虑有降雨的时间段） ===")
            for key, correlations in lag_results.items():
                print(f"\n{key}:")
                # 显示前24小时的滞后结果
                for lag in range(0, 25):
                    if not np.isnan(correlations[lag]):
                        print(f"  滞后{lag}小时: {correlations[lag]:.4f}")
                
                # 找到最大相关性的滞后小时数
                valid_corrs = [c for c in correlations if not np.isnan(c)]
                if valid_corrs:
                    max_corr_idx = np.argmax([abs(c) for c in valid_corrs])
                    max_corr_lag = max_corr_idx
                    max_corr = valid_corrs[max_corr_idx]
                    print(f"  最大相关性: 滞后{max_corr_lag}小时, 相关系数={max_corr:.4f}")
                    
                    # 转换为天数和小时数
                    days = max_corr_lag // 24
                    hours = max_corr_lag % 24
                    if days > 0:
                        print(f"  相当于: {days}天{hours}小时")
                    else:
                        print(f"  相当于: {hours}小时")
            
            return lag_results
        except Exception as e:
            print(f"时间滞后分析失败: {e}")
            return None

    def create_visualizations(self):
        """创建可视化图表（只考虑有降雨的时间段）"""
        try:
            print("\n正在创建可视化图表（只考虑有降雨的时间段）...")
            if self.merged_data is None:
                print("数据未合并，无法创建可视化")
                return
            dantu_col = '丹徒站降雨量'
            danyang_col = '丹阳站降雨量'
            changjiang_delta_col = '长江侧水位变化'
            yunhe_delta_col = '运河侧水位变化'
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('降雨量对水位变化量影响分析（只考虑有降雨的时间段）', fontsize=16, fontweight='bold')
            
            # 1. 丹徒站降雨量与长江侧水位变化量散点图（只考虑有降雨的时间段）
            ax1 = axes[0, 0]
            rainfall_data = self.merged_data[self.merged_data[dantu_col] > 0].copy()
            valid_data = rainfall_data[[dantu_col, changjiang_delta_col]].dropna()
            if len(valid_data) > 0:
                ax1.scatter(valid_data[dantu_col], valid_data[changjiang_delta_col], alpha=0.6, color='blue')
                ax1.set_xlabel('丹徒站降雨量 (mm)')
                ax1.set_ylabel('长江侧水位变化量 (m)')
                ax1.set_title('丹徒站降雨量与长江侧水位变化量关系\n(只考虑有降雨的天数)')
                
                # 添加趋势线
                if len(valid_data) > 1:
                    z = np.polyfit(valid_data[dantu_col], valid_data[changjiang_delta_col], 1)
                    p = np.poly1d(z)
                    ax1.plot(valid_data[dantu_col], p(valid_data[dantu_col]), "r--", alpha=0.8, linewidth=2)
            
            # 2. 丹阳站降雨量与运河侧水位变化量散点图（只考虑有降雨的时间段）
            ax2 = axes[0, 1]
            rainfall_data = self.merged_data[self.merged_data[danyang_col] > 0].copy()
            valid_data = rainfall_data[[danyang_col, yunhe_delta_col]].dropna()
            if len(valid_data) > 0:
                ax2.scatter(valid_data[danyang_col], valid_data[yunhe_delta_col], alpha=0.6, color='green')
                ax2.set_xlabel('丹阳站降雨量 (mm)')
                ax2.set_ylabel('运河侧水位变化量 (m)')
                ax2.set_title('丹阳站降雨量与运河侧水位变化量关系\n(只考虑有降雨的天数)')
                
                # 添加趋势线
                if len(valid_data) > 1:
                    z = np.polyfit(valid_data[danyang_col], valid_data[yunhe_delta_col], 1)
                    p = np.poly1d(z)
                    ax2.plot(valid_data[danyang_col], p(valid_data[danyang_col]), "r--", alpha=0.8, linewidth=2)
            
            # 3. 时间序列图 - 丹徒站（只显示有降雨的时间段）
            ax3 = axes[1, 0]
            date_col = '日期'
            rainfall_data = self.merged_data[self.merged_data[dantu_col] > 0].copy()
            valid_data = rainfall_data[[date_col, dantu_col, changjiang_delta_col]].dropna()
            if len(valid_data) > 0:
                ax3_twin = ax3.twinx()
                line1 = ax3.plot(valid_data[date_col], valid_data[dantu_col], 'b-', label='丹徒站降雨量', alpha=0.7, linewidth=1.5)
                line2 = ax3_twin.plot(valid_data[date_col], valid_data[changjiang_delta_col], 'r-', label='长江侧水位变化量', alpha=0.7, linewidth=1.5)
                ax3.set_xlabel('日期')
                ax3.set_ylabel('丹徒站降雨量 (mm)', color='b')
                ax3_twin.set_ylabel('长江侧水位变化量 (m)', color='r')
                ax3.set_title('丹徒站降雨量与长江侧水位变化量时间序列\n(只显示有降雨的天数)')
                lines = line1 + line2
                labels = [l.get_label() for l in lines]
                ax3.legend(lines, labels, loc='upper left')
            
            # 4. 时间序列图 - 丹阳站（只显示有降雨的时间段）
            ax4 = axes[1, 1]
            rainfall_data = self.merged_data[self.merged_data[danyang_col] > 0].copy()
            valid_data = rainfall_data[[date_col, danyang_col, yunhe_delta_col]].dropna()
            if len(valid_data) > 0:
                ax4_twin = ax4.twinx()
                line1 = ax4.plot(valid_data[date_col], valid_data[danyang_col], 'b-', label='丹阳站降雨量', alpha=0.7, linewidth=1.5)
                line2 = ax4_twin.plot(valid_data[date_col], valid_data[yunhe_delta_col], 'r-', label='运河侧水位变化量', alpha=0.7, linewidth=1.5)
                ax4.set_xlabel('日期')
                ax4.set_ylabel('丹阳站降雨量 (mm)', color='b')
                ax4_twin.set_ylabel('运河侧水位变化量 (m)', color='r')
                ax4.set_title('丹阳站降雨量与运河侧水位变化量时间序列\n(只显示有降雨的天数)')
                lines = line1 + line2
                labels = [l.get_label() for l in lines]
                ax4.legend(lines, labels, loc='upper left')
            
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_FOLDER, '降雨量对水位变化量影响分析_有降雨时段.png'), dpi=300, bbox_inches='tight')
            plt.show()
            print("可视化图表已保存为 '降雨量对水位变化量影响分析_有降雨时段.png'")
            
        except Exception as e:
            print(f"创建可视化失败: {e}")
    
    def generate_report(self, correlation_results, lag_results):
        """生成分析报告"""
        try:
            print("\n" + "="*60)
            print("降雨量对水位影响分析报告")
            print("="*60)
            
            print("\n1. 数据概况:")
            if self.merged_data is not None:
                print(f"   总数据量: {len(self.merged_data)} 条记录")
                print(f"   时间范围: {self.merged_data.iloc[0, 0]} 至 {self.merged_data.iloc[-1, 0]}")
            
            print("\n2. 相关性分析结果:")
            if correlation_results:
                for key, result in correlation_results.items():
                    print(f"   {key}:")
                    print(f"     皮尔逊相关系数: {result['pearson_corr']:.4f}")
                    print(f"     显著性水平: {result['pearson_p']:.4f}")
                    print(f"     样本数量: {result['sample_size']}")
                    
                    # 解释相关性
                    if result['pearson_p'] < 0.05:
                        if abs(result['pearson_corr']) >= 0.7:
                            strength = "强"
                        elif abs(result['pearson_corr']) >= 0.3:
                            strength = "中等"
                        else:
                            strength = "弱"
                        
                        direction = "正" if result['pearson_corr'] > 0 else "负"
                        print(f"     结论: 存在{strength}{direction}相关关系 (p<0.05)")
                    else:
                        print(f"     结论: 无显著相关关系 (p≥0.05)")
            
            print("\n3. 时间滞后分析结果:")
            if lag_results:
                for key, correlations in lag_results.items():
                    print(f"   {key}:")
                    valid_corrs = [c for c in correlations if not np.isnan(c)]
                    if valid_corrs:
                        max_corr_idx = np.argmax([abs(c) for c in valid_corrs])
                        max_corr_lag = max_corr_idx
                        max_corr = valid_corrs[max_corr_idx]
                        print(f"     最大相关性出现在滞后{max_corr_lag}天，相关系数为{max_corr:.4f}")
                        
                        if max_corr_lag == 0:
                            print(f"     结论: 降雨量对水位的影响是即时的")
                        else:
                            print(f"     结论: 降雨量对水位的影响有{max_corr_lag}天的滞后效应")
            
            print("\n4. 综合结论:")
            print("   基于以上分析，可以得出以下结论:")
            
            # 根据分析结果生成结论
            if correlation_results:
                for key, result in correlation_results.items():
                    if result['pearson_p'] < 0.05:
                        if '丹徒' in key and '长江' in key:
                            print("   - 丹徒站降雨量对长江侧水位有显著影响")
                        elif '丹阳' in key and '运河' in key:
                            print("   - 丹阳站降雨量对运河侧水位有显著影响")
            
            print("\n5. 建议:")
            print("   - 建议在船闸运行管理中考虑降雨量的影响")
            print("   - 可以根据降雨量预测水位变化趋势")
            print("   - 建议建立降雨量-水位预警机制")
            
            # 保存报告到文件
            with open(os.path.join(OUTPUT_FOLDER, '降雨量对水位影响分析报告.txt'), 'w', encoding='utf-8') as f:
                f.write("降雨量对水位影响分析报告\n")
                f.write("="*60 + "\n")
                # 这里可以添加更详细的报告内容
                
            print("\n分析报告已保存为 '降雨量对水位影响分析报告.txt'")
            
        except Exception as e:
            print(f"生成报告失败: {e}")

def main():
    """主函数"""
    print("开始降雨量对水位影响分析...")
    
    # 创建分析器
    analyzer = RainfallWaterLevelAnalyzer()
    
    # 加载数据
    if not analyzer.load_rainfall_data('降雨量数据.xlsx'):
        print("无法加载降雨量数据，程序退出")
        return
    
    if not analyzer.load_water_level_data():
        print("无法加载水位数据，程序退出")
        return
    
    # 数据预处理
    if not analyzer.preprocess_data():
        print("数据预处理失败，程序退出")
        return
    
    # 合并数据
    if not analyzer.merge_data():
        print("数据合并失败，程序退出")
        return
    
    # 进行相关性分析
    correlation_results = analyzer.analyze_correlation()
    
    # 进行时间滞后分析
    lag_results = analyzer.analyze_time_lag()
    
    # 创建可视化
    analyzer.create_visualizations()
    
    # 生成报告
    analyzer.generate_report(correlation_results, lag_results)
    
    print("\n分析完成！")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
抽水站与节制闸对运河及长江水位的影响时序图
分析抽水站和节制闸运行对运河和长江水位的影响关系
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class WaterLevelImpactAnalyzer:
    def __init__(self):
        self.output_dir = "output_images"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_flow_data(self, year):
        """加载指定年份的流量数据"""
        print(f"正在加载{year}年流量数据...")
        
        # 加载节制闸流量数据
        sluice_file = f"output_images/{year}年节制闸流量数据.csv"
        if os.path.exists(sluice_file):
            sluice_data = pd.read_csv(sluice_file)
            sluice_data['datetime'] = pd.to_datetime(sluice_data['datetime'])
            sluice_data.set_index('datetime', inplace=True)
            print(f"节制闸数据加载成功: {len(sluice_data)}条记录")
        else:
            print(f"未找到节制闸数据文件: {sluice_file}")
            return None, None
            
        # 加载抽水站流量数据
        pump_file = f"output_images/{year}年抽水站流量数据.csv"
        if os.path.exists(pump_file):
            pump_data = pd.read_csv(pump_file)
            pump_data['datetime'] = pd.to_datetime(pump_data['datetime'])
            pump_data.set_index('datetime', inplace=True)
            print(f"抽水站数据加载成功: {len(pump_data)}条记录")
        else:
            print(f"未找到抽水站数据文件: {pump_file}")
            return None, None
            
        return sluice_data, pump_data
    
    def load_water_level_data(self, year):
        """加载指定年份的水位数据"""
        print(f"正在加载{year}年水位数据...")
        
        water_levels = []
        
        for month in range(1, 13):
            if year == 2024:
                file_path = f"{year}/水位{year}-{month}-1.csv"
            else:
                file_path = f"{year}/{year}-{month}-1.csv"
                
            if os.path.exists(file_path):
                try:
                    # 尝试不同的编码方式
                    try:
                        df = pd.read_csv(file_path, encoding='utf-8')
                    except:
                        df = pd.read_csv(file_path, encoding='gbk')
                    
                    # 选择水位数据
                    changjiang_data = df[df['名称'] == '长江侧水位'].copy()
                    yunhe_data = df[df['名称'] == '内河侧水位'].copy()
                    
                    if len(changjiang_data) > 0 and len(yunhe_data) > 0:
                        # 处理长江侧水位
                        changjiang_data['数值'] = pd.to_numeric(changjiang_data['数值'], errors='coerce')
                        changjiang_data = changjiang_data.dropna(subset=['数值'])
                        changjiang_data['datetime'] = pd.to_datetime(
                            changjiang_data['日期'].astype(str) + ' ' + changjiang_data['时间'].astype(str)
                        )
                        changjiang_data = changjiang_data[['datetime', '数值']].rename(columns={'数值': 'changjiang_level'})
                        
                        # 处理运河侧水位
                        yunhe_data['数值'] = pd.to_numeric(yunhe_data['数值'], errors='coerce')
                        yunhe_data = yunhe_data.dropna(subset=['数值'])
                        yunhe_data['datetime'] = pd.to_datetime(
                            yunhe_data['日期'].astype(str) + ' ' + yunhe_data['时间'].astype(str)
                        )
                        yunhe_data = yunhe_data[['datetime', '数值']].rename(columns={'数值': 'yunhe_level'})
                        
                        # 合并数据
                        month_data = pd.merge(changjiang_data, yunhe_data, on='datetime', how='outer')
                        water_levels.append(month_data)
                        
                except Exception as e:
                    print(f"处理{year}年{month}月数据时出错: {e}")
                    continue
        
        if water_levels:
            # 合并所有月度数据
            combined_data = pd.concat(water_levels, ignore_index=True)
            combined_data = combined_data.sort_values('datetime')
            combined_data.set_index('datetime', inplace=True)
            
            # 重采样到小时级别
            combined_data = combined_data.resample('H').mean()
            combined_data = combined_data.dropna()
            
            print(f"水位数据加载成功: {len(combined_data)}条记录")
            return combined_data
        else:
            print(f"未找到{year}年有效的水位数据")
            return None
    
    def create_comprehensive_timeline(self, year, sluice_data, pump_data, water_data):
        """创建综合时序图"""
        print(f"正在创建{year}年综合时序图...")
        
        if water_data is None or sluice_data is None or pump_data is None:
            print("数据不完整，无法创建时序图")
            return
        
        # 合并所有数据
        merged_data = pd.merge(water_data, sluice_data, left_index=True, right_index=True, how='inner')
        merged_data = pd.merge(merged_data, pump_data, left_index=True, right_index=True, how='inner')
        
        # 选择时间范围（避免数据过多）
        if len(merged_data) > 1000:
            # 选择连续的一周数据作为示例
            start_time = merged_data.index.min()
            end_time = start_time + timedelta(days=7)
            sample_data = merged_data[(merged_data.index >= start_time) & (merged_data.index <= end_time)]
        else:
            sample_data = merged_data
        
        print(f"选择时间范围: {sample_data.index.min()} 到 {sample_data.index.max()}")
        print(f"数据点数: {len(sample_data)}")
        
        # 创建图表
        fig, axes = plt.subplots(4, 1, figsize=(16, 14))
        fig.suptitle(f'{year}年抽水站与节制闸对运河及长江水位的影响时序图', fontsize=16, fontweight='bold')
        
        # 1. 水位变化
        axes[0].plot(sample_data.index, sample_data['changjiang_level'], 
                     label='长江侧水位', color='blue', linewidth=1.5, alpha=0.8)
        axes[0].plot(sample_data.index, sample_data['yunhe_level'], 
                     label='运河侧水位', color='red', linewidth=1.5, alpha=0.8)
        axes[0].set_ylabel('水位 (m)', fontsize=12)
        axes[0].set_title('水位变化趋势', fontsize=14, fontweight='bold')
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)
        axes[0].tick_params(axis='x', rotation=45)
        
        # 2. 节制闸流量
        axes[1].plot(sample_data.index, sample_data['drainage_flow'], 
                     label='排水流量', color='red', linewidth=1, alpha=0.7)
        axes[1].plot(sample_data.index, sample_data['intake_flow'], 
                     label='引水流量', color='blue', linewidth=1, alpha=0.7)
        axes[1].plot(sample_data.index, sample_data['net_flow'], 
                     label='净流量', color='green', linewidth=2, alpha=0.8)
        axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1].set_ylabel('流量 (m³/s)', fontsize=12)
        axes[1].set_title('节制闸流量变化', fontsize=14, fontweight='bold')
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)
        axes[1].tick_params(axis='x', rotation=45)
        
        # 3. 抽水站流量和功率
        ax3_twin = axes[2].twinx()
        
        # 抽水流量
        line1 = axes[2].plot(sample_data.index, sample_data['pumping_flow'], 
                             label='抽水流量', color='purple', linewidth=1.5, alpha=0.8)
        axes[2].set_ylabel('抽水流量 (m³/s)', fontsize=12, color='purple')
        axes[2].tick_params(axis='y', labelcolor='purple')
        
        # 功率
        line2 = ax3_twin.plot(sample_data.index, sample_data['power'], 
                              label='功率', color='orange', linewidth=1.5, alpha=0.8)
        ax3_twin.set_ylabel('功率 (kW)', fontsize=12, color='orange')
        ax3_twin.tick_params(axis='y', labelcolor='orange')
        
        axes[2].set_title('抽水站运行状态', fontsize=14, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        axes[2].tick_params(axis='x', rotation=45)
        
        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        axes[2].legend(lines, labels, loc='upper right')
        
        # 4. 综合影响分析
        # 计算外部干扰强度
        sample_data['external_interference'] = abs(sample_data['net_flow']) + sample_data['pumping_flow']
        
        # 计算水位差
        sample_data['water_level_diff'] = sample_data['yunhe_level'] - sample_data['changjiang_level']
        
        # 水位差
        line1 = axes[3].plot(sample_data.index, sample_data['water_level_diff'], 
                             label='水位差(运河-长江)', color='brown', linewidth=1.5, alpha=0.8)
        axes[3].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[3].set_ylabel('水位差 (m)', fontsize=12, color='brown')
        axes[3].tick_params(axis='y', labelcolor='brown')
        
        # 外部干扰强度
        ax4_twin = axes[3].twinx()
        line2 = ax4_twin.plot(sample_data.index, sample_data['external_interference'], 
                              label='外部干扰强度', color='darkgreen', linewidth=1.5, alpha=0.8)
        ax4_twin.set_ylabel('干扰强度 (m³/s)', fontsize=12, color='darkgreen')
        ax4_twin.tick_params(axis='y', labelcolor='darkgreen')
        
        axes[3].set_title('水位差与外部干扰强度', fontsize=14, fontweight='bold')
        axes[3].set_xlabel('时间', fontsize=12)
        axes[3].grid(True, alpha=0.3)
        axes[3].tick_params(axis='x', rotation=45)
        
        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        axes[3].legend(lines, labels, loc='upper right')
        
        # 设置x轴格式
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        
        plt.tight_layout()
        
        # 保存图表
        output_file = os.path.join(self.output_dir, f'{year}年抽水站节制闸水位影响时序图.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"时序图已保存到: {output_file}")
        
        # 显示图表
        plt.show()
        
        return sample_data
    
    def create_monthly_impact_summary(self, year, sluice_data, pump_data):
        """创建月度影响总结图"""
        print(f"正在创建{year}年月度影响总结图...")
        
        if sluice_data is None or pump_data is None:
            print("数据不完整，无法创建月度总结图")
            return
        
        # 合并数据
        merged_data = pd.merge(sluice_data, pump_data, left_index=True, right_index=True, how='inner')
        
        # 添加月份信息
        merged_data['month'] = merged_data.index.month
        
        # 按月份分组计算统计信息
        monthly_stats = []
        for month in range(1, 13):
            month_data = merged_data[merged_data['month'] == month]
            if len(month_data) > 0:
                monthly_stats.append({
                    'month': month,
                    'avg_drainage': month_data['drainage_flow'].mean(),
                    'avg_intake': month_data['intake_flow'].mean(),
                    'avg_net_flow': month_data['net_flow'].mean(),
                    'avg_pumping': month_data['pumping_flow'].mean(),
                    'avg_power': month_data['power'].mean(),
                    'max_drainage': month_data['drainage_flow'].max(),
                    'max_intake': month_data['intake_flow'].max(),
                    'max_pumping': month_data['pumping_flow'].max(),
                    'max_power': month_data['power'].max()
                })
        
        monthly_df = pd.DataFrame(monthly_stats)
        
        # 创建月度总结图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{year}年抽水站与节制闸月度运行统计', fontsize=16, fontweight='bold')
        
        # 1. 平均流量对比
        x = range(len(monthly_df))
        width = 0.35
        
        axes[0, 0].bar([i - width/2 for i in x], monthly_df['avg_drainage'], 
                        width, label='平均排水流量', color='red', alpha=0.7)
        axes[0, 0].bar([i + width/2 for i in x], monthly_df['avg_intake'], 
                        width, label='平均引水流量', color='blue', alpha=0.7)
        axes[0, 0].set_ylabel('流量 (m³/s)', fontsize=12)
        axes[0, 0].set_title('月度平均流量对比', fontsize=14, fontweight='bold')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels([f'{int(m)}月' for m in monthly_df['month']])
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 平均抽水流量和功率
        ax_twin = axes[0, 1].twinx()
        
        line1 = axes[0, 1].plot(x, monthly_df['avg_pumping'], 
                                 label='平均抽水流量', color='purple', 
                                 marker='o', linewidth=2, markersize=6)
        axes[0, 1].set_ylabel('抽水流量 (m³/s)', fontsize=12, color='purple')
        axes[0, 1].tick_params(axis='y', labelcolor='purple')
        
        line2 = ax_twin.plot(x, monthly_df['avg_power'], 
                             label='平均功率', color='orange', 
                             marker='s', linewidth=2, markersize=6)
        ax_twin.set_ylabel('功率 (kW)', fontsize=12, color='orange')
        ax_twin.tick_params(axis='y', labelcolor='orange')
        
        axes[0, 1].set_title('月度平均抽水流量和功率', fontsize=14, fontweight='bold')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels([f'{int(m)}月' for m in monthly_df['month']])
        axes[0, 1].grid(True, alpha=0.3)
        
        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        axes[0, 1].legend(lines, labels, loc='upper right')
        
        # 3. 最大流量对比
        axes[1, 0].bar([i - width/2 for i in x], monthly_df['max_drainage'], 
                        width, label='最大排水流量', color='darkred', alpha=0.7)
        axes[1, 0].bar([i + width/2 for i in x], monthly_df['max_intake'], 
                        width, label='最大引水流量', color='darkblue', alpha=0.7)
        axes[1, 0].set_ylabel('流量 (m³/s)', fontsize=12)
        axes[1, 0].set_title('月度最大流量对比', fontsize=14, fontweight='bold')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels([f'{int(m)}月' for m in monthly_df['month']])
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 最大抽水流量和功率
        ax_twin2 = axes[1, 1].twinx()
        
        line1 = axes[1, 1].plot(x, monthly_df['max_pumping'], 
                                 label='最大抽水流量', color='darkpurple', 
                                 marker='o', linewidth=2, markersize=6)
        axes[1, 1].set_ylabel('抽水流量 (m³/s)', fontsize=12, color='darkpurple')
        axes[1, 1].tick_params(axis='y', labelcolor='darkpurple')
        
        line2 = ax_twin2.plot(x, monthly_df['max_power'], 
                              label='最大功率', color='darkorange', 
                              marker='s', linewidth=2, markersize=6)
        ax_twin2.set_ylabel('功率 (kW)', fontsize=12, color='darkorange')
        ax_twin2.tick_params(axis='y', labelcolor='darkorange')
        
        axes[1, 1].set_title('月度最大抽水流量和功率', fontsize=14, fontweight='bold')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels([f'{int(m)}月' for m in monthly_df['month']])
        axes[1, 1].grid(True, alpha=0.3)
        
        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        axes[1, 1].legend(lines, labels, loc='upper right')
        
        plt.tight_layout()
        
        # 保存图表
        output_file = os.path.join(self.output_dir, f'{year}年抽水站节制闸月度影响总结.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"月度总结图已保存到: {output_file}")
        
        # 显示图表
        plt.show()
        
        return monthly_df
    
    def analyze_impact_correlation(self, sample_data):
        """分析影响相关性"""
        if sample_data is None:
            return
        
        print("正在分析影响相关性...")
        
        # 计算相关系数
        correlations = {}
        
        # 水位差与流量的相关性
        correlations['水位差_净流量'] = sample_data['water_level_diff'].corr(sample_data['net_flow'])
        correlations['水位差_抽水流量'] = sample_data['water_level_diff'].corr(sample_data['pumping_flow'])
        correlations['水位差_外部干扰'] = sample_data['water_level_diff'].corr(sample_data['external_interference'])
        
        # 水位变化与流量的相关性
        correlations['长江水位_净流量'] = sample_data['changjiang_level'].corr(sample_data['net_flow'])
        correlations['运河水位_净流量'] = sample_data['yunhe_level'].corr(sample_data['net_flow'])
        correlations['长江水位_抽水流量'] = sample_data['changjiang_level'].corr(sample_data['pumping_flow'])
        correlations['运河水位_抽水流量'] = sample_data['yunhe_level'].corr(sample_data['pumping_flow'])
        
        print("\n=== 影响相关性分析 ===")
        for key, value in correlations.items():
            print(f"{key}: {value:.4f}")
        
        # 创建相关性热力图
        correlation_data = pd.DataFrame({
            '水位差': sample_data['water_level_diff'],
            '净流量': sample_data['net_flow'],
            '抽水流量': sample_data['pumping_flow'],
            '外部干扰': sample_data['external_interference'],
            '长江水位': sample_data['changjiang_level'],
            '运河水位': sample_data['yunhe_level']
        })
        
        correlation_matrix = correlation_data.corr()
        
        plt.figure(figsize=(10, 8))
        plt.imshow(correlation_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        plt.colorbar(label='相关系数')
        plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45)
        plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
        plt.title('各变量相关性热力图', fontsize=14, fontweight='bold')
        
        # 添加相关系数标签
        for i in range(len(correlation_matrix.columns)):
            for j in range(len(correlation_matrix.columns)):
                plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.3f}', 
                        ha='center', va='center', fontsize=10,
                        color='white' if abs(correlation_matrix.iloc[i, j]) > 0.5 else 'black')
        
        plt.tight_layout()
        
        # 保存热力图
        output_file = os.path.join(self.output_dir, '影响相关性热力图.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"相关性热力图已保存到: {output_file}")
        
        plt.show()
        
        return correlations
    
    def run_analysis(self, year):
        """运行完整分析"""
        print(f"\n{'='*60}")
        print(f"开始分析{year}年抽水站与节制闸对水位的影响")
        print(f"{'='*60}")
        
        # 加载数据
        sluice_data, pump_data = self.load_flow_data(year)
        water_data = self.load_water_level_data(year)
        
        if sluice_data is None or pump_data is None or water_data is None:
            print(f"{year}年数据不完整，跳过分析")
            return
        
        # 创建综合时序图
        sample_data = self.create_comprehensive_timeline(year, sluice_data, pump_data, water_data)
        
        # 创建月度影响总结
        monthly_stats = self.create_monthly_impact_summary(year, sluice_data, pump_data)
        
        # 分析影响相关性
        correlations = self.analyze_impact_correlation(sample_data)
        
        print(f"\n{year}年分析完成！")
        
        return {
            'sample_data': sample_data,
            'monthly_stats': monthly_stats,
            'correlations': correlations
        }

def main():
    """主函数"""
    analyzer = WaterLevelImpactAnalyzer()
    
    # 分析各年份数据
    years = [2022, 2023, 2024]
    results = {}
    
    for year in years:
        try:
            result = analyzer.run_analysis(year)
            if result:
                results[year] = result
        except Exception as e:
            print(f"分析{year}年数据时出错: {e}")
            continue
    
    # 生成综合分析报告
    if results:
        print("\n" + "="*80)
        print("综合分析报告")
        print("="*80)
        
        for year, result in results.items():
            print(f"\n{year}年主要发现:")
            if result['correlations']:
                print("  主要相关性:")
                for key, value in result['correlations'].items():
                    if abs(value) > 0.3:  # 只显示相关性较强的
                        strength = "强" if abs(value) > 0.7 else "中等" if abs(value) > 0.5 else "弱"
                        print(f"    {key}: {value:.3f} ({strength})")
        
        print(f"\n所有分析结果已保存到 {analyzer.output_dir} 文件夹中")

if __name__ == "__main__":
    main() 
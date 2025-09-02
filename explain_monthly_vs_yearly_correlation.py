import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
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

def explain_correlation_difference():
    """解释月度相关系数为什么大于年度平均相关系数"""
    
    print("潮位与水位相关性分析：月度 vs 年度差异解释")
    print("=" * 60)
    
    # 年度相关系数（来自之前的分析）
    yearly_correlations = {
        2022: 0.5034,
        2023: 0.7186,
        2024: 0.4741
    }
    
    # 月度相关系数（来自月度分析）
    monthly_correlations = {
        2022: {
            1: 0.9160, 2: 0.8866, 3: 0.8376, 4: 0.8425, 5: 0.8454, 6: 0.8642,
            7: 0.6498, 8: 0.8333, 9: 0.9242, 10: 0.9170, 11: 0.9233, 12: 0.8837
        },
        2023: {
            1: 0.9391, 2: 0.9321, 3: 0.9221, 4: 0.8809, 5: 0.9126, 6: 0.8498,
            7: 0.6763, 10: 0.8216, 11: 0.8873, 12: 0.9180  # 8、9月缺失
        },
        2024: {
            1: 0.9108, 2: 0.8297, 3: 0.8610, 4: 0.6504, 5: 0.6877, 6: 0.5119,
            7: 0.6836, 8: 0.5994, 9: 0.7502, 10: 0.9255, 11: 0.8797, 12: 0.9587
        }
    }
    
    print("\n1. 数据对比分析")
    print("-" * 40)
    
    for year in [2022, 2023, 2024]:
        yearly_corr = yearly_correlations[year]
        monthly_corrs = monthly_correlations[year]
        monthly_avg = np.mean(list(monthly_corrs.values()))
        
        print(f"\n{year}年:")
        print(f"  年度相关系数: {yearly_corr:.4f}")
        print(f"  月度平均相关系数: {monthly_avg:.4f}")
        print(f"  差异: {monthly_avg - yearly_corr:.4f}")
        print(f"  月度相关系数范围: {min(monthly_corrs.values()):.4f} - {max(monthly_corrs.values()):.4f}")
    
    print("\n2. 原因分析")
    print("-" * 40)
    
    print("\n2.1 时间尺度效应")
    print("   - 年度分析：包含全年所有数据，时间跨度大")
    print("   - 月度分析：只分析单月数据，时间跨度小")
    print("   - 长时间跨度会引入更多噪声和外部因素")
    
    print("\n2.2 季节性影响")
    print("   - 不同月份的水文条件差异很大")
    print("   - 汛期、枯水期对相关性影响显著")
    print("   - 年度分析混合了不同季节的特征")
    
    print("\n2.3 数据异质性")
    print("   - 年度数据包含多种水文状态")
    print("   - 月度数据相对同质，相关性更稳定")
    print("   - 异质性会降低整体相关性")
    
    print("\n2.4 外部干扰因素")
    print("   - 年度数据包含更多外部干扰（如极端天气）")
    print("   - 月度数据受外部干扰相对较少")
    print("   - 干扰因素会降低相关性强度")
    
    # 创建可视化图表
    create_comparison_plots(yearly_correlations, monthly_correlations)
    
    print("\n3. 统计验证")
    print("-" * 40)
    
    # 计算统计指标
    all_yearly = list(yearly_correlations.values())
    all_monthly_avgs = []
    
    for year in [2022, 2023, 2024]:
        monthly_avg = np.mean(list(monthly_correlations[year].values()))
        all_monthly_avgs.append(monthly_avg)
    
    print(f"年度相关系数统计:")
    print(f"  平均值: {np.mean(all_yearly):.4f}")
    print(f"  标准差: {np.std(all_yearly):.4f}")
    print(f"  变异系数: {np.std(all_yearly)/np.mean(all_yearly):.4f}")
    
    print(f"\n月度平均相关系数统计:")
    print(f"  平均值: {np.mean(all_monthly_avgs):.4f}")
    print(f"  标准差: {np.std(all_monthly_avgs):.4f}")
    print(f"  变异系数: {np.std(all_monthly_avgs)/np.mean(all_monthly_avgs):.4f}")
    
    print(f"\n差异分析:")
    print(f"  月度平均 - 年度平均: {np.mean(all_monthly_avgs) - np.mean(all_yearly):.4f}")
    print(f"  相对提升: {(np.mean(all_monthly_avgs) - np.mean(all_yearly))/np.mean(all_yearly)*100:.1f}%")
    
    print("\n4. 结论")
    print("-" * 40)
    print("月度相关系数大于年度平均相关系数是正常现象，主要原因包括：")
    print("1. 时间尺度效应：短时间尺度数据更同质")
    print("2. 季节性影响：月度数据避免了季节间差异")
    print("3. 数据异质性：年度数据包含更多变异性")
    print("4. 外部干扰：长时间跨度引入更多噪声")
    print("\n这种现象在水文学研究中很常见，说明潮位与水位的关系在不同时间尺度上表现不同。")

def create_comparison_plots(yearly_correlations, monthly_correlations):
    """创建对比图表"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    years = list(yearly_correlations.keys())
    yearly_values = list(yearly_correlations.values())
    monthly_avgs = []
    
    for year in years:
        monthly_avg = np.mean(list(monthly_correlations[year].values()))
        monthly_avgs.append(monthly_avg)
    
    # 1. 年度 vs 月度平均对比
    x = np.arange(len(years))
    width = 0.35
    
    ax1.bar(x - width/2, yearly_values, width, label='年度相关系数', color='skyblue', alpha=0.7)
    ax1.bar(x + width/2, monthly_avgs, width, label='月度平均相关系数', color='lightcoral', alpha=0.7)
    
    ax1.set_xlabel('年份')
    ax1.set_ylabel('皮尔逊相关系数')
    ax1.set_title('年度 vs 月度平均相关系数对比')
    ax1.set_xticks(x)
    ax1.set_xticklabels(years)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, (yearly, monthly) in enumerate(zip(yearly_values, monthly_avgs)):
        ax1.text(i - width/2, yearly + 0.02, f'{yearly:.3f}', ha='center', va='bottom')
        ax1.text(i + width/2, monthly + 0.02, f'{monthly:.3f}', ha='center', va='bottom')
    
    # 2. 差异分析
    differences = [monthly - yearly for monthly, yearly in zip(monthly_avgs, yearly_values)]
    ax2.bar(years, differences, color='lightgreen', alpha=0.7)
    ax2.set_xlabel('年份')
    ax2.set_ylabel('差异 (月度 - 年度)')
    ax2.set_title('月度与年度相关系数差异')
    ax2.grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, diff in enumerate(differences):
        ax2.text(i, diff + (0.01 if diff >= 0 else -0.01), f'{diff:.3f}', 
                ha='center', va='bottom' if diff >= 0 else 'top')
    
    # 3. 月度相关系数分布
    all_monthly_values = []
    for year in years:
        all_monthly_values.extend(list(monthly_correlations[year].values()))
    
    ax3.hist(all_monthly_values, bins=15, color='orange', alpha=0.7, edgecolor='black')
    ax3.axvline(np.mean(all_monthly_values), color='red', linestyle='--', 
                label=f'平均值: {np.mean(all_monthly_values):.3f}')
    ax3.set_xlabel('月度相关系数')
    ax3.set_ylabel('频次')
    ax3.set_title('月度相关系数分布')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 年度相关系数分布
    ax4.hist(yearly_values, bins=5, color='blue', alpha=0.7, edgecolor='black')
    ax4.axvline(np.mean(yearly_values), color='red', linestyle='--', 
                label=f'平均值: {np.mean(yearly_values):.3f}')
    ax4.set_xlabel('年度相关系数')
    ax4.set_ylabel('频次')
    ax4.set_title('年度相关系数分布')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, '月度vs年度相关系数对比分析.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n对比分析图表已保存为: 月度vs年度相关系数对比分析.png")

def demonstrate_with_simple_example():
    """用简单例子演示这个现象"""
    
    print("\n5. 简单示例演示")
    print("-" * 40)
    
    # 模拟数据：假设有两个季节，每个季节有不同的相关性
    np.random.seed(42)
    
    # 季节1：强相关性
    season1_tide = np.random.normal(0, 1, 1000)
    season1_water = 0.8 * season1_tide + np.random.normal(0, 0.3, 1000)
    season1_corr = np.corrcoef(season1_tide, season1_water)[0, 1]
    
    # 季节2：弱相关性
    season2_tide = np.random.normal(0, 1, 1000)
    season2_water = 0.3 * season2_tide + np.random.normal(0, 0.8, 1000)
    season2_corr = np.corrcoef(season2_tide, season2_water)[0, 1]
    
    # 合并数据
    combined_tide = np.concatenate([season1_tide, season2_tide])
    combined_water = np.concatenate([season1_water, season2_water])
    combined_corr = np.corrcoef(combined_tide, combined_water)[0, 1]
    
    print("模拟示例：")
    print(f"  季节1相关系数: {season1_corr:.4f}")
    print(f"  季节2相关系数: {season2_corr:.4f}")
    print(f"  季节平均相关系数: {(season1_corr + season2_corr)/2:.4f}")
    print(f"  合并数据相关系数: {combined_corr:.4f}")
    print(f"  差异: {(season1_corr + season2_corr)/2 - combined_corr:.4f}")
    
    print("\n这个例子说明：")
    print("- 当数据包含不同相关性的子集时")
    print("- 子集的相关性会高于整体相关性")
    print("- 这是数据异质性导致的正常现象")

if __name__ == "__main__":
    explain_correlation_difference()
    demonstrate_with_simple_example() 
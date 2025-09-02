import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def create_water_level_difference_flowchart():
    """创建水位差预测模型流程图"""
    fig, ax = plt.subplots(1, 1, figsize=(20, 16))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 16)
    ax.axis('off')
    
    # 标题
    title_box = FancyBboxPatch((1, 15), 18, 0.8, 
                               boxstyle="round,pad=0.1", 
                               facecolor='lightblue', 
                               edgecolor='black', 
                               linewidth=2)
    ax.add_patch(title_box)
    ax.text(10, 15.4, '水位差预测模型 - 双模块架构', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    # 数据输入层
    input_box = FancyBboxPatch((2, 13.5), 16, 1.2, 
                               boxstyle="round,pad=0.1", 
                               facecolor='lightgreen', 
                               edgecolor='black', 
                               linewidth=2)
    ax.add_patch(input_box)
    ax.text(10, 14.1, '数据输入层', ha='center', va='center', fontsize=14, fontweight='bold')
    
    # 数据输入层的子框
    tide_box = FancyBboxPatch((2.5, 13.7), 4, 0.6, 
                              boxstyle="round,pad=0.05", 
                              facecolor='lightyellow', 
                              edgecolor='black', 
                              linewidth=1)
    ax.add_patch(tide_box)
    ax.text(4.5, 14, '潮位数据\n(延迟4-8h)', ha='center', va='center', fontsize=10)
    
    flow_box = FancyBboxPatch((7, 13.7), 4, 0.6, 
                              boxstyle="round,pad=0.05", 
                              facecolor='lightyellow', 
                              edgecolor='black', 
                              linewidth=1)
    ax.add_patch(flow_box)
    ax.text(9, 14, '流量数据\n(节制闸+\n抽水站)', ha='center', va='center', fontsize=10)
    
    history_box = FancyBboxPatch((11.5, 13.7), 4, 0.6, 
                                 boxstyle="round,pad=0.05", 
                                 facecolor='lightyellow', 
                                 edgecolor='black', 
                                 linewidth=1)
    ax.add_patch(history_box)
    ax.text(13.5, 14, '历史\n水位数据', ha='center', va='center', fontsize=10)
    
    # 特征工程层
    feature_box = FancyBboxPatch((2, 11.8), 16, 1.2, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor='lightcoral', 
                                 edgecolor='black', 
                                 linewidth=2)
    ax.add_patch(feature_box)
    ax.text(10, 12.4, '特征工程层', ha='center', va='center', fontsize=14, fontweight='bold')
    
    # 长江侧特征
    yangtze_feature = FancyBboxPatch((2.5, 12), 6.5, 0.6, 
                                     boxstyle="round,pad=0.05", 
                                     facecolor='lightblue', 
                                     edgecolor='black', 
                                     linewidth=1)
    ax.add_patch(yangtze_feature)
    ax.text(5.75, 12.3, '长江侧特征 (18维输入)', ha='center', va='center', fontsize=10)
    
    # 运河侧特征
    yunhe_feature = FancyBboxPatch((11, 12), 6.5, 0.6, 
                                   boxstyle="round,pad=0.05", 
                                   facecolor='lightblue', 
                                   edgecolor='black', 
                                   linewidth=1)
    ax.add_patch(yunhe_feature)
    ax.text(14.25, 12.3, '运河侧特征 (30维输入)', ha='center', va='center', fontsize=10)
    
    # 双模块预测层
    module_box = FancyBboxPatch((1, 8.5), 18, 2.5, 
                                boxstyle="round,pad=0.1", 
                                facecolor='lightgray', 
                                edgecolor='black', 
                                linewidth=2)
    ax.add_patch(module_box)
    
    # 模块1: 长江水位预测
    yangtze_module = FancyBboxPatch((1.5, 9.5), 8, 1.5, 
                                    boxstyle="round,pad=0.1", 
                                    facecolor='lightgreen', 
                                    edgecolor='black', 
                                    linewidth=2)
    ax.add_patch(yangtze_module)
    ax.text(5.5, 10.25, '模块1: 长江水位预测', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # 长江模块内部
    yangtze_lstm = FancyBboxPatch((2, 9.7), 6, 0.6, 
                                   boxstyle="round,pad=0.05", 
                                   facecolor='lightyellow', 
                                   edgecolor='black', 
                                   linewidth=1)
    ax.add_patch(yangtze_lstm)
    ax.text(5, 10, 'LSTM模型 (1层LSTM)', ha='center', va='center', fontsize=10)
    
    # 模块2: 运河水位预测
    yunhe_module = FancyBboxPatch((10.5, 9.5), 8, 1.5, 
                                  boxstyle="round,pad=0.1", 
                                  facecolor='lightgreen', 
                                  edgecolor='black', 
                                  linewidth=2)
    ax.add_patch(yunhe_module)
    ax.text(14.5, 10.25, '模块2: 运河水位预测', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # 运河模块内部
    yunhe_lstm = FancyBboxPatch((11, 9.7), 6, 0.6, 
                                 boxstyle="round,pad=0.05", 
                                 facecolor='lightyellow', 
                                 edgecolor='black', 
                                 linewidth=1)
    ax.add_patch(yunhe_lstm)
    ax.text(14, 10, 'LSTM模型 (2层LSTM)', ha='center', va='center', fontsize=10)
    
    # 水位差计算层
    diff_box = FancyBboxPatch((2, 6.8), 16, 1.2, 
                              boxstyle="round,pad=0.1", 
                              facecolor='lightblue', 
                              edgecolor='black', 
                              linewidth=2)
    ax.add_patch(diff_box)
    ax.text(10, 7.4, '水位差计算层', ha='center', va='center', fontsize=14, fontweight='bold')
    
    # 水位差计算内部
    diff_calc = FancyBboxPatch((3, 7), 14, 0.6, 
                               boxstyle="round,pad=0.05", 
                               facecolor='lightyellow', 
                               edgecolor='black', 
                               linewidth=1)
    ax.add_patch(diff_calc)
    ax.text(10, 7.3, '水位差 = |长江水位 - 运河水位|  输出: 3小时水位差预测', 
            ha='center', va='center', fontsize=10)
    
    # 结果输出层
    output_box = FancyBboxPatch((2, 5.3), 16, 1.2, 
                                boxstyle="round,pad=0.1", 
                                facecolor='lightcoral', 
                                edgecolor='black', 
                                linewidth=2)
    ax.add_patch(output_box)
    ax.text(10, 5.9, '结果输出层', ha='center', va='center', fontsize=14, fontweight='bold')
    
    # 结果输出内部
    output_inner = FancyBboxPatch((3, 5.5), 14, 0.6, 
                                  boxstyle="round,pad=0.05", 
                                  facecolor='lightyellow', 
                                  edgecolor='black', 
                                  linewidth=1)
    ax.add_patch(output_inner)
    ax.text(10, 5.8, 'CSV文件: 包含所有预测结果 - 长江水位(1h,2h,3h) - 运河水位(1h,2h,3h) - 水位差(1h,2h,3h)', 
            ha='center', va='center', fontsize=9)
    
    # 添加连接箭头
    # 标题到数据输入层
    ax.annotate('', xy=(10, 13.5), xytext=(10, 15.8),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # 数据输入层到特征工程层
    ax.annotate('', xy=(10, 11.8), xytext=(10, 13.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # 特征工程层到双模块层
    ax.annotate('', xy=(10, 8.5), xytext=(10, 11.8),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # 双模块层到水位差计算层
    ax.annotate('', xy=(10, 6.8), xytext=(10, 8.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # 水位差计算层到结果输出层
    ax.annotate('', xy=(10, 5.3), xytext=(10, 6.8),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # 添加模块1和模块2的输入特征说明
    # 长江模块输入特征
    yangtze_input = FancyBboxPatch((1.5, 8.8), 8, 0.6, 
                                    boxstyle="round,pad=0.05", 
                                    facecolor='lightyellow', 
                                    edgecolor='black', 
                                    linewidth=1)
    ax.add_patch(yangtze_input)
    ax.text(5.5, 9.1, '输入: 18维特征 - 历史12h水位(12维) - 延迟潮位(4维) - 流量特征(2维)', 
            ha='center', va='center', fontsize=8)
    
    # 运河模块输入特征
    yunhe_input = FancyBboxPatch((10.5, 8.8), 8, 0.6, 
                                  boxstyle="round,pad=0.05", 
                                  facecolor='lightyellow', 
                                  edgecolor='black', 
                                  linewidth=1)
    ax.add_patch(yunhe_input)
    ax.text(14.5, 9.1, '输入: 30维特征 - 长江12h水位(12维) - 运河12h水位(12维) - 延迟潮位(4维) - 流量特征(2维)', 
            ha='center', va='center', fontsize=8)
    
    # 添加输出说明
    yangtze_output = FancyBboxPatch((1.5, 8.3), 8, 0.4, 
                                     boxstyle="round,pad=0.05", 
                                     facecolor='lightyellow', 
                                     edgecolor='black', 
                                     linewidth=1)
    ax.add_patch(yangtze_output)
    ax.text(5.5, 8.5, '输出: 3小时预测', ha='center', va='center', fontsize=9)
    
    yunhe_output = FancyBboxPatch((10.5, 8.3), 8, 0.4, 
                                   boxstyle="round,pad=0.05", 
                                   facecolor='lightyellow', 
                                   edgecolor='black', 
                                   linewidth=1)
    ax.add_patch(yunhe_output)
    ax.text(14.5, 8.5, '输出: 3小时预测', ha='center', va='center', fontsize=9)
    
    plt.title('水位差预测模型流程图', fontsize=18, fontweight='bold', pad=20)
    plt.tight_layout()
    
    return fig

def create_module1_detail():
    """创建模块1: 长江水位预测的详细结构图"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # 标题
    ax.text(6, 7.5, '模块1: 长江水位预测 (LSTM模型)', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    # 输入特征框
    input_box = FancyBboxPatch((0.5, 5.5), 11, 1.5, 
                               boxstyle="round,pad=0.1", 
                               facecolor='lightblue', 
                               edgecolor='black', 
                               linewidth=2)
    ax.add_patch(input_box)
    ax.text(6, 6.25, '输入特征 (18维)', ha='center', va='center', fontsize=14, fontweight='bold')
    
    # 特征详细说明
    ax.text(1, 6, '├── 历史12小时水位数据 (12维)', fontsize=10)
    ax.text(1.5, 5.7, '├── 延迟潮位特征 (4维)', fontsize=10)
    ax.text(1.5, 5.4, '└── 流量特征 (2维)', fontsize=10)
    
    # LSTM结构框
    lstm_box = FancyBboxPatch((0.5, 2.5), 11, 2.5, 
                              boxstyle="round,pad=0.1", 
                              facecolor='lightgreen', 
                              edgecolor='black', 
                              linewidth=2)
    ax.add_patch(lstm_box)
    ax.text(6, 4.75, 'LSTM结构:', ha='center', va='center', fontsize=14, fontweight='bold')
    
    # 网络结构
    ax.text(1, 4.5, '输入层(18) → LSTM层(64) → 全连接层', fontsize=10)
    ax.text(1, 4.2, '    ↓           ↓              ↓', fontsize=10)
    ax.text(1, 3.9, '  序列输入    单向LSTM       64→32→3', fontsize=10)
    ax.text(1, 3.6, '               Dropout       ReLU+Dropout', fontsize=10)
    
    # 输出说明
    output_box = FancyBboxPatch((0.5, 0.5), 11, 1.5, 
                                boxstyle="round,pad=0.1", 
                                facecolor='lightcoral', 
                                edgecolor='black', 
                                linewidth=2)
    ax.add_patch(output_box)
    ax.text(6, 1.25, '输出: 未来1小时、2小时、3小时的长江水位预测', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    
    plt.title('模块1: 长江水位预测详细结构', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    return fig

def create_module2_detail():
    """创建模块2: 运河水位预测的详细结构图"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # 标题
    ax.text(6, 7.5, '模块2: 运河水位预测 (LSTM模型)', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    # 输入特征框
    input_box = FancyBboxPatch((0.5, 5.5), 11, 1.5, 
                               boxstyle="round,pad=0.1", 
                               facecolor='lightblue', 
                               edgecolor='black', 
                               linewidth=2)
    ax.add_patch(input_box)
    ax.text(6, 6.25, '输入特征 (30维)', ha='center', va='center', fontsize=14, fontweight='bold')
    
    # 特征详细说明
    ax.text(1, 6, '├── 长江侧历史12小时水位 (12维)', fontsize=10)
    ax.text(1, 5.7, '├── 运河侧历史12小时水位 (12维)', fontsize=10)
    ax.text(1, 5.4, '├── 延迟潮位特征 (4维)', fontsize=10)
    ax.text(1, 5.1, '└── 流量特征 (2维)', fontsize=10)
    
    # LSTM结构框
    lstm_box = FancyBboxPatch((0.5, 2.5), 11, 2.5, 
                              boxstyle="round,pad=0.1", 
                              facecolor='lightgreen', 
                              edgecolor='black', 
                              linewidth=2)
    ax.add_patch(lstm_box)
    ax.text(6, 4.75, 'LSTM结构:', ha='center', va='center', fontsize=14, fontweight='bold')
    
    # 网络结构
    ax.text(1, 4.5, '输入层(30) → LSTM层1(128) → LSTM层2(128) → 全连接层', fontsize=10)
    ax.text(1, 4.2, '    ↓           ↓              ↓              ↓', fontsize=10)
    ax.text(1, 3.9, '  序列输入    双向LSTM      双向LSTM       128→64→32→3', fontsize=10)
    ax.text(1, 3.6, '               Dropout       Dropout', fontsize=10)
    
    # 输出说明
    output_box = FancyBboxPatch((0.5, 0.5), 11, 1.5, 
                                boxstyle="round,pad=0.1", 
                                facecolor='lightcoral', 
                                edgecolor='black', 
                                linewidth=2)
    ax.add_patch(output_box)
    ax.text(6, 1.25, '输出: 未来1小时、2小时、3小时的运河水位预测', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    
    plt.title('模块2: 运河水位预测详细结构', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    return fig

def main():
    """主函数"""
    print("正在生成水位差预测模型流程图...")
    
    # 创建主流程图
    main_fig = create_water_level_difference_flowchart()
    main_fig.savefig('水位差预测模型主流程图.png', dpi=300, bbox_inches='tight')
    print("主流程图已保存为: 水位差预测模型主流程图.png")
    
    # 创建模块1详细图
    module1_fig = create_module1_detail()
    module1_fig.savefig('模块1_长江水位预测详细结构.png', dpi=300, bbox_inches='tight')
    print("模块1详细图已保存为: 模块1_长江水位预测详细结构.png")
    
    # 创建模块2详细图
    module2_fig = create_module2_detail()
    module2_fig.savefig('模块2_运河水位预测详细结构.png', dpi=300, bbox_inches='tight')
    print("模块2详细图已保存为: 模块2_运河水位预测详细结构.png")
    
    # 显示所有图
    plt.show()
    
    print("所有流程图生成完成！")

if __name__ == '__main__':
    main() 
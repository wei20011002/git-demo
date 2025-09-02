import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def create_simple_architecture_diagram():
    """创建简化的长江水位预测模型架构图"""
    
    # 创建图形和坐标轴
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # 定义颜色
    input_color = '#E3F2FD'      # 浅蓝色 - 输入层
    hidden_color = '#FFF3E0'     # 浅橙色 - 隐藏层
    output_color = '#F3E5F5'     # 浅紫色 - 输出层
    connection_color = '#424242' # 深灰色 - 连接线
    
    # 1. 标题
    ax.text(7, 7.5, '长江水位预测模型架构图', fontsize=16, fontweight='bold', ha='center', va='center')
    
    # 2. 输入层
    input_layer = FancyBboxPatch((1, 4.5), 2.5, 1.5, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor=input_color, 
                                 edgecolor='black', linewidth=2)
    ax.add_patch(input_layer)
    ax.text(2.25, 5.25, '输入层', fontsize=12, fontweight='bold', ha='center', va='center')
    ax.text(2.25, 4.9, '17维特征', fontsize=10, ha='center', va='center')
    
    # 输入特征说明
    ax.text(2.25, 4.3, '• 历史水位序列 (12维)', fontsize=9, ha='center')
    ax.text(2.25, 4.1, '• 潮位延迟特征 (3维)', fontsize=9, ha='center')
    ax.text(2.25, 3.9, '• 流量特征 (2维)', fontsize=9, ha='center')
    
    # 3. 隐藏层1
    hidden1 = FancyBboxPatch((4.5, 4.5), 2.5, 1.5, 
                             boxstyle="round,pad=0.1", 
                             facecolor=hidden_color, 
                             edgecolor='black', linewidth=2)
    ax.add_patch(hidden1)
    ax.text(5.75, 5.25, '隐藏层1', fontsize=12, fontweight='bold', ha='center', va='center')
    ax.text(5.75, 4.9, '64个神经元', fontsize=10, ha='center', va='center')
    
    # 隐藏层1说明
    ax.text(5.75, 4.3, '• 全连接层', fontsize=9, ha='center')
    ax.text(5.75, 4.1, '• ReLU激活', fontsize=9, ha='center')
    ax.text(5.75, 3.9, '• Dropout(0.1)', fontsize=9, ha='center')
    
    # 4. 隐藏层2
    hidden2 = FancyBboxPatch((8, 4.5), 2.5, 1.5, 
                             boxstyle="round,pad=0.1", 
                             facecolor=hidden_color, 
                             edgecolor='black', linewidth=2)
    ax.add_patch(hidden2)
    ax.text(9.25, 5.25, '隐藏层2', fontsize=12, fontweight='bold', ha='center', va='center')
    ax.text(9.25, 4.9, '32个神经元', fontsize=10, ha='center', va='center')
    
    # 隐藏层2说明
    ax.text(9.25, 4.3, '• 全连接层', fontsize=9, ha='center')
    ax.text(9.25, 4.1, '• ReLU激活', fontsize=9, ha='center')
    ax.text(9.25, 3.9, '• Dropout(0.1)', fontsize=9, ha='center')
    
    # 5. 输出层
    output_layer = FancyBboxPatch((11.5, 4.5), 2.5, 1.5, 
                                  boxstyle="round,pad=0.1", 
                                  facecolor=output_color, 
                                  edgecolor='black', linewidth=2)
    ax.add_patch(output_layer)
    ax.text(12.75, 5.25, '输出层', fontsize=12, fontweight='bold', ha='center', va='center')
    ax.text(12.75, 4.9, '3维预测', fontsize=10, ha='center', va='center')
    
    # 输出说明
    ax.text(12.75, 4.3, '• 1小时后水位', fontsize=9, ha='center')
    ax.text(12.75, 4.1, '• 2小时后水位', fontsize=9, ha='center')
    ax.text(12.75, 3.9, '• 3小时后水位', fontsize=9, ha='center')
    
    # 6. 连接箭头 - 横向连接
    # 输入层到隐藏层1
    ax.annotate('', xy=(4.5, 5.25), xytext=(3.5, 5.25),
                arrowprops=dict(arrowstyle='->', lw=2.5, color=connection_color))
    ax.text(4, 5.5, 'W₁ (64×17)', fontsize=10, ha='center', fontweight='bold')
    
    # 隐藏层1到隐藏层2
    ax.annotate('', xy=(8, 5.25), xytext=(7, 5.25),
                arrowprops=dict(arrowstyle='->', lw=2.5, color=connection_color))
    ax.text(7.5, 5.5, 'W₂ (32×64)', fontsize=10, ha='center', fontweight='bold')
    
    # 隐藏层2到输出层
    ax.annotate('', xy=(11.5, 5.25), xytext=(10.5, 5.25),
                arrowprops=dict(arrowstyle='->', lw=2.5, color=connection_color))
    ax.text(11, 5.5, 'W₃ (3×32)', fontsize=10, ha='center', fontweight='bold')
    
    # 7. 网络配置 - 放在下方
    config_box = FancyBboxPatch((1, 1.5), 12, 1.2, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#F1F8E9', 
                                edgecolor='black', linewidth=1.5)
    ax.add_patch(config_box)
    
    ax.text(7, 2.1, '网络配置', fontsize=12, fontweight='bold', ha='center')
    ax.text(3, 1.8, '优化器: Adam', fontsize=10, ha='center')
    ax.text(3, 1.6, '学习率: 0.001', fontsize=10, ha='center')
    ax.text(7, 1.8, '损失函数: MSE', fontsize=10, ha='center')
    ax.text(7, 1.6, '批次大小: 32', fontsize=10, ha='center')
    ax.text(11, 1.8, '激活函数: ReLU', fontsize=10, ha='center')
    ax.text(11, 1.6, '正则化: Dropout(0.1)', fontsize=10, ha='center')
    
    # 8. 添加图例
    legend_elements = [
        patches.Patch(color=input_color, label='输入层'),
        patches.Patch(color=hidden_color, label='隐藏层'),
        patches.Patch(color=output_color, label='输出层')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98), 
              fontsize=10, frameon=True, fancybox=True, shadow=True, title='图例')
    
    plt.tight_layout()
    return fig

def main():
    """主函数"""
    print("正在生成横向布局的长江水位预测模型架构图...")
    
    # 创建架构图
    fig = create_simple_architecture_diagram()
    
    # 保存图片
    output_path = '长江水位预测模型架构图_横向布局.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"横向布局架构图已保存到: {output_path}")
    
    # 显示图片
    plt.show()
    
    # 关闭图形以释放内存
    plt.close(fig)

if __name__ == '__main__':
    main() 
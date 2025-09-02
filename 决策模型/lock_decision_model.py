import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os
from datetime import datetime

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class LockDecisionModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.output_folder = None
        
    def load_data(self):
        """加载标记好的数据"""
        print("正在加载开通闸决策数据...")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_dir, '水位数据_小时_内河长江_带标记_含吨位.csv')
        
        self.df = pd.read_csv(data_path, parse_dates=['时间'])
        self.df.set_index('时间', inplace=True)
        
        print(f"数据加载完成，共 {len(self.df)} 条记录")
        print(f"时间范围: {self.df.index.min()} 到 {self.df.index.max()}")
        
        # 数据统计
        print(f"开通闸次数: {self.df['是否开通闸_小时'].sum()}")
        print(f"不开通闸次数: {len(self.df) - self.df['是否开通闸_小时'].sum()}")
        print(f"开通闸比例: {self.df['是否开通闸_小时'].mean():.2%}")
        
    def create_output_folder(self):
        """创建输出文件夹"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_folder = os.path.join(script_dir, '决策模型结果')
        os.makedirs(self.output_folder, exist_ok=True)
        print(f"输出文件夹: {self.output_folder}")
        
    def prepare_features(self):
        """准备特征数据"""
        print("\n=== 特征准备 ===")
        
        # 基本特征
        features = ['内河侧水位', '长江侧水位', '待过闸船舶吨位']
        X = self.df[features].copy()
        y = self.df['是否开通闸_小时'].copy()
        
        # 添加核心衍生特征
        X['水位差'] = X['内河侧水位'] - X['长江侧水位']
        X['水位比'] = X['内河侧水位'] / X['长江侧水位']
        X['总水位'] = X['内河侧水位'] + X['长江侧水位']
        
        # 添加未来一小时特征（与分析脚本保持一致，共10个特征）
        X['未来一小时内河水位'] = X['内河侧水位'].shift(-1)
        X['未来一小时长江水位'] = X['长江侧水位'].shift(-1)
        X['未来一小时水位差'] = X['未来一小时内河水位'] - X['未来一小时长江水位']
        X['未来一小时水位比'] = X['未来一小时内河水位'] / X['未来一小时长江水位']
        
        # 添加时间特征
        # X['月份'] = self.df.index.month  # 已移除月份特征
        
        # 删除包含NaN的行（主要是最后一行，因为无法获取未来一小时的数据）
        X = X.dropna()
        y = y[X.index]
        
        print(f"特征数量: {X.shape[1]}")
        print(f"特征列表: {list(X.columns)}")
        
        # 保存特征重要性分析
        feature_stats = X.describe()
        feature_stats.to_csv(os.path.join(self.output_folder, '特征统计信息.csv'), encoding='utf-8-sig')
        
        return X, y
        
    def train_model(self, X, y):
        """训练随机森林模型"""
        print("\n=== 模型训练 ===")
        
        # 数据分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"训练集大小: {len(X_train)}")
        print(f"测试集大小: {len(X_test)}")
        print(f"训练集开通闸比例: {y_train.mean():.2%}")
        print(f"测试集开通闸比例: {y_test.mean():.2%}")
        
        # 特征标准化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 使用默认参数训练模型
        print("正在训练随机森林模型...")
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train_scaled, y_train)
        print("模型训练完成")
        
        # 保存训练好的模型
        model_path = os.path.join(self.output_folder, 'lock_decision_model.pkl')
        joblib.dump(self.model, model_path)
        print(f"模型已保存到: {model_path}")
        
        # 保存标准化器，供后续预测保持一致
        scaler_path = os.path.join(self.output_folder, 'scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        print(f"标准化器已保存到: {scaler_path}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
        
    def evaluate_model(self, X_train_scaled, X_test_scaled, y_train, y_test):
        """评估模型性能"""
        print("\n=== 模型评估 ===")
        
        # 训练集预测
        y_train_pred = self.model.predict(X_train_scaled)
        y_train_pred_proba = self.model.predict_proba(X_train_scaled)[:, 1]
        
        # 测试集预测
        y_test_pred = self.model.predict(X_test_scaled)
        y_test_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # 计算评估指标
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        train_precision = precision_score(y_train, y_train_pred)
        test_precision = precision_score(y_test, y_test_pred)
        train_recall = recall_score(y_train, y_train_pred)
        test_recall = recall_score(y_test, y_test_pred)
        train_f1 = f1_score(y_train, y_train_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        train_auc = roc_auc_score(y_train, y_train_pred_proba)
        test_auc = roc_auc_score(y_test, y_test_pred_proba)
        
        print("训练集性能:")
        print(f"  准确率: {train_accuracy:.4f}")
        print(f"  精确率: {train_precision:.4f}")
        print(f"  召回率: {train_recall:.4f}")
        print(f"  F1分数: {train_f1:.4f}")
        print(f"  AUC: {train_auc:.4f}")
        
        print("\n测试集性能:")
        print(f"  准确率: {test_accuracy:.4f}")
        print(f"  精确率: {test_precision:.4f}")
        print(f"  召回率: {test_recall:.4f}")
        print(f"  F1分数: {test_f1:.4f}")
        print(f"  AUC: {test_auc:.4f}")
        
        # 保存评估结果
        evaluation_results = {
            '指标': ['准确率', '精确率', '召回率', 'F1分数', 'AUC'],
            '训练集': [train_accuracy, train_precision, train_recall, train_f1, train_auc],
            '测试集': [test_accuracy, test_precision, test_recall, test_f1, test_auc]
        }
        
        eval_df = pd.DataFrame(evaluation_results)
        eval_df.to_csv(os.path.join(self.output_folder, '模型评估结果.csv'), index=False, encoding='utf-8-sig')
        
        return y_test_pred, y_test_pred_proba
        
    def analyze_feature_importance(self, X):
        """分析特征重要性"""
        print("\n=== 特征重要性分析 ===")
        
        feature_importance = pd.DataFrame({
            '特征': X.columns,
            '重要性': self.model.feature_importances_
        }).sort_values('重要性', ascending=False)
        
        print("特征重要性排序:")
        for i, row in feature_importance.iterrows():
            print(f"  {row['特征']}: {row['重要性']:.4f}")
        
        # 保存特征重要性
        feature_importance.to_csv(os.path.join(self.output_folder, '特征重要性.csv'), index=False, encoding='utf-8-sig')
        
        # 绘制特征重要性图
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.barh(range(len(feature_importance)), feature_importance['重要性'])
        plt.yticks(range(len(feature_importance)), feature_importance['特征'])
        plt.xlabel('重要性')
        plt.title('特征重要性排序')
        plt.gca().invert_yaxis()
        
        return feature_importance
        
    def generate_visualizations(self, X, y, y_test_pred, y_test_pred_proba, y_test):
        """生成可视化图表"""
        print("\n=== 生成可视化图表 ===")
        
        # 特征重要性图
        feature_importance = self.analyze_feature_importance(X)
        
        # 混淆矩阵
        plt.subplot(2, 2, 2)
        cm = confusion_matrix(y_test, y_test_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('混淆矩阵')
        plt.ylabel('真实值')
        plt.xlabel('预测值')
        
        # 预测概率分布
        plt.subplot(2, 2, 3)
        plt.hist(y_test_pred_proba[y_test == 0], bins=30, alpha=0.7, label='不开通闸', density=True)
        plt.hist(y_test_pred_proba[y_test == 1], bins=30, alpha=0.7, label='开通闸', density=True)
        plt.xlabel('预测概率')
        plt.ylabel('密度')
        plt.title('预测概率分布')
        plt.legend()
        
        # 特征与开通闸关系
        plt.subplot(2, 2, 4)
        top_features = feature_importance.head(5)['特征'].values
        for feature in top_features[:3]:  # 只显示前3个特征
            plt.scatter(X[feature], y, alpha=0.5, label=feature)
        plt.xlabel('特征值')
        plt.ylabel('是否开通闸')
        plt.title('主要特征与开通闸关系')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, '模型分析图表.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 生成详细报告
        self.generate_report(X, y, y_test_pred, y_test_pred_proba, y_test)
        
    def generate_report(self, X, y, y_test_pred, y_test_pred_proba, y_test):
        """生成详细报告"""
        print("\n=== 生成分析报告 ===")
        
        # 计算特征重要性
        feature_importance = self.analyze_feature_importance(X)
        
        # 计算评估指标
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred)
        test_recall = recall_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        test_auc = roc_auc_score(y_test, y_test_pred_proba)
        
        report = f"""开通闸决策模型分析报告
============================================================

📊 模型类型: 随机森林分类器
📅 分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📈 数据概况:
  总数据量: {len(self.df):,} 条记录
  时间范围: {self.df.index.min().strftime('%Y-%m-%d')} 到 {self.df.index.max().strftime('%Y-%m-%d')}
  开通闸次数: {self.df['是否开通闸_小时'].sum():,}
  不开通闸次数: {len(self.df) - self.df['是否开通闸_小时'].sum():,}
  开通闸比例: {self.df['是否开通闸_小时'].mean():.2%}

🏗️ 模型配置:
  特征数量: {X.shape[1]}
  主要特征: {', '.join(feature_importance.head(5)['特征'].values)}
  最佳参数: {self.model.get_params()}

🎯 模型性能:
  测试集准确率: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)
  测试集精确率: {test_precision:.4f} ({test_precision*100:.2f}%)
  测试集召回率: {test_recall:.4f} ({test_recall*100:.2f}%)
  测试集F1分数: {test_f1:.4f} ({test_f1*100:.2f}%)
  测试集AUC: {test_auc:.4f} ({test_auc*100:.2f}%)

🔍 特征重要性 (前10):
"""
        
        for i, row in feature_importance.head(10).iterrows():
            report += f"   {i+1}. {row['特征']}: {row['重要性']:.4f}\n"
        
        report += f"""
📋 模型评估:
  • 准确率: 模型预测正确的比例
  • 精确率: 预测为开通闸中实际开通闸的比例
  • 召回率: 实际开通闸中被正确预测的比例
  • F1分数: 精确率和召回率的调和平均
  • AUC: ROC曲线下面积，越接近1越好

✅ 结论:
  • 模型性能{'优秀' if test_accuracy > 0.9 else '良好' if test_accuracy > 0.8 else '一般'} (准确率 {test_accuracy*100:.1f}%)
  • 特征重要性分析显示主要影响因素
  • 可用于实际开通闸决策支持

📁 生成的文件:
  • 模型分析图表.png - 可视化分析
  • 特征重要性.csv - 特征重要性排序
  • 模型评估结果.csv - 详细评估指标
  • 特征统计信息.csv - 特征统计信息
  • lock_decision_model.pkl - 训练好的模型
  • 决策模型报告.txt - 本报告
"""
        
        with open(os.path.join(self.output_folder, '决策模型报告.txt'), 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("分析报告已生成完成！")
        
    def run_analysis(self):
        """运行完整分析"""
        self.load_data()
        self.create_output_folder()
        X, y = self.prepare_features()
        X_train_scaled, X_test_scaled, y_train, y_test = self.train_model(X, y)
        y_test_pred, y_test_pred_proba = self.evaluate_model(X_train_scaled, X_test_scaled, y_train, y_test)
        self.generate_visualizations(X, y, y_test_pred, y_test_pred_proba, y_test)
        print(f"\n分析完成！所有结果已保存到: {self.output_folder}")

if __name__ == "__main__":
    model = LockDecisionModel()
    model.run_analysis() 
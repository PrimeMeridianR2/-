import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 核极限学习机实现
class KELM:
    def __init__(self, C=1.0, kernel='rbf', gamma=1.0):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.X_train = None
        self.beta = None

    def _kernel_matrix(self, X, Y):
        if self.kernel == 'rbf':
            pairwise_dists = cdist(X, Y, 'euclidean')
            return np.exp(-self.gamma * pairwise_dists **2)
        else:
            raise ValueError(f"不支持的核函数类型: {self.kernel}")

    def fit(self, X, y):
        self.X_train = X
        n_samples = X.shape[0]
        K = self._kernel_matrix(X, X)
        y = y.values.reshape(-1, 1) if isinstance(y, pd.Series) else y.reshape(-1, 1)
        I = np.eye(n_samples)
        self.beta = np.linalg.inv(K + I / self.C) @ y

    def predict(self, X):
        if self.X_train is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        if X.ndim == 1:
            X = X.reshape(1, -1)
        K_test = self._kernel_matrix(X, self.X_train)
        return (K_test @ self.beta).flatten()


# 蚁群优化算法（加速版）
class ACOOptimizer:
    def __init__(self, n_ants=20, max_iter=10, rho=0.8, Q=2,  # 减少蚂蚁数量和迭代次数
                 C_range=(0.1, 100), gamma_range=(0.01, 10), grid_size=15):  # 缩小网格
        self.n_ants = n_ants
        self.max_iter = max_iter
        self.rho = rho
        self.Q = Q
        self.C_range = C_range
        self.gamma_range = gamma_range
        self.grid_size = grid_size
        self.pheromone = np.ones((grid_size, grid_size))
        self.C_values = np.linspace(C_range[0], C_range[1], grid_size)
        self.gamma_values = np.linspace(gamma_range[0], gamma_range[1], grid_size)
        self.best_solution = (self.C_values[0], self.gamma_values[0])
        self.best_fitness = float('inf')
        self.iteration_best_fitness = []

    def _roulette_wheel_selection(self, probabilities):
        if np.sum(probabilities) == 0:
            probabilities = np.ones_like(probabilities) / len(probabilities)
        cumulative_prob = np.cumsum(probabilities)
        r = np.random.rand()
        for i, prob in enumerate(cumulative_prob):
            if r <= prob:
                return i
        return len(probabilities) - 1

    def _evaluate_fitness(self, C_idx, gamma_idx, X_train, y_train, X_val, y_val):
        try:
            C = self.C_values[C_idx]
            gamma = self.gamma_values[gamma_idx]
            kelm = KELM(C=C, gamma=gamma)
            kelm.fit(X_train, y_train)
            y_pred = kelm.predict(X_val)
            return mean_squared_error(y_val, y_pred)
        except Exception as e:
            print(f"参数评估失败: {str(e)}")
            return float('inf')

    def optimize(self, X_train, y_train):
        try:
            X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
            for iter in range(self.max_iter):
                solutions = []
                fitness_values = []
                for ant in range(self.n_ants):
                    C_prob = self.pheromone.sum(axis=1) / self.pheromone.sum()
                    C_idx = self._roulette_wheel_selection(C_prob)
                    gamma_prob = self.pheromone[C_idx, :] / self.pheromone[C_idx, :].sum()
                    gamma_idx = self._roulette_wheel_selection(gamma_prob)
                    fitness = self._evaluate_fitness(C_idx, gamma_idx, X_tr, y_tr, X_val, y_val)
                    solutions.append((C_idx, gamma_idx))
                    fitness_values.append(fitness)
                    if fitness < self.best_fitness:
                        self.best_fitness = fitness
                        self.best_solution = (self.C_values[C_idx], self.gamma_values[gamma_idx])
                self.pheromone *= self.rho
                if fitness_values:
                    best_idx = np.argmin(fitness_values)
                    best_C_idx, best_gamma_idx = solutions[best_idx]
                    self.pheromone[best_C_idx, best_gamma_idx] += self.Q / (fitness_values[best_idx] + 1e-10)
                current_best = np.min(fitness_values) if fitness_values else float('inf')
                self.iteration_best_fitness.append(current_best)
                print(f"ACO迭代 {iter + 1}/{self.max_iter}, 本轮最佳MSE: {current_best:.6f}")
            return self.best_solution
        except Exception as e:
            print(f"蚁群优化失败: {str(e)}")
            return self.best_solution


# 主函数（加速版）
def train_and_ensemble_models():
    # 1. 检查文件路径
    file_path = r'd_train2.csv'
    if not os.path.exists(file_path):
        print(f"错误：文件不存在！路径：{os.path.abspath(file_path)}")
        return {}, {}, {}, {}

    # 2. 加载数据
    try:
        encodings = ['utf-8', 'gbk', 'gb2312']
        data = None
        for encoding in encodings:
            try:
                data = pd.read_csv(file_path, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        if data is None:
            print(f"错误：文件编码不支持（已尝试{encodings}）")
            return {}, {}, {}, {}
    except Exception as e:
        print(f"加载失败：{str(e)}")
        return {}, {}, {}, {}

    # 3. 数据预处理（含分类特征处理）
    try:
        # 区分数值和分类特征
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(exclude=[np.number]).columns

        # 填充缺失值
        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
        for col in categorical_cols:
            data[col] = data[col].fillna(data[col].mode()[0])

        # 独热编码处理分类特征
        if len(categorical_cols) > 0:
            data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

        # 划分特征和目标（目标列名根据实际修改）
        target_col = '血糖'  # 替换为你的目标列名
        if target_col not in data.columns:
            print(f"错误：缺少目标列 '{target_col}'")
            return {}, {}, {}, {}
        X = data.drop(columns=[target_col])
        y = data[target_col]

        # 减少样本量（可选，进一步加速）
        sample_ratio = 0.8  # 使用80%的样本
        if len(X) > 1000:  # 仅当样本数较多时才采样
            X = X.sample(frac=sample_ratio, random_state=42)
            y = y.loc[X.index]

        if X.shape[1] == 0:
            print("错误：特征列为空")
            return {}, {}, {}, {}

        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    except Exception as e:
        print(f"预处理失败：{str(e)}")
        return {}, {}, {}, {}

    # 4. 训练模型（加速参数）
    models, predictions, metrics = {}, {}, {}

    # 随机森林（减少树数量）
    try:
        print("\n训练随机森林...")
        rf_params = {
            'n_estimators': 200,  # 大幅减少树数量（原1000）
            'max_depth': 5,       # 限制树深度
            'max_features': 'sqrt',  # 更快的特征选择
            'random_state': 42,
            'n_jobs': -1          # 并行计算
        }
        rf = RandomForestRegressor(** rf_params)
        rf.fit(X_train, y_train)
        pred = rf.predict(X_test)
        metrics['Random Forest'] = {
            'MSE': mean_squared_error(y_test, pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, pred)),
            'MAE': mean_absolute_error(y_test, pred),
            'MAPE': mean_absolute_percentage_error(y_test, pred),
            'R2': r2_score(y_test, pred)
        }
        models['Random Forest'] = rf
        predictions['Random Forest'] = pred
    except Exception as e:
        print(f"随机森林失败：{str(e)}")

    # XGBoost（减少迭代和深度）
    try:
        print("\n训练XGBoost...")
        xgb_params = {
            'n_estimators': 300,  # 减少迭代（原1000）
            'max_depth': 6,       # 降低树深度（原12）
            'subsample': 0.8,     # 样本采样
            'colsample_bytree': 0.8,  # 特征采样
            'random_state': 42,
            'n_jobs': -1
        }
        xgb = XGBRegressor(**xgb_params)
        xgb.fit(X_train, y_train)
        pred = xgb.predict(X_test)
        metrics['XGBoost'] = {
            'MSE': mean_squared_error(y_test, pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, pred)),
            'MAE': mean_absolute_error(y_test, pred),
            'MAPE': mean_absolute_percentage_error(y_test, pred),
            'R2': r2_score(y_test, pred)
        }
        models['XGBoost'] = xgb
        predictions['XGBoost'] = pred
    except Exception as e:
        print(f"XGBoost失败：{str(e)}")

    # ACO-KELM（减少优化迭代）
    try:
        print("\n训练ACO-KELM...")
        aco = ACOOptimizer(
            n_ants=20,    # 减少蚂蚁数量（原50）
            max_iter=10   # 减少迭代次数（原25）
        )
        best_C, best_gamma = aco.optimize(X_train, y_train)
        kelm = KELM(C=best_C, gamma=best_gamma)
        kelm.fit(X_train, y_train)
        pred = kelm.predict(X_test)
        metrics['ACO-KELM'] = {
            'MSE': mean_squared_error(y_test, pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, pred)),
            'MAE': mean_absolute_error(y_test, pred),
            'MAPE': mean_absolute_percentage_error(y_test, pred),
            'R2': r2_score(y_test, pred)
        }
        models['ACO-KELM'] = kelm
        predictions['ACO-KELM'] = pred
    except Exception as e:
        print(f"ACO-KELM失败：{str(e)}")

    # 模型融合
    weights = {}
    try:
        if len(predictions) >= 2:
            rmse = {k: v['RMSE'] for k, v in metrics.items() if 'RMSE' in v and v['RMSE'] > 0}
            if rmse:
                total = sum(1 / r for r in rmse.values())
                weights = {k: (1 / r) / total for k, r in rmse.items()}
                ensemble_pred = np.zeros_like(y_test)
                for k, w in weights.items():
                    ensemble_pred += w * predictions[k]
                metrics['Ensemble'] = {
                    'MSE': mean_squared_error(y_test, ensemble_pred),
                    'RMSE': np.sqrt(mean_squared_error(y_test, ensemble_pred)),
                    'MAE': mean_absolute_error(y_test, ensemble_pred),
                    'MAPE': mean_absolute_percentage_error(y_test, ensemble_pred),
                    'R2': r2_score(y_test, ensemble_pred)
                }
    except Exception as e:
        print(f"融合失败：{str(e)}")

    # 5. 简化可视化（只保留关键图）
    try:
        if metrics:
            # 只画指标对比图（减少画图耗时）
            metrics_df = pd.DataFrame({
                'Model': list(metrics.keys()),
                'MSE': [m['MSE'] for m in metrics.values()],
                'RMSE': [m['RMSE'] for m in metrics.values()],
                'R²': [m['R2'] for m in metrics.values()]
            })
            melted = pd.melt(metrics_df, id_vars=['Model'], var_name='Metric', value_name='Value')
            plt.figure(figsize=(12, 6))
            sns.barplot(x='Model', y='Value', hue='Metric', data=melted)
            plt.title('模型关键指标对比')
            plt.tight_layout()
            plt.show()
    except Exception as e:
        print(f"可视化失败：{str(e)}")

    return models, predictions, metrics, weights


if __name__ == "__main__":
    models, predictions, metrics, weights = train_and_ensemble_models()
    if metrics:
        print("\n最终评估结果:")
        for name, m in metrics.items():
            print(f"\n{name}:")
            for k, v in m.items():
                print(f"  {k}: {v:.4f}" if k != 'MAPE' else f"  {k}: {v:.4%}")
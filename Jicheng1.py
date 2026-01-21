import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import seaborn as sns
import chardet

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 数据加载
with open('d_train.csv', 'rb') as f:
    result = chardet.detect(f.read(10000))
print(f"检测到的编码格式: {result['encoding']}")
print(f"检测可信度: {result['confidence']}")

try:
    train_data = pd.read_csv('d_train.csv', encoding=result['encoding'])
    print(train_data.head())
except UnicodeDecodeError:
    print(f"使用检测到的编码格式 {result['encoding']} 读取文件失败")

train_data = pd.read_csv('d_train.csv', encoding='gb2312')
test_A_data = pd.read_csv('d_test_A.csv', encoding='gb2312')
test_B_data = pd.read_csv('d_test_B.csv', encoding='gb2312')
answer_A = pd.read_csv('d_answer_a.csv', encoding='gb2312')
answer_B = pd.read_csv('d_answer_b.csv', encoding='gb2312')

# 数据预处理
test_A_combined = pd.concat([test_A_data, answer_A.iloc[:, -1]], axis=1)
test_B_combined = pd.concat([test_B_data, answer_B.iloc[:, -1]], axis=1)
all_data = pd.concat([train_data, test_A_combined, test_B_combined], ignore_index=True)
X = all_data.iloc[:, :-1]
y = all_data.iloc[:, -1]

# 缺失值处理
numeric_cols = X.select_dtypes(include='number').columns
categorical_cols = X.select_dtypes(exclude='number').columns
if not numeric_cols.empty:
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())
if not categorical_cols.empty:
    X[categorical_cols] = X[categorical_cols].fillna(X[categorical_cols].mode().iloc[0])

# 处理非数值特征与缺失值
valid_indices = y.notna()
X, y = X[valid_indices], y[valid_indices]

for col in X.select_dtypes(exclude=['number']).columns:
    if X[col].nunique() < 10:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    else:
        X = X.drop(col, axis=1)
X = X.fillna(0)

# 特征选择（保留15个重要特征）
rf_selector = RandomForestRegressor(n_estimators=50, random_state=42)
rf_selector.fit(X, y)
selected_features = pd.Series(rf_selector.feature_importances_, index=X.columns).sort_values(ascending=False).head(15).index
X_selected = X[selected_features]

# 特征相关性热图
corr_matrix = X_selected.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
plt.title('特征相关性热图')
plt.savefig('feature_correlation.png')
plt.close()

# 数据分割
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# LightGBM模型（贝叶斯优化）
def lgb_objective(params):
    model = lgb.LGBMRegressor(
        max_depth=int(params['max_depth']),
        num_leaves=int(params['num_leaves']),
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        reg_alpha=params['reg_alpha'],
        reg_lambda=params['reg_lambda'],
        learning_rate=0.05, n_estimators=100, random_state=42
    )
    model.fit(X_train, y_train)
    return {'loss': mean_squared_error(y_test, model.predict(X_test)), 'status': STATUS_OK}

lgb_best = fmin(
    fn=lgb_objective,
    space={'max_depth': hp.choice('max_depth', [3, 5]),
           'num_leaves': hp.choice('num_leaves', [5, 14, 28]),
           'subsample': hp.choice('subsample', [0.8, 1.0]),
           'colsample_bytree': hp.choice('colsample_bytree', [0.8, 1.0]),
           'reg_alpha': hp.choice('reg_alpha', [0, 0.5]),
           'reg_lambda': hp.choice('reg_lambda', [0, 0.5])},
    algo=tpe.suggest, max_evals=20, trials=Trials(), rstate=np.random.default_rng(42)
)

lgb_model = lgb.LGBMRegressor(
    max_depth=[3, 5][lgb_best['max_depth']],
    num_leaves=[5, 14, 28][lgb_best['num_leaves']],
    subsample=[0.8, 1.0][lgb_best['subsample']],
    colsample_bytree=[0.8, 1.0][lgb_best['colsample_bytree']],
    reg_alpha=[0, 0.5][lgb_best['reg_alpha']],
    reg_lambda=[0, 0.5][lgb_best['reg_lambda']],
    learning_rate=0.05, n_estimators=100, random_state=42
).fit(X_train, y_train)
print("LightGBM 回归模型训练完成")

# XGBoost模型（贝叶斯优化）
def xgb_objective(params):
    model = xgb.XGBRegressor(
        max_depth=int(params['max_depth']),
        learning_rate=params['learning_rate'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        reg_alpha=params['reg_alpha'],
        reg_lambda=params['reg_lambda'],
        n_estimators=100, random_state=42
    )
    model.fit(X_train, y_train)
    return {'loss': mean_squared_error(y_test, model.predict(X_test)), 'status': STATUS_OK}

xgb_best = fmin(
    fn=xgb_objective,
    space={'max_depth': hp.choice('max_depth', [3, 5, 7]),
           'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
           'subsample': hp.uniform('subsample', 0.6, 1.0),
           'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
           'reg_alpha': hp.uniform('reg_alpha', 0, 1.0),
           'reg_lambda': hp.uniform('reg_lambda', 0, 1.0)},
    algo=tpe.suggest, max_evals=20, trials=Trials(), rstate=np.random.default_rng(42)
)

xgb_model = xgb.XGBRegressor(
    max_depth=[3, 5, 7][xgb_best['max_depth']],
    learning_rate=xgb_best['learning_rate'],
    subsample=xgb_best['subsample'],
    colsample_bytree=xgb_best['colsample_bytree'],
    reg_alpha=xgb_best['reg_alpha'],
    reg_lambda=xgb_best['reg_lambda'],
    n_estimators=100, random_state=42
).fit(X_train, y_train)
print("XGBoost 回归模型训练完成")

# CatBoost模型（贝叶斯优化）
def cat_objective(params):
    model = cb.CatBoostRegressor(
        depth=int(params['depth']),
        learning_rate=params['learning_rate'],
        random_strength=params['random_strength'],
        l2_leaf_reg=params['l2_leaf_reg'],
        subsample=params['subsample'],
        iterations=200, random_state=42, verbose=0
    )
    model.fit(X_train, y_train)
    return {'loss': mean_squared_error(y_test, model.predict(X_test)), 'status': STATUS_OK}

cat_best = fmin(
    fn=cat_objective,
    space={'depth': hp.choice('depth', [4, 6, 8]),
           'learning_rate': hp.uniform('learning_rate', 0.01, 0.1),
           'random_strength': hp.uniform('random_strength', 0.01, 0.5),
           'l2_leaf_reg': hp.uniform('l2_leaf_reg', 1, 10),
           'subsample': hp.uniform('subsample', 0.6, 1.0)},
    algo=tpe.suggest, max_evals=20, trials=Trials(), rstate=np.random.default_rng(42)
)

cat_model = cb.CatBoostRegressor(
    depth=[4, 6, 8][cat_best['depth']],
    learning_rate=cat_best['learning_rate'],
    random_strength=cat_best['random_strength'],
    l2_leaf_reg=cat_best['l2_leaf_reg'],
    subsample=cat_best['subsample'],
    iterations=200, random_state=42, verbose=0
).fit(X_train, y_train)
print("CatBoost 回归模型训练完成")

# Stacking集成
def get_stacking_features(model, X_train, y_train, X_test, n_folds=3):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    train_feats, test_feats = np.zeros(X_train.shape[0]), np.zeros(X_test.shape[0])
    for train_idx, val_idx in kf.split(X_train):
        model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        train_feats[val_idx] = model.predict(X_train.iloc[val_idx])
        test_feats += model.predict(X_test) / n_folds
    return train_feats.reshape(-1, 1), test_feats.reshape(-1, 1)

valid_models = [m for m in [lgb_model, xgb_model, cat_model] if m is not None]
stack_train, stack_test = zip(*[get_stacking_features(m, X_train, y_train, X_test) for m in valid_models])
X_stack_train, X_stack_test = np.hstack(stack_train), np.hstack(stack_test)

meta_model = LinearRegression().fit(X_stack_train, y_train)
print("Stacking集成模型训练完成")

# 模型评估
def evaluate_model(model, X, y, name):
    y_pred = model.predict(X)
    print(f"\n{name} 指标: MSE={mean_squared_error(y, y_pred):.4f}, R2={r2_score(y, y_pred):.4f}")

print("\n模型评估结果:")
evaluate_model(lgb_model, X_test, y_test, "LightGBM")
evaluate_model(xgb_model, X_test, y_test, "XGBoost")
evaluate_model(cat_model, X_test, y_test, "CatBoost")
evaluate_model(meta_model, X_stack_test, y_test, "Stacking集成")
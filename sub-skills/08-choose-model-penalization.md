# Sub-Skill 08: 选择建模方法与惩罚策略

> **所属 Meta-Skill:** [临床预测模型开发指南](../clinical-prediction-model-development-zh.md)  
> **上一步:** [Sub-Skill 07: 处理缺失数据](07-handle-missing-data.md)  
> **下一步:** [Sub-Skill 09: 拟合预测模型](09-fit-prediction-model.md)

---

## 核心任务

选择适合你的数据特征、样本量和结局类型的建模方法，并使用惩罚（正则化）控制过拟合。

---

## 方法选择决策树

```
样本量充足？
├── 是 → 结局类型？
│   ├── 连续 → OLS + 样条 / GAM / 岭回归
│   ├── 二分类 → Logistic + 样条 / 岭 / LASSO / Elastic Net
│   ├── 时间-事件 → Cox + 惩罚 / 灵活参数模型
│   └── 竞争风险 → Fine-Gray / 特定原因 Cox
└── 否 → 必须使用强惩罚（岭回归优先）
    → 减少参数数量
    → 考虑更简单的模型结构

需要自动变量选择？
├── 否 → 岭回归（保留所有变量，收缩系数）
└── 是 → LASSO（产生稀疏解）或 Elastic Net（折中）

连续预测变量的非线性关系？
├── 是 → GAM (pygam) 或 限制性立方样条 (patsy + statsmodels)
└── 否 → 线性项即可
```

---

## 惩罚方法详解

### 岭回归 (Ridge / L2)

- **惩罚项:** λ × Σ(β²)
- **效果:** 收缩所有系数，但不归零
- **适用:** 所有预测变量都有贡献，不需要变量选择
- **优点:** 稳定，尤其小样本 / 高共线性时

### LASSO (L1)

- **惩罚项:** λ × Σ|β|
- **效果:** 可将部分系数压缩至零（自动变量选择）
- **适用:** 假设许多预测变量无贡献
- **注意:** 小样本中不稳定，可能随机选择变量

### Elastic Net

- **惩罚项:** α × L1 + (1-α) × L2
- **效果:** 折中方案，兼顾变量选择和稳定性
- **调参:** 交叉验证选择 λ 和 α

---

## Python 代码

### 1. 限制性立方样条（statsmodels + patsy）

```python
import statsmodels.api as sm
from patsy import dmatrices

# 用 dmatrices 同时生成 y 和设计矩阵 X
formula = "y ~ cr(x1, df=3) + cr(x2, df=3) + C(x3) + C(x4) + C(x5)"
y, X = dmatrices(formula, data=data, return_type='dataframe')

# 连续型结局 + 样条
model = sm.OLS(y, X).fit()

# 二分类结局 + 样条（加 L1 惩罚稳定收敛）
model = sm.Logit(y, X).fit_regularized(method='l1', alpha=0.01)
```

### 2. GAM（pygam 包）

```python
from pygam import LinearGAM, LogisticGAM, s

# 连续型结局 GAM
gam = LinearGAM(s(0) + s(1) + f(2) + f(3) + f(4))
gam.fit(X, y)

# 二分类结局 GAM
gam = LogisticGAM(s(0) + s(1) + f(2) + f(3) + f(4))
gam.fit(X, y)

# 自动搜索最佳 λ（平滑参数）
gam.gridsearch(X, y)
print(gam.summary())
```

### 3. 岭回归 / LASSO / Elastic Net（sklearn）

```python
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.preprocessing import SplineTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 定义预处理：连续变量做样条，分类变量做 One-Hot
preprocessor = ColumnTransformer([
    ('spline', SplineTransformer(n_knots=3, degree=3), ['x1', 'x2']),
    ('cat', OneHotEncoder(drop='first'), ['x3', 'x4', 'x5'])
])

# ========== 岭回归 (alpha=0) ==========
pipe_ridge = Pipeline([
    ('prep', preprocessor),
    ('model', RidgeCV(alphas=np.logspace(-3, 3, 100), cv=10))
])
pipe_ridge.fit(X, y)
print(f"最优 alpha: {pipe_ridge.named_steps['model'].alpha_}")

# ========== LASSO (alpha=1) ==========
pipe_lasso = Pipeline([
    ('prep', preprocessor),
    ('model', LassoCV(alphas=np.logspace(-3, 3, 100), cv=10, max_iter=5000))
])
pipe_lasso.fit(X, y)

# ========== Elastic Net ==========
pipe_en = Pipeline([
    ('prep', preprocessor),
    ('model', ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1],
        cv=10, max_iter=5000
    ))
])
pipe_en.fit(X, y)
```

---

## 模型复杂度控制

| 策略 | 适用场景 |
|------|----------|
| 减少预测变量 | 小样本 |
| 使用样条（而非高阶多项式） | 捕捉非线性 |
| 岭回归收缩 | 高共线性、所有变量都有贡献 |
| LASSO 稀疏化 | 高维数据、变量筛选 |
| 交叉验证选 λ | 所有惩罚模型 |

---

## 质量检查清单

- [ ] 复杂模型使用了惩罚/收缩方法
- [ ] 模型复杂度与样本量匹配（偏差-方差权衡）
- [ ] 已探索连续型预测变量的函数形式（样条，而非分类）
- [ ] 惩罚参数（λ, α）通过交叉验证选择
- [ ] 未使用逐步回归或单变量 P 值筛选

---

## 返回 Router

← [上一步: Sub-Skill 07](07-handle-missing-data.md)  
← [返回主指南](../clinical-prediction-model-development-zh.md)  
→ [进入下一步: Sub-Skill 09](09-fit-prediction-model.md)

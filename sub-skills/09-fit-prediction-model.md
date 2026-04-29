# Sub-Skill 09: 拟合预测模型

> **所属 Meta-Skill:** [临床预测模型开发指南](../clinical-prediction-model-development-zh.md)  
> **上一步:** [Sub-Skill 08: 选择建模方法](08-choose-model-penalization.md)  
> **下一步:** [Sub-Skill 10: 评估区分度](10-assess-discrimination.md)

---

## 核心任务

使用选定的方法开发模型，确保所有预先指定的预测变量被正确处理，并在多重插补场景下正确合并结果。

---

## 基本流程

### 1. 数据准备

```python
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 确保变量类型正确
data['x3'] = data['x3'].astype('category')  # 二分类变量
data['x5'] = data['x5'].astype('category')  # 多分类变量

# 检查共线性 (VIF)
X_numeric = data[['x1', 'x2', 'x3', 'x4', 'x5']].select_dtypes(include=[np.number])
vif_data = pd.DataFrame()
vif_data["Variable"] = X_numeric.columns
vif_data["VIF"] = [variance_inflation_factor(X_numeric.values, i) 
                   for i in range(X_numeric.shape[1])]
print(vif_data)  # VIF 应 < 5-10
```

### 2. 模型拟合

#### 场景 A：无缺失数据

```python
import statsmodels.api as sm
from patsy import dmatrices

# dmatrices 同时处理公式左侧的结局变量和右侧的预测变量
formula = "y ~ cr(x1, df=3) + cr(x2, df=3) + C(x3) + C(x4) + C(x5)"
y, X = dmatrices(formula, data=data, return_type='dataframe')
model = sm.Logit(y, X).fit_regularized(method='l1', alpha=0.01)
```

#### 场景 B：多重插补后

```python
# 在每个插补数据集中分别拟合模型
models = []
for i in range(n_impute):
    # 插补后的数据应无缺失值
    y_imp, X_imp = dmatrices(formula, data=imputed[i], return_type='dataframe')
    model_i = sm.Logit(y_imp, X_imp).fit_regularized(
        method='l1', alpha=0.01, disp=False
    )
    models.append(model_i)
```

---

## 多重插补下的合并策略

### 策略 A：合并系数（简单模型）

```python
import numpy as np

# 从各插补模型中提取系数并取平均
coefs = np.array([m.params.values for m in models])
pooled_coef = np.mean(coefs, axis=0)
pooled_se = np.sqrt(np.mean(coefs.var(axis=0)) + (1 + 1/n_impute) * coefs.var(axis=0).mean())
# 注：完整 Rubin 法则需更多步骤，推荐策略 B
```

### 策略 B：合并预测值（推荐，更通用）

```python
import numpy as np
from patsy import dmatrices

def predict_merged(imputed_datasets, models):
    """
    对各插补模型，在对应插补后的数据上预测，然后取平均。
    注意：必须用插补后的数据（无缺失值），不能用含 NaN 的原始数据。
    """
    predictions = []
    for i, model in enumerate(models):
        # 使用该模型对应的插补数据集
        df = imputed_datasets[i]
        _, X_new = dmatrices(model.model.formula, data=df, return_type='dataframe')
        # 对齐列顺序
        train_cols = model.model.exog_names
        for c in train_cols:
            if c not in X_new.columns:
                X_new[c] = 0
        X_new = X_new[train_cols]
        pred = model.predict(X_new)
        predictions.append(pred)
    return np.mean(np.column_stack(predictions), axis=1)

# 使用
final_prediction = predict_merged(imputed_datasets, models)
```

**为什么策略 B 更优？**
- 适用于复杂模型（如 ML、GAM）
- 当各插补数据集的模型结构不同时（如 LASSO 选了不同变量）
- 不依赖系数的线性组合假设

---

## 模型公式检查清单

在运行模型前，逐项确认：

| 检查项 | 说明 |
|--------|------|
| 预测变量集合 | 与预先指定的集合一致 |
| 连续变量处理 | 使用样条（rcs/s）而非分类 |
| 分类变量编码 | 参考水平合理 |
| 交互项 | 仅纳入有先验假设的交互 |
| 惩罚参数 | 已通过交叉验证选定 |
| 模型复杂度 | 不超过样本量允许的自由度 |

---

## 质量检查清单

- [ ] 模型公式与预先指定的预测变量集一致
- [ ] 连续变量使用样条而非分类
- [ ] 若使用 MI：已在每个插补数据集中拟合模型
- [ ] 预测值已正确合并（策略 B 推荐）
- [ ] 检查了共线性（VIF）
- [ ] 模型残差已初步检查

---

## 返回 Router

← [上一步: Sub-Skill 08](08-choose-model-penalization.md)  
← [返回主指南](../clinical-prediction-model-development-zh.md)  
→ [进入下一步: Sub-Skill 10](10-assess-discrimination.md)

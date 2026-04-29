# Sub-Skill 10: 评估区分度

> **所属 Meta-Skill:** [临床预测模型开发指南](../clinical-prediction-model-development-zh.md)  
> **上一步:** [Sub-Skill 09: 拟合预测模型](09-fit-prediction-model.md)  
> **下一步:** [Sub-Skill 11: 评估校准度](11-assess-calibration.md)

---

## 核心任务

量化模型区分不同结局患者的能力，并报告乐观校正的估计值。

---

## 区分度指标

| 结局类型 | 指标 | 范围 | 解释 |
|----------|------|------|------|
| **二分类** | C-statistic / AUC | 0.5-1.0 | 0.5=随机，1.0=完美；≥0.7 可接受，≥0.8 良好 |
| **时间-事件** | Time-dependent AUC / C-index | 0.5-1.0 | 同上，但考虑时间维度 |
| **连续型** | R-squared | 0-1 | 解释方差比例；需结合领域判断 |

---

## 重要原则

### 必须报告乐观校正值

**表观性能（Apparent Performance）**——在训练数据上评估——总是过于乐观。

**必须校正：**
- Bootstrap 乐观校正
- 或 K-fold 交叉验证

**禁止仅报告表观性能。**

---

## Python 代码

### 二分类结局

```python
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 表观 AUC（过于乐观，仅作参考）
auc_apparent = roc_auc_score(y_true, predicted_prob)
print(f"表观 AUC: {auc_apparent:.3f}")

# Bootstrap 校正（推荐）
# 注意：若数据有缺失值，需先用插补后的数据
np.random.seed(42)
n_boot = 500
optimism = []

preprocess = ColumnTransformer([
    ('spline', SplineTransformer(n_knots=3, degree=3, include_bias=False),
     ['x1', 'x2']),
    ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'),
     ['x3', 'x4', 'x5'])
])

for b in range(n_boot):
    boot_idx = list(resample(range(len(data)), random_state=b))
    boot_X = data.iloc[boot_idx]   # data 应为插补后的数据，无缺失值
    boot_y = y_true[boot_idx]
    
    if len(np.unique(boot_y)) < 2:
        continue
    
    # 2. 在 bootstrap 中拟合模型
    X_boot = preprocess.fit_transform(boot_X)
    model_boot = LogisticRegression(max_iter=1000).fit(X_boot, boot_y)
    
    # 3. Bootstrap 中预测（表观）
    pred_boot = model_boot.predict_proba(X_boot)[:, 1]
    auc_boot = roc_auc_score(boot_y, pred_boot)
    
    # 4. 原始数据中预测（测试）
    X_full = preprocess.transform(data)
    pred_test = model_boot.predict_proba(X_full)[:, 1]
    auc_test = roc_auc_score(y_true, pred_test)
    
    # 5. 乐观度
    optimism.append(auc_boot - auc_test)

mean_optimism = np.mean(optimism)
auc_corrected = auc_apparent - mean_optimism
print(f"Bootstrap 成功: {len(optimism)}/{n_boot}")
print(f"乐观校正 AUC: {auc_corrected:.3f}")
```

### 时间-事件结局

```python
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.linear_model import CoxPHSurvivalAnalysis

# 需要结构化生存数据: y = [(event, time), ...]
# Time-dependent AUC
auc_times, mean_auc = cumulative_dynamic_auc(
    y_train, y_test, predicted_risk, times=[365, 730, 1825]  # 1,2,5年(天)
)
for t, auc in zip([1, 2, 5], mean_auc):
    print(f"{t}年 AUC: {auc:.3f}")
```

### 连续型结局

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# R-squared（表观）
r2_apparent = r2_score(observed, predicted)

# 综合性能函数
def calculate_performance(observed, predicted):
    mae = mean_absolute_error(observed, predicted)
    mse = mean_squared_error(observed, predicted)
    r2 = r2_score(observed, predicted)
    return {"MAE": mae, "MSE": mse, "R2": r2}
```

---

## 报告规范

| 项目 | 要求 |
|------|------|
| 表观性能 | 可报告，但需标注 |
| 乐观校正性能 | **必须报告** |
| 置信区间 | 必须提供（Bootstrap 百分位法或 BCa） |
| 多次验证 | 若做了交叉验证，报告各折及平均值 |

---

## 质量检查清单

- [ ] 已选择与结局类型匹配的区分度指标
- [ ] 已报告乐观校正估计值（非仅表观性能）
- [ ] 已提供置信区间
- [ ] 区分度指标的解释结合了临床场景

---

## 返回 Router

← [上一步: Sub-Skill 09](09-fit-prediction-model.md)  
← [返回主指南](../clinical-prediction-model-development-zh.md)  
→ [进入下一步: Sub-Skill 11](11-assess-calibration.md)

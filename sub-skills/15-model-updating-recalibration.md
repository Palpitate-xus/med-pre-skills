# Sub-Skill: 模型更新与再校准

> **所属 Meta-Skill:** [临床预测模型开发指南](../clinical-prediction-model-development-zh.md)

---

## 核心任务

当模型在新数据上表现下降（校准偏移、区分度降低）时，用最少的额外数据修正模型，而非从头重新开发。

---

## 为什么模型会"过时"

| 原因 | 表现 | 示例 |
|------|------|------|
| **时间漂移** | 整体风险水平变化 | 新冠期间患者基线特征改变 |
| **治疗进步** | 干预效果变化 | 新疗法降低了原有预测变量的权重 |
| **人群变化** | 目标人群特征迁移 | 从三甲医院推广到社区医院 |
| **测量方法改变** | 预测变量定义或检测方法更新 | 实验室参考范围调整 |
| **病例混合变化** | 疾病谱改变 | 轻症患者比例增加 |

---

## 更新策略选择

```
模型性能下降？
├── 仅校准偏移（截距≠0，斜率≈1）
│   └── → 简单再校准（调整截距）
├── 过拟合显现（斜率 < 1）
│   └── → 全面再校准（调整截距 + 斜率）
├── 某些变量效应改变
│   └── → 模型修正（更新系数）
├── 需要加入新预测变量
│   └── → 模型扩展（加入新变量并重新估计）
└── 全面失效
    └── → 从头重新开发（最后手段）
```

---

## 方法 1：简单再校准（Recalibration-in-the-large）

修正模型的**系统性偏移**——整体预测偏高或偏低。

### 原理

```
logit(p_observed) = α₀ + logit(p_predicted)
```

若 α₀ ≠ 0，说明存在系统性偏移。用新数据估计 α₀，然后调整所有预测。

### Python 代码

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.special import logit, expit

# 原始模型的预测概率
pred_original = model.predict_proba(X_new)[:, 1]

# 在新数据上估计校准截距
logit_pred = logit(np.clip(pred_original, 0.001, 0.999))
cal_model = LogisticRegression(solver='lbfgs', max_iter=1000)
cal_model.fit(logit_pred.reshape(-1, 1), y_new)

# 调整后的预测
alpha0 = cal_model.intercept_[0]
logit_adjusted = alpha0 + logit_pred
pred_recalibrated = expit(logit_adjusted)

print(f"再校准截距 α₀ = {alpha0:.3f}")
# α₀ > 0 → 原始模型系统性低估风险
# α₀ < 0 → 原始模型系统性高估风险
```

### 适用场景
- 外部验证发现校准截距明显偏离 0
- 斜率接近 1，仅整体偏移
- 更新数据量较小（仅需数百例）

---

## 方法 2：全面再校准（Full Recalibration）

同时修正**系统性偏移**和**过拟合/欠拟合**。

### 原理

```
logit(p_observed) = α₀ + α₁ × logit(p_predicted)
```

- α₀：修正系统性偏移
- α₁：修正预测概率的"极端程度"
  - α₁ < 1：原始模型预测过于极端，需"收缩"
  - α₁ > 1：原始模型预测过于保守

### Python 代码

```python
from sklearn.linear_model import LinearRegression
from scipy.special import logit, expit

logit_pred = logit(np.clip(pred_original, 0.001, 0.999))

# 线性回归估计截距和斜率
# 注意：这里用 LinearRegression 而非 LogisticRegression
# 因为我们已经在 logit 尺度上
recal_model = LinearRegression()
recal_model.fit(logit_pred.reshape(-1, 1), logit(np.clip(y_new, 0.001, 0.999)))

alpha0 = recal_model.intercept_[0]
alpha1 = recal_model.coef_[0]

# 调整后的预测
pred_full_recal = expit(alpha0 + alpha1 * logit_pred)

print(f"再校准截距 α₀ = {alpha0:.3f}, 斜率 α₁ = {alpha1:.3f}")
```

### 适用场景
- 外部验证同时发现截距偏移和斜率偏离 1
- 区分度尚可（AUC 变化不大），但校准明显变差

---

## 方法 3：模型修正（Model Revision）

保持原模型的预测变量集合，但**重新估计所有系数**。

### Python 代码

```python
from sklearn.linear_model import LogisticRegression

# 使用原始模型的预测变量
# 但在新数据上重新拟合（可用惩罚稳定估计）
model_revised = LogisticRegression(
    penalty='l2',
    C=1.0,
    max_iter=1000
)
model_revised.fit(X_new, y_new)

# 比较原始系数 vs. 修正后系数
coef_comparison = pd.DataFrame({
    'variable': feature_names,
    'original': original_model.coef_[0],
    'revised': model_revised.coef_[0]
})
```

### 适用场景
- 某些变量的效应在新人群中明显改变（如治疗变量权重变化）
- 有足够的新数据（EPV ≥ 10）
- 原模型的变量集合仍然合理

---

## 方法 4：模型扩展（Model Extension）

在原有模型基础上**加入新预测变量**。

### 策略

```python
# 原模型变量 + 新变量
X_extended = pd.concat([X_original, X_new_variable], axis=1)

# 推荐：在扩展数据集上使用惩罚回归
from sklearn.linear_model import LogisticRegression

model_extended = LogisticRegression(
    penalty='l2',
    C=1.0,
    max_iter=1000
)
model_extended.fit(X_extended, y_new)
```

### 关键原则
- **原变量保持：** 即使某些原变量在新数据中"不显著"，也不应随意删除（保持模型结构稳定性）
- **新变量需有先验依据：** 不应用纯数据驱动方式加入变量
- **需报告增量价值：** 新模型相比原模型，AUC/校准/DCA 的改善幅度

---

## 更新后的验证

模型更新后，必须在**独立的验证集**上评估：

```python
from sklearn.metrics import roc_auc_score, brier_score_loss

# 区分度
auc_updated = roc_auc_score(y_val, pred_updated)

# 校准
# 重新计算校准截距和斜率

# 临床效用（DCA）
# 比较原模型、更新模型、默认策略
```

### 报告规范

| 项目 | 要求 |
|------|------|
| 更新方法 | 明确说明用了哪种策略（简单/全面/修正/扩展） |
| 更新样本量 | 报告用于更新的样本量和事件数 |
| 原模型性能 | 更新前在新数据上的性能（作为参照） |
| 更新后性能 | 在独立验证集上的性能 |
| 系数变化 | 报告哪些变量的系数发生了显著变化 |

---

## 质量检查清单

- [ ] 已确认模型确实需要更新（而非数据问题或误用）
- [ ] 已选择适当的更新策略（从简单到复杂）
- [ ] 更新数据能代表目标使用人群
- [ ] 更新后模型在独立验证集上评估
- [ ] 已比较原模型和更新模型的性能差异
- [ ] 已报告更新模型的局限性（更新数据量、时间跨度等）

---

## 返回 Router

← [返回主指南](../clinical-prediction-model-development-zh.md)

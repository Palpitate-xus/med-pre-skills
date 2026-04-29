# Sub-Skill 11: 评估校准度

> **所属 Meta-Skill:** [临床预测模型开发指南](../clinical-prediction-model-development-zh.md)  
> **上一步:** [Sub-Skill 10: 评估区分度](10-assess-discrimination.md)  
> **下一步:** [Sub-Skill 12: 决策曲线分析](12-decision-curve-analysis.md)

---

## 核心任务

验证模型的预测概率是否与观测到的实际概率一致。一个模型可以区分度很好，但校准很差。

---

## 为什么校准比区分度更重要

**场景示例：**
- 模型 A：AUC = 0.85，但预测风险 20% 的患者实际风险 40%（校准差）
- 模型 B：AUC = 0.75，但预测风险 20% 的患者实际风险 19%（校准好）

**用于临床决策时，校准良好的模型更可靠。**

---

## 校准评估方法

### 1. 校准图 (Calibration Plot)

- X 轴：模型预测概率
- Y 轴：观测到的实际概率
- 理想线：45°对角线（预测 = 观测）
- LOESS 平滑线 + 95% 置信区间

### 2. 校准截距 (Calibration Intercept)

- **理想值：** 0
- **解读：** 整体预测是否系统性偏高/偏低
  - 截距 > 0：系统性低估风险
  - 截距 < 0：系统性高估风险

### 3. 校准斜率 (Calibration Slope)

- **理想值：** 1
- **解读：**
  - 斜率 < 1：**过拟合**（预测过于极端，高风险预测过高，低风险预测过低）
  - 斜率 > 1：**欠拟合**（预测过于保守，趋向均值）
  - 斜率 ≈ 0.5-0.8：常见于未校正的复杂模型，需要收缩

---

## Python 代码

### 校准图

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import logit, expit
from scipy.interpolate import UnivariateSpline

# 分位数分组计算观测概率
cal_data = pd.DataFrame({
    'predicted': predicted_prob,
    'observed': y_true
})

# 分 10 组 (deciles)
cal_data['decile'] = pd.qcut(cal_data['predicted'], q=10, duplicates='drop')
grouped = cal_data.groupby('decile').agg({
    'predicted': 'mean',
    'observed': 'mean'
}).reset_index()

# LOESS 平滑
from statsmodels.nonparametric.smoothers_lowess import lowess
loess_fit = lowess(cal_data['observed'], cal_data['predicted'], frac=0.3)

# 绘图
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(grouped['predicted'], grouped['observed'], 
           s=100, color='blue', zorder=3, label='Deciles')
ax.plot(loess_fit[:, 0], loess_fit[:, 1], 'r-', label='LOESS')
ax.plot([0, 1], [0, 1], 'k--', label='Ideal')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel('Predicted Probability')
ax.set_ylabel('Observed Proportion')
ax.legend()
ax.set_aspect('equal')
plt.show()
```

### 校准截距和斜率

```python
from sklearn.linear_model import LogisticRegression

# 拟合校准模型: logit(observed) = intercept + slope * logit(predicted)
logit_pred = logit(np.clip(predicted_prob, 0.001, 0.999)).values
cal_model = LogisticRegression(solver='lbfgs', max_iter=1000)
cal_model.fit(logit_pred.reshape(-1, 1), y_true.values if hasattr(y_true, 'values') else y_true)

intercept = cal_model.intercept_[0]
slope = cal_model.coef_[0][0]
print(f"校准截距: {intercept:.3f} (理想值: 0)")
print(f"校准斜率: {slope:.3f} (理想值: 1)")

# 乐观校正：Bootstrap 中重复
from sklearn.utils import resample
n_boot = 500
boot_slopes = []
for _ in range(n_boot):
    boot_idx = resample(range(len(data)))
    boot_data = data.iloc[boot_idx]
    # ... 拟合模型、预测、计算校准斜率 ...
```

### 时间-事件结局的校准

```python
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.metrics import concordance_index_censored

# 按预测风险分 5 组
data['risk_group'] = pd.qcut(predicted_risk, q=5, labels=['G1', 'G2', 'G3', 'G4', 'G5'])

# 各组 Kaplan-Meier 曲线
for group in data['risk_group'].unique():
    mask = data['risk_group'] == group
    time, survival = kaplan_meier_estimator(
        data['event'][mask], data['time'][mask]
    )
    plt.step(time, survival, where='post', label=f'Group {group}')
plt.xlabel('Time')
plt.ylabel('Survival')
plt.legend()
plt.show()
```

---

## 校准问题的修正

| 问题 | 表现 | 修正方法 |
|------|------|----------|
| 斜率 < 1（过拟合） | 预测过于极端 | 收缩/惩罚系数；简化模型 |
| 截距 ≠ 0（系统偏移） | 整体偏高/偏低 | 重新校准（recalibration） |
| 非线性校准误差 | 校准曲线非直线 | 加入非线性变换；重新建模 |

---

## 质量检查清单

- [ ] 已生成并检视校准图
- [ ] 已报告校准截距和斜率
- [ ] 在预测概率的完整范围内进行了评估
- [ ] 已报告乐观校正后的校准度
- [ ] 若校准不良，已尝试修正或讨论

---

## 示例（RRMS）

> 模型在预测概率低于 ~35% 时校准良好；LOESS 平滑线接近理想对角线。在更高预测概率区域（最大约 60%），由于事件数少，校准不确定性增大，95% CI 变宽。

---

## 返回 Router

← [上一步: Sub-Skill 10](10-assess-discrimination.md)  
← [返回主指南](../clinical-prediction-model-development-zh.md)  
→ [进入下一步: Sub-Skill 12](12-decision-curve-analysis.md)

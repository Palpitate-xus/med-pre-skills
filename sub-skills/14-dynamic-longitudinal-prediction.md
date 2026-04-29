# Sub-Skill: 动态预测与纵向数据

> **所属 Meta-Skill:** [临床预测模型开发指南](../clinical-prediction-model-development-zh.md)

---

## 核心任务

利用患者在多个时间点的重复测量数据，进行**动态风险更新**——即随着时间推移和新数据的获取，不断更新对未来结局的预测。

---

## 与静态预测的区别

| 维度 | 静态预测（本指南主体） | 动态预测 |
|------|------------------------|----------|
| 预测时间点 | 仅在基线（time zero）预测一次 | 在多个随访时间点更新预测 |
| 输入数据 | 仅基线预测变量 | 基线 + 截至当前的所有纵向测量 |
| 临床场景 | 入院时评估 5 年死亡风险 | 每次复查后更新当年复发风险 |
| 方法复杂度 | 较低 | 较高 |

---

## 方法 1：Landmarking（里程碑法）

最实用的动态预测方法，将纵向数据转化为多个"时间切片"的静态预测问题。

### 原理

```
时间点 0        时间点 1        时间点 2        时间点 3
  │              │              │              │
  ▼              ▼              ▼              ▼
基线预测      用 t=0~1 数据    用 t=0~2 数据    用 t=0~3 数据
             预测 t>1 结局     预测 t>2 结局     预测 t>3 结局
```

### Python 代码

```python
import pandas as pd
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import cumulative_dynamic_auc

# 假设 data 包含：patient_id, time, event, landmark_time, biomarker_value

landmark_times = [0, 6, 12, 18]  # 月份
models = {}

for lm in landmark_times:
    # 1. 提取在 landmark 时间点仍存活的患者
    at_risk = data[(data['time'] > lm) | 
                   ((data['time'] <= lm) & (data['event'] == 1))]
    
    # 2. 构建 landmark 数据集：用截至 lm 的所有测量
    landmark_data = at_risk[at_risk['measurement_time'] <= lm]
    
    # 3. 取每个患者截至 landmark 的最新测量
    latest = landmark_data.sort_values('measurement_time') \
                          .groupby('patient_id').last().reset_index()
    
    # 4. 更新 time（从 landmark 到事件的时间）
    latest['time_from_lm'] = latest['time'] - lm
    latest['time_from_lm'] = latest['time_from_lm'].clip(lower=0)
    
    # 5. 拟合 Cox 模型
    y = [(e, t) for e, t in zip(latest['event'], latest['time_from_lm'])]
    X = latest[['biomarker_value', 'age', 'sex']]
    
    model = CoxPHSurvivalAnalysis().fit(X, y)
    models[lm] = model

# 使用：患者在 12 个月时的最新 biomarker，用 models[12] 预测未来风险
```

### Landmarking 的优缺点

| 优点 | 缺点 |
|------|------|
| 概念简单，易于实现 | 忽略测量之间的轨迹信息 |
| 可直接使用标准 Cox/Logistic | 每个 landmark 需要足够的事件数 |
| 临床解释性强 | 未考虑测量误差 |

---

## 方法 2：联合模型（Joint Model）

同时建模纵向生物标志物的轨迹和生存结局，利用两者的关联进行动态预测。

### 模型结构

```
纵向子模型：  biomarker(t) = β₀ + β₁×time + β₂×treatment + b₀ + b₁×time + ε
               （混合效应模型，捕捉个体轨迹）

生存子模型：  h(t) = h₀(t) × exp(γ₁×treatment + α×biomarker_true(t))
               （Cox 模型，用纵向标志物的真实值作为时依协变量）

关联参数 α：  纵向标志物每变化一个单位，风险的变化倍数
```

### 为什么联合模型优于 Landmarking？

1. **利用完整轨迹：** 不只是最新值，而是整个变化趋势
2. **处理测量误差：** 区分真实生物信号和随机测量误差
3. **处理缺失：** 即使某次随访缺失，仍可利用混合效应估计

### Python 实现现状

- **R 语言：** `JMbayes2` 包是标准工具（推荐）
- **Python：** 原生支持较弱。可用 `statsmodels` 拟合混合效应模型，再用 `lifelines` 拟合 Cox，但需手动实现关联

### 简化替代方案（Python）

若无法使用 R，可用以下两步近似：

```python
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sksurv.linear_model import CoxPHSurvivalAnalysis

# 步骤 1：用混合效应模型拟合 biomarker 轨迹
# 提取每个患者的随机效应（个体化截距和斜率）
model_lme = smf.mixedlm(
    "biomarker ~ time + treatment",
    data=data_long,
    groups=data_long["patient_id"],
    re_formula="~time"
).fit()

# 提取每个人的预测轨迹参数
random_effects = model_lme.random_effects

# 步骤 2：将个体轨迹参数（截距、斜率）加入生存模型
# 作为"总结性"预测变量
data_surv['biomarker_slope'] = [
    random_effects.get(pid, {}).get('time', 0) 
    for pid in data_surv['patient_id']
]

# 拟合 Cox 模型
y_surv = [(e, t) for e, t in zip(data_surv['event'], data_surv['time'])]
X_surv = data_surv[['biomarker_slope', 'baseline_value', 'treatment']]
model_cox = CoxPHSurvivalAnalysis().fit(X_surv, y_surv)
```

> **注意：** 这是联合模型的**近似**，未完全处理测量误差和时依协变量的内部依赖性。若研究需要发表级方法学严谨性，建议使用 `JMbayes2` (R)。

---

## 动态预测的性能评估

### 时间依赖性性能指标

```python
from sksurv.metrics import cumulative_dynamic_auc

# 评估不同 landmark 时间点模型的区分度
for lm, model in models.items():
    pred_risk = model.predict(X_test)
    auc_times, mean_auc = cumulative_dynamic_auc(
        y_train, y_test, pred_risk, 
        times=[lm + 30, lm + 90, lm + 180]  # 预测 landmark 后 1/3/6 个月
    )
    print(f"Landmark {lm}月: 动态 AUC = {mean_auc.mean():.3f}")
```

### 预测误差（Calibration）

动态预测的校准更复杂，需分别评估每个 landmark：
- 在每个 landmark 时间点，检查预测概率 vs. 实际观测概率
- 理想：各 landmark 的校准斜率均接近 1

---

## 质量检查清单

- [ ] 已明确动态预测的临床场景（何时更新、更新频率）
- [ ] 已选择 Landmarking 或联合模型（根据数据复杂度和样本量）
- [ ] 每个 landmark 时间点有足够的事件数（≥ 10×参数）
- [ ] 纵向测量已处理缺失（混合效应或多重插补）
- [ ] 已报告不同时间点的动态 AUC 或 C-index
- [ ] 已评估各 landmark 的校准度
- [ ] 若使用联合模型：已检查纵向子模型的拟合优度

---

## 返回 Router

← [返回主指南](../clinical-prediction-model-development-zh.md)

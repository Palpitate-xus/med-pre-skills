# Sub-Skill: 算法公平性评估

> **所属 Meta-Skill:** [临床预测模型开发指南](../clinical-prediction-model-development-zh.md)

---

## 核心任务

评估模型在不同亚组（如性别、种族、年龄组、社会经济状态）间的性能是否一致，识别并修正算法偏见。

---

## 为什么公平性不可忽视

**场景示例：**
- 某心肌梗死风险模型在男性中 AUC = 0.85，在女性中 AUC = 0.68
- 某 AI 影像模型在深色皮肤患者中假阳性率显著更高
- 某预后模型在老年患者中系统性高估风险

> **临床后果：** 亚组性能差异会导致某些人群被过度治疗或治疗不足，加剧健康不平等。

---

## 公平性指标

### 1. 人口统计均等（Demographic Parity）

不同亚组的**阳性预测率**应相等。

```
P(Ŷ = 1 | A = 0) = P(Ŷ = 1 | A = 1)
```

- 适用于：筛查工具（如乳腺癌筛查）
- 局限：若基线患病率不同，强行均等可能不合理

### 2. 机会均等（Equalized Odds）

不同亚组的**真阳性率（TPR）**和**假阳性率（FPR）**应相等。

```
P(Ŷ = 1 | Y = 1, A = 0) = P(Ŷ = 1 | Y = 1, A = 1)   # TPR 相等
P(Ŷ = 1 | Y = 0, A = 0) = P(Ŷ = 1 | Y = 0, A = 1)   # FPR 相等
```

- 适用于：诊断/预后模型
- 最受推荐，兼顾了区分度和公平性

### 3. 预测率均等（Predictive Rate Parity / Calibration）

不同亚组中，给定预测概率时，**实际发生概率应相等**。

```
P(Y = 1 | Ŷ = p, A = 0) = P(Y = 1 | Ŷ = p, A = 1)
```

- 即：各亚组的校准曲线应重合
- 临床预测模型中**最重要的公平性指标**

---

## Python 代码

### 亚组性能比较

```python
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix

def evaluate_subgroup_performance(y_true, y_pred_prob, subgroup, threshold=0.5):
    """
    按亚组评估模型性能
    
    Parameters:
    -----------
    y_true : 真实结局
    y_pred_prob : 预测概率
    subgroup : 亚组标签（如 '男'/'女'）
    threshold : 分类阈值
    """
    results = []
    for group in subgroup.unique():
        mask = subgroup == group
        y_g = y_true[mask]
        prob_g = y_pred_prob[mask]
        pred_g = (prob_g >= threshold).astype(int)
        
        # 区分度
        auc_g = roc_auc_score(y_g, prob_g) if len(y_g.unique()) > 1 else None
        
        # 混淆矩阵
        tn, fp, fn, tp = confusion_matrix(y_g, pred_g).ravel()
        
        # 率
        tpr = tp / (tp + fn) if (tp + fn) > 0 else None  # 敏感性
        fpr = fp / (fp + tn) if (fp + tn) > 0 else None  # 1-特异性
        ppv = tp / (tp + fp) if (tp + fp) > 0 else None  # 阳性预测值
        
        results.append({
            'subgroup': group,
            'n': len(y_g),
            'events': y_g.sum(),
            'AUC': auc_g,
            'TPR': tpr,
            'FPR': fpr,
            'PPV': ppv
        })
    
    return pd.DataFrame(results)

# 使用
df_subgroups = evaluate_subgroup_performance(
    y_true, predicted_prob, data['sex'], threshold=0.2
)
print(df_subgroups)
```

### 校准曲线按亚组绘制

```python
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

fig, ax = plt.subplots(figsize=(8, 8))

for group in data['sex'].unique():
    mask = data['sex'] == group
    loess_fit = lowess(
        y_true[mask].values, 
        predicted_prob[mask].values, 
        frac=0.3
    )
    ax.plot(loess_fit[:, 0], loess_fit[:, 1], label=f'Sex={group}')

ax.plot([0, 1], [0, 1], 'k--', label='Ideal')
ax.set_xlabel('Predicted Probability')
ax.set_ylabel('Observed Proportion')
ax.legend()
plt.show()
```

### 公平性差距量化

```python
def fairness_gap(df, metric='AUC', reference_group=None):
    """
    计算各亚组与参考组的性能差距
    """
    if reference_group is None:
        reference_group = df.loc[df['n'].idxmax(), 'subgroup']
    
    ref_val = df[df['subgroup'] == reference_group][metric].values[0]
    
    df['gap'] = df[metric] - ref_val
    df['relative_gap'] = df['gap'] / ref_val
    
    return df[['subgroup', metric, 'gap', 'relative_gap']]

# 使用
gaps = fairness_gap(df_subgroups, metric='AUC', reference_group='男')
print(gaps)
```

---

## 如何修正不公平

### 策略 1：重新加权（Reweighting）

在模型训练时给少数/弱势亚组更高的样本权重。

```python
from sklearn.utils.class_weight import compute_sample_weight

# 按亚组和结局的联合分布计算权重
sample_weights = compute_sample_weight(
    class_weight='balanced',
    y=data['sex'].astype(str) + '_' + data['outcome'].astype(str)
)

model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train, sample_weight=sample_weights)
```

### 策略 2：阈值调整（Threshold Tuning）

为不同亚组设置不同的决策阈值，使 TPR/FPR 相等。

```python
def find_threshold_for_tpr(y_true, y_prob, target_tpr=0.85):
    """找到达到目标 TPR 的阈值"""
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    idx = np.argmin(np.abs(tpr - target_tpr))
    return thresholds[idx]

# 为各亚组分别找阈值
thresholds = {}
for group in data['sex'].unique():
    mask = data['sex'] == group
    thresholds[group] = find_threshold_for_tpr(
        y_true[mask], predicted_prob[mask], target_tpr=0.85
    )

print(thresholds)
# {'男': 0.18, '女': 0.24}  → 女性需要更高阈值才能达到相同 TPR
```

### 策略 3：在模型中加入交互项

让模型显式学习亚组特异性的效应。

```python
from patsy import dmatrices

# 加入 sex × 预测变量的交互项
formula = "y ~ sex * (cr(age, df=3) + x2 + x3)"
y, X = dmatrices(formula, data=data, return_type='dataframe')
model = LogisticRegression(max_iter=1000).fit(X, y)
```

> **注意：** 加入交互项会增加参数数量，需确保样本量充足。

---

## 报告规范

### 必须报告

| 项目 | 内容 |
|------|------|
| 评估的亚组 | 性别、年龄组、种族、社会经济状态等 |
| 各亚组样本量 | 尤其关注少数亚组是否样本充足 |
| 各亚组 AUC / C-index | 区分度是否一致 |
| 各亚组校准截距/斜率 | 校准是否一致 |
| 公平性差距 | 各亚组与参考组的绝对和相对差距 |

### 解释标准

- **AUC 差距 ≤ 0.05：** 通常认为可接受
- **AUC 差距 0.05-0.10：** 需讨论临床影响
- **AUC 差距 > 0.10：** 模型在该亚组中不可靠，需修正或限制使用

---

## 质量检查清单

- [ ] 已识别并报告模型可能影响的敏感亚组
- [ ] 各亚组有足够样本量进行可靠评估（建议每亚组 ≥ 100 例，≥ 10 个事件）
- [ ] 已报告各亚组的区分度和校准度
- [ ] 已量化公平性差距（绝对差距 + 相对差距）
- [ ] 若发现不公平：已尝试修正或已明确标注模型局限
- [ ] 已在讨论中说明模型的公平性影响

---

## 返回 Router

← [返回主指南](../clinical-prediction-model-development-zh.md)

# Sub-Skill 12: 决策曲线分析 (DCA)

> **所属 Meta-Skill:** [临床预测模型开发指南](../clinical-prediction-model-development-zh.md)  
> **上一步:** [Sub-Skill 11: 评估校准度](11-assess-calibration.md)  
> **下一步:** [Sub-Skill 13: 验证与报告](13-validate-report-tripod.md)

---

## 核心任务

判断一个校准良好、区分度高的模型是否真的能改善临床决策——而非仅仅在统计上表现好。

---

## 为什么需要 DCA

**场景：** 模型 AUC = 0.85，校准也很好。但它有用吗？

- 如果 "全治" 策略的净获益比 "用模型决策" 还高 → 模型无用
- 如果 "不治" 策略在大多数阈值下更好 → 模型无用
- 只有模型能在某个临床相关阈值范围内提供**更高的净获益** → 模型有用

---

## DCA 原理

### 三种策略比较

| 策略 | 描述 |
|------|------|
| **Treat All** | 对所有患者采取干预措施 |
| **Treat None** | 对任何患者都不采取干预措施 |
| **Model-based** | 仅对模型预测风险 > 阈值的患者采取干预 |

### 净获益 (Net Benefit)

```
Net Benefit = (True Positives / N) - (False Positives / N) × (pt / (1 - pt))
```

其中 `pt` 是概率阈值（临床决策点）。

---

## Python 代码

### 使用 dcurves（Python 版）

```python
import pandas as pd
import dcurves

# 准备数据
df = pd.DataFrame({
    'outcome': y_true,
    'predicted_risk': predicted_prob
})

# 运行 DCA
dca_results = dcurves.dca(
    data=df,
    outcome='outcome',
    modelnames=['predicted_risk'],
    thresholds=[i/100 for i in range(0, 50)]  # 0-0.5
)

# 绘图
dcurves.plot_dca(dca_results)
```

### 手动计算 DCA

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_net_benefit(y_true, y_pred, threshold):
    """计算单一阈值下的净获益"""
    w = threshold / (1 - threshold)
    tp = np.sum((y_pred >= threshold) & (y_true == 1))
    fp = np.sum((y_pred >= threshold) & (y_true == 0))
    n = len(y_true)
    return (tp / n) - (fp / n) * w

# 计算各策略在不同阈值下的净获益
thresholds = np.arange(0.01, 0.50, 0.01)
results = []

for pt in thresholds:
    # Model-based
    nb_model = calculate_net_benefit(y_true, predicted_prob, pt)
    # Treat all
    nb_all = np.mean(y_true) - (1 - np.mean(y_true)) * pt / (1 - pt)
    # Treat none
    nb_none = 0
    results.append({
        'threshold': pt,
        'model': nb_model,
        'treat_all': nb_all,
        'treat_none': nb_none
    })

dca_df = pd.DataFrame(results)

# 绘图
plt.plot(dca_df['threshold'], dca_df['model'], label='Model')
plt.plot(dca_df['threshold'], dca_df['treat_all'], '--', label='Treat All')
plt.plot(dca_df['threshold'], dca_df['treat_none'], 'k-', label='Treat None')
plt.xlabel('Threshold')
plt.ylabel('Net Benefit')
plt.legend()
plt.show()
```

### 交互式阈值选择

```python
# 找到模型净获益 > treat-all 且 > treat-none 的阈值范围
valid = dca_df[
    (dca_df['model'] > dca_df['treat_none']) & 
    (dca_df['model'] > dca_df['treat_all'])
]
if len(valid) > 0:
    print(f"模型优于全治/不治的阈值范围: {valid['threshold'].min():.2f} - {valid['threshold'].max():.2f}")
```

---

## DCA 解读指南

### 理想结果

```
净获益
  ↑
  │    ╭────── 模型
  │   ╱        
  │  ╱   ╭──── 全治
  │ ╱   ╱
  │╱   ╱
  ├──╱────────── 不治 (净获益 = 0)
  └──────────────→ 阈值
        ↑
     临床决策点在此范围内
```

### 报告要点

1. **阈值范围：** 模型优于默认策略的概率阈值区间
2. **净获益幅度：** 相比全治/不治，每 100 个患者中能多避免多少不良结局
3. **临床相关性：** 该阈值范围是否对应实际的临床决策点

---

## 示例（RRMS）

- **治疗决策：** 是否从一线 DMT 升级到高效 DMT
- **临床阈值范围：** 10%-35%（医生通常在此范围考虑升级）
- **DCA 结果：** 模型在 15%-30% 阈值范围内净获益高于全治和不治
- **结论：** 在此临床场景下，使用模型指导治疗决策优于默认策略

---

## 质量检查清单

- [ ] 已进行并报告 DCA
- [ ] 已考察临床相关的阈值范围（而非仅 0-1 全部范围）
- [ ] 模型在相关范围内展示了相比全治/不治的净获益
- [ ] 已报告具体的有效阈值区间
- [ ] 已解释净获益的临床意义

---

## 返回 Router

← [上一步: Sub-Skill 11](11-assess-calibration.md)  
← [返回主指南](../clinical-prediction-model-development-zh.md)  
→ [进入下一步: Sub-Skill 13](13-validate-report-tripod.md)

# Sub-Skill 07: 恰当处理缺失数据

> **所属 Meta-Skill:** [临床预测模型开发指南](../clinical-prediction-model-development-zh.md)  
> **上一步:** [Sub-Skill 06: 计算样本量](06-sample-size-calculation.md)  
> **下一步:** [Sub-Skill 08: 选择建模方法](08-choose-model-penalization.md)

---

## 核心任务

使用适当的方法处理缺失数据，避免完全案例分析引入的偏倚。

---

## 缺失数据机制

| 机制 | 描述 | 处理方式 |
|------|------|----------|
| **MCAR** (完全随机缺失) | 缺失与任何变量无关 | 完全案例尚可，但仍损失功效 |
| **MAR** (随机缺失) | 缺失与其他观测变量有关 | **多重插补**是标准方法 |
| **MNAR** (非随机缺失) | 缺失与未观测值本身有关 | 需敏感性分析，可能需专门方法 |

> 实践中通常假设 MAR，并用多重插补处理。

---

## 推荐方法：多重插补 (Multiple Imputation)

### 为什么不是单一插补或完全案例分析？

| 方法 | 问题 |
|------|------|
| 完全案例分析 | 损失样本、引入偏倚（若缺失非 MCAR） |
| 均值/中位数插补 | 低估方差、扭曲关系 |
| 单一随机插补 | 未反映插补不确定性 |
| **多重插补** | 保留样本、反映不确定性、推荐标准 |

---

## Python 代码

### 基础多重插补

```python
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor

n_impute = 10  # 通常 10-50 次插补

# 注意：IterativeImputer 要求所有列为数值型
# 分类变量需先编码为数值（如 One-Hot 或 Label Encoding）
# 若需保留分类变量，推荐使用 miceforest（见下方）

# IterativeImputer: 基于链式方程的多重插补
# 使用 ExtraTrees 作为估计器（比默认 BayesianRidge 更强）
imputer = IterativeImputer(
    estimator=ExtraTreesRegressor(n_estimators=10, random_state=42),
    max_iter=10,
    random_state=42,
    sample_posterior=True  # 关键：从后验采样，实现多重插补
)

# 拟合并生成插补数据
imputed = []
for i in range(n_impute):
    imputer.random_state = 42 + i
    imputed_data = pd.DataFrame(
        imputer.fit_transform(data),
        columns=data.columns,
        index=data.index
    )
    imputed.append(imputed_data)
```

### 使用 miceforest（推荐，更接近 R 的 mice）

```python
import miceforest as mf

# 创建内核
kernel = mf.ImputationKernel(
    data,
    datasets=n_impute,  # 插补次数
    save_models=1
)

# 运行 MICE
kernel.mice(
    variable_parameters={'x1': {'pmm_scale': 1.5}},
    verbose=True
)

# 获取插补后的数据集
imputed = []
for i in range(n_impute):
    imputed.append(kernel.complete_data(dataset=i))
```

### 关键要点

1. **纳入辅助变量 (Auxiliary Variables)**
   - 与预测变量或结局相关，但**不在最终模型中**的变量
   - 帮助提高插补质量，但不增加模型复杂度

2. **插补模型匹配分析模型**
   - 若分析用样条 → 插补也用样条
   - 若分析考虑交互 → 插补也纳入交互

3. **多中心数据需考虑聚类**
   - 在插补模型中加入中心变量（如 `clust`）
   - 或使用多水平插补方法

---

## 替代方案

| 场景 | 替代方法 | Python 库 |
|------|----------|-----------|
| 大量变量 | MICE (链式方程多重插补) | `miceforest`, `sklearn IterativeImputer` |
| 多水平结构 | 多水平多重插补 | `miceforest` (支持聚类) |
| 时间序列 | 前向/后向填充 + 不确定性 | `sklearn` (SimpleImputer strategy) |
| 高维数据 | 随机森林插补 | `sklearn IterativeImputer` + `ExtraTrees` |

---

## 缺失数据报告规范

- 报告每个变量的缺失比例
- 报告缺失数据模式（哪些变量常一起缺失）
- 描述缺失机制假设（MCAR/MAR/MNAR）
- 说明插补方法、插补次数、辅助变量
- 进行敏感性分析（对比不同假设下的结果）

---

## 质量检查清单

- [ ] 已检查并报告缺失数据模式
- [ ] 使用多重插补（而非单一插补或完全案例分析）
- [ ] 插补模型中纳入了辅助变量
- [ ] 插补模型复杂度与分析模型匹配
- [ ] 若存在聚类（多中心），已加以考虑
- [ ] 插补次数充足（通常 ≥ 10）
- [ ] 已进行敏感性分析

---

## 返回 Router

← [上一步: Sub-Skill 06](06-sample-size-calculation.md)  
← [返回主指南](../clinical-prediction-model-development-zh.md)  
→ [进入下一步: Sub-Skill 08](08-choose-model-penalization.md)

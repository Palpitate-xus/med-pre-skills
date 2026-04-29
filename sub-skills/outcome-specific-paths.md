# Sub-Skill: 按结局类型的实施路径

> **所属 Meta-Skill:** [临床预测模型开发指南](../clinical-prediction-model-development-zh.md)

---

## 连续型结局

### 样本量计算
```python
# 手动计算（参考 Riley 等公式）
# n = max(parameters / (R² * shrinkage), parameters * 10)
# 例如 R²=0.6, parameters=10, shrinkage=0.9:
n_continuous = max(10 / (0.6 * 0.9), 10 * 10)  # ≈ 100
```

### 模型选择
- OLS + 限制性立方样条（`statsmodels + patsy`）
- GAM（`pygam`）
- 岭回归（`sklearn RidgeCV`）

### 性能指标
| 指标 | 说明 | 理想值 |
|------|------|--------|
| MAE | 平均绝对误差 | 越小越好 |
| MSE | 均方误差 | 越小越好 |
| R² | 解释方差比例 | 越高越好，但需防过拟合 |

### 校准
- 预测值 vs. 观测值散点图
- 理想线：y = x
- LOESS 平滑检验线性关系

---

## 二分类结局

### 样本量计算
```python
from sub_skills_06 import pmsampsize_binary  # 或自定义函数（见 Sub-Skill 06）

n = pmsampsize_binary(
    cstatistic=0.80,
    parameters=12,
    prevalence=0.12
)
```

### 模型选择
- Logistic 回归 + 样条（`statsmodels Logit + patsy`）
- 惩罚 Logistic（`sklearn LogisticRegression`）
- 支持向量机 / 随机森林（大样本时）

### 性能指标
| 指标 | 说明 | 理想值 |
|------|------|--------|
| AUC / C-statistic | 区分度 | ≥ 0.7 可接受，≥ 0.8 良好 |
| Brier Score | 概率校准 | 0-0.25 可接受，越低越好 |
| Calibration intercept | 系统性偏移 | 接近 0 |
| Calibration slope | 过拟合/欠拟合 | 接近 1 |

### 校准
- 校准图（分位数分组或 LOESS）
- 校准截距和斜率
- 可靠性曲线

### 临床效用
- 决策曲线分析（DCA）
- 阈值范围评估

---

## 时间-事件型结局

### 样本量计算
```python
# 生存型结局样本量估算（简化版）
# 需要事件数 ≥ 10×参数（最低），推荐 ≥ 20×参数
events_needed = 15 * 15  # 15 个参数，EPV=15
n_survival = events_needed / 0.12  # 除以事件发生率
```

### 模型选择
- Cox 比例风险 + 惩罚（`sksurv CoxPHSurvivalAnalysis`）
- 灵活参数生存模型（`lifelines`）
- 加速失效时间模型

### 性能指标
| 指标 | 说明 |
|------|------|
| Time-dependent AUC | 时依区分度 |
| C-index | 整体区分度 |
| Calibration at fixed time | 固定时间点校准 |
| Brier Score at fixed time | 固定时间点概率准确性 |

### 校准
- 固定时间点（如 1 年、2 年、5 年）的校准图
- 观测累积发生率 vs. 预测累积发生率

### 特殊考量
- **比例风险假设：** 检查 Schoenfeld 残差
- **竞争风险：** 若存在，使用 Fine-Gray 或特定原因模型

---

## 竞争风险

### 场景
结局 A 的发生可能被结局 B "阻止"（如：心血管死亡 vs. 非心血管死亡）

### 模型选择
| 方法 | 适用 |
|------|------|
| Fine-Gray 次分布风险 | 关注某特定结局的累积发生率 |
| 特定原因 Cox | 关注某特定结局的风险率 |
| 多状态模型 | 复杂的转移过程 |

### 性能指标
- 累积发生率的校准（而非单纯生存概率）
- Time-dependent AUC 针对特定原因

---

## 有序分类结局

### 场景
结局有自然顺序但间距不等（如 mRS 0-6）

### 模型选择
- 比例优势有序 Logistic（`statsmodels MNLogit` 或 `mord`）
- 连续型替代（若间距可近似为等距）

### 优势
- 比二分类化保留更多信息
- 可估计各等级的概率

---

## 返回 Router

← [返回主指南](../clinical-prediction-model-development-zh.md)

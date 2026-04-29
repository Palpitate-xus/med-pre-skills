# Sub-Skill 02: 检索现有模型

> **所属 Meta-Skill:** [临床预测模型开发指南](../clinical-prediction-model-development-zh.md)  
> **上一步:** [Sub-Skill 01: 明确研究目的](01-define-aim-population-users.md)  
> **下一步:** [Sub-Skill 03: 选择数据来源](03-select-data-source.md)

---

## 核心任务

在从零开发之前，先确认是否已有可用模型。很多时候，**验证/更新现有模型**比重新开发更优。

---

## 检索策略

### 1. 系统检索（PROGRESS 框架）

| 检索要素 | 示例关键词 |
|----------|-----------|
| Population | "multiple sclerosis", "stroke", "heart failure" |
| Predictors | "clinical prediction", "risk score", "nomogram" |
| Outcome | "relapse", "mortality", "readmission" |
| Setting | "outpatient", "ICU", "emergency department" |

### 2. 关键数据库

- PubMed / MEDLINE
- Embase
- Cochrane Library
- PROSPERO（系统评价注册库）
- 专业注册库（如 COVID-19 预测模型注册库）

### 3. 评价工具

| 工具 | 用途 | 适用场景 | 版本 |
|------|------|----------|------|
| **PROBAST+AI** | 评估预测模型的偏倚风险和适用性 | 评价单个预测模型研究 | 2025（替代 2019 版） |
| **CHARMS** | 系统评价中预测模型研究的数据提取 | 做系统综述时提取数据 | 2014 |

> **PROBAST+AI 更新要点：** 2025 版区分了**开发**（16 个信号问题）和**评估**（18 个信号问题）两条轨道，并新增了对算法公平性（fairness）的评估。

**评价关注重点：** 样本量、缺失数据处理、验证策略、校准度报告、模型是否经过外部验证

---

## 决策树

```
是否存在已有模型？
├── 否 → 进入 Sub-Skill 03，从头开发
├── 是 → 该模型是否经过充分外部验证？
│   ├── 是 → 是否在你的目标人群中验证过？
│   │   ├── 是 → 考虑直接采用（进入 Sub-Skill 13 做本地验证）
│   │   └── 否 → 进行外部验证（Sub-Skill 13）
│   └── 否 → 模型质量如何？
│       ├── 质量好 → 先外部验证，再考虑更新
│       └── 质量差 → 考虑从头开发（Sub-Skill 03）
```

---

## 质量检查清单

- [ ] 已完成系统文献检索（考虑 PROGRESS 框架）
- [ ] 已识别并对现有模型进行了严格评价（使用 PROBAST）
- [ ] 已论证决策：开发新模型 vs. 验证/更新现有模型
- [ ] 决策理由已记录

---

## 返回 Router

← [上一步: Sub-Skill 01](01-define-aim-population-users.md)  
← [返回主指南](../clinical-prediction-model-development-zh.md)  
→ [进入下一步: Sub-Skill 03](03-select-data-source.md)

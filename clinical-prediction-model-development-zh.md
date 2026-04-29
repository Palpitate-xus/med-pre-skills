# Meta-Skill: 临床预测模型开发指南

## 来源

**原文:** Efthimiou O, Seo M, Chalkou K, Debray T, Egger M, Salanti G. "Developing clinical prediction models: a step-by-step guide." *BMJ*. 2024;386:e078276.  
**DOI:** [10.1136/bmj-2023-078276](https://doi.org/10.1136/bmj-2023-078276)  
**PMID:** 39227063 | **PMCID:** PMC11369751  
**配套文章:** "Evaluation of clinical prediction models (part 2)" (*BMJ* 2024;384:e074820)

---

## 快速入口 — 你处于哪个阶段？

| 你的状态 | 推荐入口 |
|----------|----------|
| 刚有一个临床预测的想法，还没开始 | [Step 1: 明确目的与人群](sub-skills/01-define-aim-population-users.md) |
| 不确定是否要开发新模型 | [Step 2: 检索现有模型](sub-skills/02-search-existing-models.md) |
| 数据在手，准备分析 | [Step 4: 定义结局](sub-skills/04-define-outcome.md) → [Step 5: 选预测变量](sub-skills/05-select-predictors.md) → [Step 6: 算样本量](sub-skills/06-sample-size-calculation.md) |
| 数据有缺失，不知道怎么处理 | [Step 7: 缺失数据处理](sub-skills/07-handle-missing-data.md) |
| 模型已拟合，要评估性能 | [Step 10: 区分度](sub-skills/10-assess-discrimination.md) → [Step 11: 校准度](sub-skills/11-assess-calibration.md) → [Step 12: DCA](sub-skills/12-decision-curve-analysis.md) |
| 要写论文/报告 | [Step 13: 验证与 TRIPOD 报告](sub-skills/13-validate-report-tripod.md) |
| 想知道哪些做法会犯错 | [常见陷阱与规避](sub-skills/common-pitfalls.md) |
| 找 Python 库和代码 | [Python 库速查表](sub-skills/python-packages-reference.md) |

---

## 核心原则

1. **模型复杂度匹配样本量** — 防范过拟合与欠拟合
2. **惩罚优于逐步选择** — 岭回归 / LASSO / Elastic Net
3. **连续变量保持连续** — 用样条，绝不事后二分类
4. **超越区分度** — 校准 + 临床效用（DCA）是必做项
5. **透明报告** — 遵循 TRIPOD，分享代码

---

## 13 步框架总览

| 步骤 | 内容 | 关键产出 | 链接 |
|:----:|------|----------|------|
| **1** | 明确目的、目标人群、预期使用者 | 研究问题说明书 | [Sub-Skill 01](sub-skills/01-define-aim-population-users.md) |
| **2** | 检索现有模型 | 文献综述 + PROBAST 评价 | [Sub-Skill 02](sub-skills/02-search-existing-models.md) |
| **3** | 选择数据来源 | 数据质量评估报告 | [Sub-Skill 03](sub-skills/03-select-data-source.md) |
| **4** | 定义结局变量 | 结局定义（客观、可测、时间点明确） | [Sub-Skill 04](sub-skills/04-define-outcome.md) |
| **5** | 选择预测变量 | 预测变量集（先验知识驱动） | [Sub-Skill 05](sub-skills/05-select-predictors.md) |
| **6** | 计算样本量 | 样本量计算结果 / 自由度预算 | [Sub-Skill 06](sub-skills/06-sample-size-calculation.md) |
| **7** | 处理缺失数据 | 插补后数据集 | [Sub-Skill 07](sub-skills/07-handle-missing-data.md) |
| **8** | 选择建模方法与惩罚 | 建模方案（statsmodels / sklearn / pygam） | [Sub-Skill 08](sub-skills/08-choose-model-penalization.md) |
| **9** | 拟合预测模型 | 拟合的模型对象 | [Sub-Skill 09](sub-skills/09-fit-prediction-model.md) |
| **10** | 评估区分度 | 乐观校正的 AUC/C-index/R² | [Sub-Skill 10](sub-skills/10-assess-discrimination.md) |
| **11** | 评估校准度 | 校准图 + 截距/斜率 | [Sub-Skill 11](sub-skills/11-assess-calibration.md) |
| **12** | 评估临床效用（DCA） | 决策曲线 + 有效阈值范围 | [Sub-Skill 12](sub-skills/12-decision-curve-analysis.md) |
| **13** | 验证与 TRIPOD 报告 | 完整论文/报告 + 共享代码 | [Sub-Skill 13](sub-skills/13-validate-report-tripod.md) |

---

## 按结局类型的快速路径

- [连续型结局](sub-skills/outcome-specific-paths.md#连续型结局)
- [二分类结局](sub-skills/outcome-specific-paths.md#二分类结局)
- [时间-事件型结局](sub-skills/outcome-specific-paths.md#时间-事件型结局)
- [竞争风险](sub-skills/outcome-specific-paths.md#竞争风险)
- [有序分类结局](sub-skills/outcome-specific-paths.md#有序分类结局)

---

## 辅助资源

| 资源 | 内容 |
|------|------|
| [常见陷阱与规避](sub-skills/common-pitfalls.md) | 10 个最常见方法学错误及正确做法 |
| [Python 库速查表](sub-skills/python-packages-reference.md) | 按用途分类的 Python 库清单 + 安装命令 + 依赖图 |
| [完整示例: RRMS 复发预测](sub-skills/rrms-example.md) | 从 Step 1 到 Step 13 的完整应用实例 |

---

## 快速参考：从数据到部署

```
1. 明确问题、人群、使用者
2. 检索现有模型（PROGRESS/PROBAST）
3. 选择代表性数据来源
4. 定义结局变量（连续型 > 二分类）
5. 选择预测变量（基于先验知识，不分类）
6. 计算样本量（自定义函数 / pmsampsize）
7. 处理缺失数据（多重插补）
8. 选择模型 + 惩罚策略
9. 拟合模型（跨插补合并）
10. 评估区分度（乐观校正）
11. 评估校准度（校准图 + 截距/斜率）
12. 评估临床效用（DCA）
13. 内部验证 + 按 TRIPOD 报告 + 分享代码
```

---

## 与报告标准的对应

| 本指南 | 标准 | 用途 | 当前版本 |
|--------|------|------|----------|
| Step 1-5, 13 | **TRIPOD+AI** | 预测模型研究的透明报告 | 2024（替代 2015 版） |
| Step 2 | **PROGRESS** | 检索现有模型的框架 | 2013 |
| Step 2 | **PROBAST+AI** | 预测模型研究的偏倚风险评估 | 2025（替代 2019 版） |
| Step 2 | **CHARMS** | 系统评价中预测模型研究的数据提取 | 2014 |
| Step 6 | **Riley 等人** | 预测模型的样本量计算（开发） | 2020 |
| Step 13 | **Riley 等人** | 预测模型的样本量计算（外部验证） | 2024 |

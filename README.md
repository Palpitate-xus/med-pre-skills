# 临床预测模型开发指南

基于 BMJ 2024 文章 *"Developing clinical prediction models: a step-by-step guide"* (Efthimiou et al.) 的完整中文实现指南，全部代码使用 Python。

**原文信息:**  
Efthimiou O, Seo M, Chalkou K, Debray T, Egger M, Salanti G. Developing clinical prediction models: a step-by-step guide. *BMJ*. 2024;386:e078276.  
DOI: [10.1136/bmj-2023-078276](https://doi.org/10.1136/bmj-2023-078276)

---

## 项目简介

本项目将 BMJ 13 步临床预测模型开发框架转化为可直接参考的技术文档，覆盖从研究设计到模型部署的完整流程：

- 明确研究问题与目标人群
- 检索并评价现有模型（PROGRESS / PROBAST）
- 数据准备、缺失值处理（多重插补）
- 建模方法选择与惩罚策略
- 模型性能评估（区分度 + 校准度 + 临床效用 DCA）
- Bootstrap 乐观校正与内部验证
- TRIPOD 规范报告

所有代码示例均使用 Python（pandas、scikit-learn、statsmodels、pygam 等），可直接复制运行。

---

## 目录结构

```
med-pre-skills/
├── clinical-prediction-model-development-zh.md   # 主指南（Router）
├── README.md                                      # 本文件
└── sub-skills/
    ├── 01-define-aim-population-users.md          # Step 1: 明确研究目的
    ├── 02-search-existing-models.md               # Step 2: 检索现有模型
    ├── 03-select-data-source.md                   # Step 3: 选择数据来源
    ├── 04-define-outcome.md                       # Step 4: 定义结局变量
    ├── 05-select-predictors.md                    # Step 5: 选择预测变量
    ├── 06-sample-size-calculation.md              # Step 6: 计算样本量
    ├── 07-handle-missing-data.md                  # Step 7: 处理缺失数据
    ├── 08-choose-model-penalization.md            # Step 8: 建模方法与惩罚
    ├── 09-fit-prediction-model.md                 # Step 9: 拟合预测模型
    ├── 10-assess-discrimination.md                # Step 10: 评估区分度
    ├── 11-assess-calibration.md                   # Step 11: 评估校准度
    ├── 12-decision-curve-analysis.md              # Step 12: 决策曲线分析
    ├── 13-validate-report-tripod.md               # Step 13: 验证与 TRIPOD 报告
    ├── common-pitfalls.md                         # 10 个常见陷阱与规避
    ├── outcome-specific-paths.md                  # 按结局类型的实施路径
    ├── python-packages-reference.md               # Python 库速查表
    └── rrms-example.md                            # 完整示例：RRMS 复发预测
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install numpy pandas matplotlib seaborn scikit-learn statsmodels patsy
pip install pygam dcurves miceforest
pip install scikit-survival lifelines
```

### 2. 选择入口

| 你的阶段 | 推荐入口 |
|----------|----------|
| 刚有想法，还没开始 | [Step 1: 明确目的与人群](sub-skills/01-define-aim-population-users.md) |
| 不确定是否要开发新模型 | [Step 2: 检索现有模型](sub-skills/02-search-existing-models.md) |
| 数据在手，准备分析 | [Step 4: 定义结局](sub-skills/04-define-outcome.md) → [Step 5: 选预测变量](sub-skills/05-select-predictors.md) → [Step 6: 算样本量](sub-skills/06-sample-size-calculation.md) |
| 数据有缺失 | [Step 7: 缺失数据处理](sub-skills/07-handle-missing-data.md) |
| 模型已拟合，要评估 | [Step 10: 区分度](sub-skills/10-assess-discrimination.md) → [Step 11: 校准度](sub-skills/11-assess-calibration.md) → [Step 12: DCA](sub-skills/12-decision-curve-analysis.md) |
| 要写论文/报告 | [Step 13: 验证与 TRIPOD 报告](sub-skills/13-validate-report-tripod.md) |

### 3. 查看完整示例

[RRMS 复发预测示例](sub-skills/rrms-example.md) 展示了从 Step 1 到 Step 13 的完整应用，可作为实际项目的参考模板。

---

## 核心原则

1. **模型复杂度匹配样本量** — 防范过拟合与欠拟合
2. **惩罚优于逐步选择** — 使用岭回归 / LASSO / Elastic Net
3. **连续变量保持连续** — 用样条，绝不事后二分类
4. **超越区分度** — 校准 + 临床效用（DCA）是必做项
5. **透明报告** — 遵循 TRIPOD，分享代码

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

---

## 与报告标准的对应

| 本指南 | 标准 | 用途 |
|--------|------|------|
| Step 1-5, 13 | **TRIPOD** | 预测模型研究的透明报告 |
| Step 2 | **PROGRESS** | 检索现有模型的框架 |
| Step 2 | **PROBAST** | 预测模型研究的偏倚风险评估 |
| Step 6 | **Riley 等人** | 预测模型的样本量计算 |

---

## 适用场景

- 临床医学研究生/博士生的预测模型论文
- 临床医生合作中的方法学参考
- 生物统计/流行病学课程教学材料
- 医院真实世界数据（RWD）分析项目

---

## 许可

文档内容基于公开的 BMJ 学术文章整理，代码示例可自由使用。

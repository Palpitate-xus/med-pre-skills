# Sub-Skill: Python 库速查表

> **所属 Meta-Skill:** [临床预测模型开发指南](../clinical-prediction-model-development-zh.md)

---

## 按用途分类

### 数据处理与缺失值

| 库 | 函数/类 | 用途 |
|----|---------|------|
| `pandas` | `read_csv()`, `DataFrame`, `qcut()` | 数据处理、分位数分组 |
| `numpy` | `mean()`, `std()`, `logspace()` | 数值计算 |
| `sklearn.impute` | `IterativeImputer`, `SimpleImputer` | 多重插补、简单插补 |
| `miceforest` | `ImputationKernel` | MICE 多重插补（推荐） |

### 回归建模

| 库 | 函数/类 | 用途 |
|----|---------|------|
| `statsmodels` | `OLS()`, `Logit()`, `GLM()` | 传统统计回归、校准检验 |
| `statsmodels.stats.outliers_influence` | `variance_inflation_factor` | VIF 共线性检查 |
| `patsy` | `dmatrices()`, `dmatrix()` | 公式接口、样条基生成 |
| `pygam` | `LinearGAM`, `LogisticGAM`, `s()` | 广义可加模型 |

### 惩罚回归

| 库 | 函数/类 | 用途 |
|----|---------|------|
| `sklearn.linear_model` | `RidgeCV`, `LassoCV`, `ElasticNetCV` | 岭回归、LASSO、弹性网络 |
| `sklearn.preprocessing` | `SplineTransformer`, `OneHotEncoder` | 样条变换、独热编码 |
| `sklearn.pipeline` | `Pipeline`, `ColumnTransformer` | 管道、列变换 |

### 模型性能评估

| 库 | 函数/类 | 用途 |
|----|---------|------|
| `sklearn.metrics` | `roc_auc_score`, `brier_score_loss`, `r2_score` | AUC、Brier、R² |
| `sklearn.metrics` | `mean_absolute_error`, `mean_squared_error` | MAE、MSE |
| `sksurv.metrics` | `cumulative_dynamic_auc`, `concordance_index_censored` | 生存分析指标 |
| `sksurv.nonparametric` | `kaplan_meier_estimator` | Kaplan-Meier 估计 |

### 决策曲线分析

| 库 | 函数/类 | 用途 |
|----|---------|------|
| `dcurves` | `dca()`, `plot_dca()` | 决策曲线分析 |

### Bootstrap 与重采样

| 库 | 函数/类 | 用途 |
|----|---------|------|
| `sklearn.utils` | `resample()` | Bootstrap 重采样 |
| `sklearn.model_selection` | `cross_val_score`, `KFold` | 交叉验证 |

### 可视化

| 库 | 函数/类 | 用途 |
|----|---------|------|
| `matplotlib` | `pyplot` | 通用绘图 |
| `seaborn` | `scatterplot`, `lineplot` | 统计可视化 |
| `statsmodels.nonparametric` | `lowess` | LOESS 平滑 |

### 生存分析

| 库 | 函数/类 | 用途 |
|----|---------|------|
| `sksurv.linear_model` | `CoxPHSurvivalAnalysis` | Cox 回归 |
| `lifelines` | `CoxPHFitter`, `KaplanMeierFitter` | 生存分析（备选） |

### 报告与展示

| 库 | 函数/类 | 用途 |
|----|---------|------|
| `streamlit` | `st.title()`, `st.metric()` | 交互式网页 App |
| `gradio` | `Interface()` | 快速模型演示 |

---

## 安装命令

```bash
pip install numpy pandas matplotlib seaborn scikit-learn statsmodels patsy
pip install pygam dcurves miceforest
pip install scikit-survival lifelines
pip install streamlit gradio
```

---

## 完整分析流程的 Python 库依赖图

```
Step 6: 样本量          Step 7: 缺失数据         Step 8-9: 建模
    ↓                        ↓                       ↓
numpy (手动公式)        sklearn / miceforest      sklearn / statsmodels / pygam
    ↓                        ↓                       ↓
                                              patsy (样条基)
                                                    ↓
Step 10: 区分度 ←────── Step 11: 校准度 ←────── 模型对象
    ↓                        ↓
sklearn.metrics          statsmodels / matplotlib
    ↓                        ↓
                     Step 12: DCA
                          ↓
                       dcurves
                          ↓
Step 13: 验证 + 报告
    ↓
sklearn.utils.resample + streamlit
```

---

## 返回 Router

← [返回主指南](../clinical-prediction-model-development-zh.md)

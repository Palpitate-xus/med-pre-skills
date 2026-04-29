# Sub-Skill 13: 内部验证与 TRIPOD 报告

> **所属 Meta-Skill:** [临床预测模型开发指南](../clinical-prediction-model-development-zh.md)  
> **上一步:** [Sub-Skill 12: 决策曲线分析](12-decision-curve-analysis.md)  
> **下一步:** 外部验证（超出本指南范围）

---

## 核心任务

通过严格的内部验证评估模型的泛化能力，并按照 TRIPOD 指南透明报告所有细节。

---

## 内部验证方法对比

| 方法 | 推荐度 | 优点 | 缺点 |
|------|--------|------|------|
| **Bootstrap** | ★★★ | 利用全部数据，稳定，可校正 optimism | 计算量大 |
| **K-fold CV** | ★★☆ | 适合大数据/ML | 小样本不稳定 |
| **Leave-one-out** | ★★☆ | 几乎用全部数据 | 方差高，计算量大 |
| **Split-sample** | ☗ | 简单 | 浪费数据，小样本极不稳定 |
| **Internal-external CV** | ★★★ | 模拟外部验证，适合多中心 | 需要聚类结构 |

---

## Bootstrap 乐观校正（推荐）

### 流程

1. 从开发数据中有放回抽取 bootstrap 样本（大小 = N）
2. 在 bootstrap 样本中拟合模型
3. 在 bootstrap 样本中评估性能 → **表观性能**
4. 在**原始完整数据**中评估同一模型 → **测试性能**
5. **乐观度** = 表观性能 - 测试性能
6. 重复 200-500 次，取平均乐观度
7. **乐观校正性能** = 开发数据的表观性能 - 平均乐观度

### Python 代码

```python
import numpy as np
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer

n_boot = 500
optimism_auc = []
optimism_brier = []

preprocess = ColumnTransformer([
    ('spline', SplineTransformer(n_knots=3, degree=3, include_bias=False),
     ['x1', 'x2']),
    ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'),
     ['x3', 'x4', 'x5'])
])

# 注意：data 应为插补后的数据（无缺失值）
for b in range(n_boot):
    # 1. Bootstrap 样本
    boot_idx = list(resample(range(len(data)), random_state=b))
    boot_data = data.iloc[boot_idx]
    boot_y = y[boot_idx]
    
    if len(np.unique(boot_y)) < 2:
        continue
    
    # 2. 在 bootstrap 中拟合模型
    X_boot = preprocess.fit_transform(boot_data)
    model_boot = LogisticRegression(max_iter=1000).fit(X_boot, boot_y)
    
    # 3. Bootstrap 中的预测（表观）
    pred_boot = model_boot.predict_proba(X_boot)[:, 1]
    auc_boot = roc_auc_score(boot_y, pred_boot)
    brier_boot = brier_score_loss(boot_y, pred_boot)
    
    # 4. 原始数据中的预测（测试）
    X_full = preprocess.transform(data)
    pred_test = model_boot.predict_proba(X_full)[:, 1]
    auc_test = roc_auc_score(y, pred_test)
    brier_test = brier_score_loss(y, pred_test)
    
    # 5. 计算性能差异（乐观度）
    optimism_auc.append(auc_boot - auc_test)
    optimism_brier.append(brier_boot - brier_test)

# 6-7. 平均乐观度并校正
mean_optimism_auc = np.mean(optimism_auc)
mean_optimism_brier = np.mean(optimism_brier)

apparent_auc = roc_auc_score(y, final_predictions)
apparent_brier = brier_score_loss(y, final_predictions)

auc_corrected = apparent_auc - mean_optimism_auc
brier_corrected = apparent_brier - mean_optimism_brier

print(f"Bootstrap 成功: {len(optimism_auc)}/{n_boot}")
print(f"表观 AUC: {apparent_auc:.3f} → 校正: {auc_corrected:.3f}")
print(f"表观 Brier: {apparent_brier:.3f} → 校正: {brier_corrected:.3f}")
```

### 与多重插补结合

```python
# 理想做法：每个 bootstrap 样本内重新做 MI
# 简化做法：在已插补的数据上 bootstrap

for _ in range(n_boot):
    for i in range(n_impute):
        boot_idx = resample(range(len(imputed[i])))
        imp_boot = imputed[i].iloc[boot_idx]
        # 拟合、评估...
```

---

## 内部-外部交叉验证 (Internal-External CV)

适用于**多中心数据**，模拟外部验证：

```
中心数 = k
for i in 1:k:
  训练集 = 除中心 i 外的所有中心
  测试集 = 中心 i
  在训练集上开发模型
  在测试集上评估性能
end
合并所有测试集的预测，计算整体性能
```

### Python 代码

```python
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import SplineTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

clusters = data['center'].unique()
predictions_iecv = []
observed_iecv = []

preprocess = ColumnTransformer([
    ('spline', SplineTransformer(n_knots=3, degree=3, include_bias=False),
     ['x1', 'x2']),
    ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'),
     ['x3', 'x4', 'x5'])
])

# 注意：data 应为插补后的数据（无缺失值）
for clust in clusters:
    train = data[data['center'] != clust]
    test = data[data['center'] == clust]
    
    # 跳过样本过少的中心
    if len(test) < 20 or len(train) < 50:
        continue
    
    X_train = preprocess.fit_transform(train)
    X_test = preprocess.transform(test)
    
    model_clust = LogisticRegression(max_iter=1000).fit(X_train, train['y'])
    pred_clust = model_clust.predict_proba(X_test)[:, 1]
    
    predictions_iecv.extend(pred_clust)
    observed_iecv.extend(test['y'])

# 评估合并后的预测
if len(observed_iecv) > 0:
    auc_iecv = roc_auc_score(observed_iecv, predictions_iecv)
    print(f"Internal-External CV AUC: {auc_iecv:.3f}")
else:
    print("IECV 未能完成（中心样本过小）")
```

---

## TRIPOD 报告清单（核心项）

### 标题与摘要
- [ ] 标题包含 "development" 或 "validation"
- [ ] 摘要说明研究类型、人群、结局、预测变量

### 方法
- [ ] 研究设计和数据来源
- [ ]  eligibility criteria（纳入/排除标准）
- [ ] 结局定义和测量方法
- [ ] 预测变量定义和测量方法
- [ ] 样本量计算
- [ ] 缺失数据处理
- [ ] 模型开发方法（包括惩罚/变量选择）
- [ ] 模型性能指标（区分度 + 校准 + 临床效用）
- [ ] 验证策略（Bootstrap / CV）

### 结果
- [ ] 参与者流程图
- [ ] 基线特征表
- [ ] 模型公式/系数
- [ ] 表观性能 vs. 乐观校正性能
- [ ] 校准图（含截距和斜率）
- [ ] DCA 图
- [ ] 内部-外部验证结果（如适用）

### 讨论
- [ ] 局限性（过拟合风险、泛化性）
- [ ] 与现有模型比较
- [ ] 临床意义解读
- [ ] 未来外部验证计划

---

## 代码与工具分享

### 推荐做法

- [ ] 分析代码上传至 GitHub（附 README）
- [ ] 提供可复现的 Jupyter Notebook 或 Quarto 文档
- [ ] 考虑制作 Streamlit App 或网页计算器
- [ ] 提供示例数据和运行说明

### Streamlit App 最小示例

```python
import streamlit as st
import numpy as np

st.title("临床预测模型计算器")

with st.sidebar:
    age = st.number_input("年龄:", value=50, min_value=18, max_value=100)
    sex = st.selectbox("性别:", ["男", "女"])
    # ... 其他输入
    predict_btn = st.button("计算风险")

if predict_btn:
    # 构建输入特征
    X_new = np.array([[age, 1 if sex == "男" else 0, ...]])
    pred_prob = model.predict_proba(X_new)[0, 1]
    
    st.metric("预测复发风险", f"{pred_prob*100:.1f}%")
    
    # 根据阈值给出建议
    if pred_prob > 0.25:
        st.warning("高风险：建议考虑升级治疗")
    else:
        st.success("低风险：继续当前治疗")

# 运行: streamlit run app.py
```

---

## 质量检查清单

- [ ] 已进行 Bootstrap（500 次推荐）或稳健的交叉验证
- [ ] 已报告所有指标的乐观校正性能
- [ ] 已完成 TRIPOD 检查清单
- [ ] 分析代码已公开分享
- [ ] 模型对最终使用者可及（网页计算器、App 或列线图）
- [ ] 已讨论局限性和未来验证计划

---

## 返回 Router

← [上一步: Sub-Skill 12](12-decision-curve-analysis.md)  
← [返回主指南](../clinical-prediction-model-development-zh.md)

---

## 下一步

本指南覆盖内部验证。模型若要在临床使用，还需：
- **外部验证**：在独立数据集、不同中心、不同时间验证
- **影响研究**：评估模型实际使用是否改善患者结局
- **持续监测**：部署后定期评估模型性能漂移

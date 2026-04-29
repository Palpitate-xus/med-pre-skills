# Sub-Skill 06: 计算所需样本量

> **所属 Meta-Skill:** [临床预测模型开发指南](../clinical-prediction-model-development-zh.md)  
> **上一步:** [Sub-Skill 05: 选择预测变量](05-select-predictors.md)  
> **下一步:** [Sub-Skill 07: 处理缺失数据](07-handle-missing-data.md)

---

## 核心任务

在分析前计算支持模型复杂度所需的样本量，或在样本量固定时反推允许的最大参数数。

---

## 为什么样本量至关重要

样本量不足会导致：
- 过拟合（模型记住噪声而非信号）
- 参数估计不稳定（高方差）
- 表现性能过于乐观
- 模型泛化能力差

**关键原则:** 样本量的重要性远超许多研究者的认知。

---

## Riley 等人的样本量框架

使用 `pmsampsize`（R 包）或对应公式计算，基于四个准则：

1. **准则 1:** 小样本中过拟合导致的乐观度可接受
2. **准则 2:** 目标预测性能（如 R²、C-statistic）能精确估计
3. **准则 3:** 预测值的小调整（shrinkage）足够
4. **准则 4:** 整体模型拟合可接受

---

## Python 代码

`pmsampsize` 目前只有 R 版本。Python 中可用基于 EPV（Events Per Variable）的简化方法估算：

```python
import numpy as np

def pmsampsize_binary(parameters, prevalence, epv=10):
    """
    二分类结局预测模型样本量估算（基于 EPV）
    
    Parameters:
    -----------
    parameters : int    模型参数个数
    prevalence : float  结局发生率 (0~1)
    epv        : int    每个参数所需事件数 (默认 10，推荐 20)
    
    Returns:
    --------
    n : int  所需总样本量
    """
    events = max(parameters * epv, 100)
    n = int(np.ceil(events / prevalence))
    print(f"参数={parameters}, 发生率={prevalence:.1%}, EPV={epv}")
    print(f"推荐最小样本量: {n} (事件数: {events})")
    return n

# ========== 二分类结局 ==========
n = pmsampsize_binary(
    parameters=12,      # 模型参数个数
    prevalence=0.12,    # 结局发生率
    epv=20              # 保守估计用 EPV=20
)

# ========== 连续型结局 ==========
# n = max(parameters / (R² × shrinkage), parameters × 10)
n_continuous = max(10 / (0.6 * 0.9), 10 * 10)  # ≈ 100
```

---

## 若样本量固定：反向计算

当样本量受限时（如 "我们只有 N=500"），反向计算允许的最大参数数：

```python
# 方法：不断调整 parameters，直到满足所有准则
# 然后将 "自由度预算" 分配给：
#   - 预测变量数量
#   - 非线性项（样条节点）
#   - 交互项
#   - 注意：分类变量的 k 个水平消耗 k-1 个参数

# 反向计算示例（固定 N=500，prevalence=0.3）
N = 500
events = N * 0.3  # 150 个事件
max_params = events / 10  # EPV=10 时最多 15 个参数
print(f"固定样本 {N}，最多允许约 {int(max_params)} 个参数")
```

### 自由度预算分配示例

假设反向计算后最大允许 20 个参数：

| 项目 | 消耗参数 | 剩余 |
|------|----------|------|
| 起始预算 | — | 20 |
| 5 个连续变量（各用 rcs 3 节点） | 5 × 2 = 10 | 10 |
| 3 个二分类变量 | 3 × 1 = 3 | 7 |
| 1 个 4 水平分类变量 | 1 × 3 = 3 | 4 |
| 截距 | 1 | 3 |
| 保留缓冲（防过拟合） | 3 | 0 |

---

## 经验法则（粗略参考）

| 结局类型 | 最低要求 | 推荐 |
|----------|----------|------|
| 二分类 | EPV ≥ 10 | EPV ≥ 20 |
| 连续型 | 10-20 例/参数 | 50+ 例/参数 |
| 生存型 | 事件数 ≥ 10×参数 | 事件数 ≥ 20×参数 |

> EPV = Events Per Variable（每个变量对应的事件数）

---

## 质量检查清单

- [ ] 分析前已完成样本量计算
- [ ] 每个参数有足够的事件数（EPV），或连续型结局有等效保障
- [ ] 若样本量固定：自由度预算已明确分配
- [ ] 实际可用样本（去除缺失后）仍满足要求
- [ ] 样本量计算方法和假设已记录

---

## 返回 Router

← [上一步: Sub-Skill 05](05-select-predictors.md)  
← [返回主指南](../clinical-prediction-model-development-zh.md)  
→ [进入下一步: Sub-Skill 07](07-handle-missing-data.md)

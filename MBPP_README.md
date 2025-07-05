 # MBPP数据集与Graph方法说明

## 📚 MBPP数据集介绍

### 数据集概述
MBPP (Mostly Basic Python Problems) 是一个Python编程问题数据集，包含974个基础到中等难度的Python编程任务。每个问题都包含问题描述、解决方案代码和测试用例。

### 数据集结构
MBPP数据集包含三个文件：

1. **`mbpp_test.jsonl`** - 测试集 (500个问题)
2. **`mbpp_validate.jsonl`** - 验证集 (90个问题)  
3. **`mbpp_public_test.jsonl`** - 公开测试集 (428个问题)

### 字段说明
每个数据样本包含以下字段：

```json
{
    "source_file": "数据来源文件名",
    "task_id": 123,
    "prompt": "问题描述和函数签名",
    "code": "标准解决方案代码",
    "test_imports": ["需要导入的模块"],
    "test_list": ["测试断言列表"],
    "entry_point": "函数入口点名称",
    "test": "完整的测试函数代码"
}
```

### 示例数据
```json
{
    "source_file": "Benchmark Questions Verification V2.ipynb",
    "task_id": 802,
    "prompt": "Write a python function to count the number of rotations required to generate a sorted array.\n\ndef count_rotation(arr):",
    "code": "def count_rotation(arr):\n    for i in range (1,len(arr)):\n        if (arr[i] < arr[i - 1]):\n            return i\n    return 0",
    "test_imports": [],
    "test_list": [
        "assert count_rotation([3,2,1]) == 1",
        "assert count_rotation([4,5,1,2,3]) == 2"
    ],
    "entry_point": "count_rotation",
    "test": "def check():\n    assert count_rotation([3,2,1]) == 1\n    assert count_rotation([4,5,1,2,3]) == 2\n    ..."
}
```

## 🔄 Graph方法详解

### 方法概述
Graph方法是AFlow框架中针对代码生成任务优化的工作流，特别适用于MBPP这类编程问题。该方法通过多步骤流程确保生成高质量的Python代码解决方案。

### 核心步骤

#### 1. **多解决方案生成** (Multi-Solution Generation)
```python
solutions = []
for _ in range(3):  # 生成3个不同的解决方案
    solution = await self.custom_code_generate(
        problem=problem, 
        entry_point=entry_point, 
        instruction=prompt_custom.CODE_GENERATE_PROMPT
    )
    solutions.append(solution['response'])
```

**目的**: 通过生成多个候选解决方案，增加找到正确答案的概率。
**过程**: 使用`CustomCodeGenerate`操作符，基于问题描述和函数入口点生成3个独立的Python函数实现。

#### 2. **自一致性集成** (Self-Consistency Ensemble)
```python
best_solution = await self.sc_ensemble(solutions=solutions, problem=problem)
```

**目的**: 从多个候选解决方案中选择最一致、最可靠的方案。
**过程**: `ScEnsemble`操作符分析3个解决方案，通过比较它们的逻辑、实现方式和代码质量，选出最佳方案。

#### 3. **代码测试验证** (Code Testing)
```python
test_result = await self.test(
    problem=problem, 
    solution=best_solution['response'], 
    entry_point=entry_point
)
```

**目的**: 验证选中的解决方案是否能通过所有测试用例。
**过程**: `Test`操作符执行数据集提供的测试用例，检查代码的正确性和健壮性。

#### 4. **错误修复机制** (Error Fixing)
```python
if test_result['result']:
    return test_result['solution'], self.llm.cost_manager.total_cost
else:
    # 如果测试失败，尝试修复解决方案
    fixed_solution = await self.custom(
        input=f"Problem: {problem}\nFailed solution: {best_solution['response']}\nError: {test_result['solution']}", 
        instruction=prompt_custom.FIX_CODE_PROMPT
    )
    return fixed_solution['response'], self.llm.cost_manager.total_cost
```

**目的**: 当解决方案未通过测试时，基于错误信息进行修复。
**过程**: 使用`Custom`操作符，结合原问题、失败的代码和错误信息，生成修复后的解决方案。

### 方法优势

1. **鲁棒性**: 多解决方案生成提高了成功率
2. **质量保证**: 自一致性集成确保选择最佳方案
3. **验证机制**: 实际测试确保代码正确性
4. **自动修复**: 失败时能自动尝试修复

### 操作符详解

- **`CustomCodeGenerate`**: 专门用于代码生成的操作符，理解编程语言语法和最佳实践
- **`ScEnsemble`**: 自一致性集成操作符，比较多个解决方案并选择最佳的
- **`Test`**: 代码测试操作符，执行测试用例并报告结果
- **`Custom`**: 通用操作符，可以处理各种自定义任务（如错误修复）

## 🚀 复现实验

### 环境准备
```bash
# 确保已安装依赖
pip install -r requirements.txt

# 配置LLM API密钥
# 编辑 config/config2.yaml 文件
```

### 运行实验
```bash
# 使用验证集运行实验
python reproduce_mbpp.py --dataset validate --model gpt-4o-mini

# 使用测试集运行实验
python reproduce_mbpp.py --dataset test --model gpt-4o-mini

# 限制样本数量（用于快速测试）
python reproduce_mbpp.py --dataset validate --model gpt-4o-mini --num_samples 10

# 指定输出文件
python reproduce_mbpp.py --dataset validate --model gpt-4o-mini --output results/my_experiment.csv
```

### 参数说明
- `--dataset`: 选择数据集类型 (`test`, `validate`, `public_test`)
- `--model`: LLM模型名称 (默认: `gpt-4o-mini`)
- `--num_samples`: 限制处理的样本数量
- `--output`: 指定输出CSV文件路径

### 实验输出
实验将生成包含以下列的CSV文件：
- `inputs`: 输入问题描述
- `prediction`: 模型生成的解决方案
- `expected_output`: 期望输出（包含测试结果和标准解决方案）
- `score`: 得分 (1.0表示通过，0.0表示失败)
- `cost`: API调用成本

## 📊 实验结果分析

### 参考结果
根据提供的实验文件 `0.83578_20240928_235319.csv`：
- **准确率**: 约83.58%
- **数据集**: 可能使用了MBPP公开测试集的部分数据
- **模型**: 使用了与gpt-4o-mini相当的模型

### 评估指标
1. **准确率**: 通过测试用例的问题比例
2. **平均成本**: 每个问题的平均API调用成本
3. **成功率**: 无错误完成处理的问题比例

### 性能优化建议
1. **提示工程**: 优化代码生成的提示词
2. **错误分析**: 分析失败案例，改进修复策略
3. **模型选择**: 尝试不同的LLM模型
4. **后处理**: 添加代码格式化和优化步骤

## 🔧 自定义实验

### 修改Graph工作流
可以通过修改 `data/results/results/MBPP/graphs_test/round_14/graph.py` 来自定义工作流：

```python
# 示例：增加代码优化步骤
async def __call__(self, problem: str, entry_point: str):
    # ... 现有步骤 ...
    
    # 添加代码优化步骤
    if test_result['result']:
        optimized_solution = await self.custom(
            input=f"Problem: {problem}\nWorking solution: {test_result['solution']}", 
            instruction="Optimize this code for better performance and readability"
        )
        return optimized_solution['response'], self.llm.cost_manager.total_cost
```

### 添加新的数据集
参考 `benchmarks/mbpp.py` 的实现，可以添加自定义的编程问题数据集。

## 🎯 最佳实践

1. **小批量测试**: 先用少量样本测试工作流是否正常
2. **成本监控**: 关注API调用成本，避免超出预算
3. **错误日志**: 保存详细的错误日志用于分析
4. **结果备份**: 及时备份实验结果
5. **版本控制**: 记录使用的模型版本和配置

## ❓ 常见问题

### Q: 如何更换使用的LLM模型？
A: 修改 `config/config2.yaml` 中的模型配置，或使用 `--model` 参数指定。

### Q: 实验中断后如何恢复？
A: 目前需要重新运行。可以考虑实现检查点机制来支持断点续传。

### Q: 如何提高准确率？
A: 可以尝试：
- 优化提示词
- 增加生成的解决方案数量
- 使用更强的LLM模型
- 添加代码检查步骤

### Q: 数据集在哪里下载？
A: 数据集文件应该位于 `data/datasets/` 目录下。如果缺失，请检查项目的数据下载脚本。
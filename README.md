# 多RAG系统评估框架

本项目是一个综合性的RAG（检索增强生成）系统评估平台，集成了QACG数据生成、DICE评估和RAGAS评估三大核心功能，为8种不同配置的RAG系统提供科学严谨的性能对比和排名。

## 🎯 核心特性

- ✅ **QACG数据生成**: 基于LlamaIndex的RAG系统，为8种配置组合生成标准化的QACG四元组测试数据
- ✅ **DICE简化版评估框架**: 基于passage粒度的检索-证据双通道评估，集成soft win机制，提供科学严谨的系统对比
- ✅ **RAGAS评估系统**: 基于RAGAS框架的指标化评估，支持多种评估指标的加权综合评分
- ✅ **知识库独立性**: 每个RAG系统独立处理JSONL数据，确保不同策略产生不同结果
- ✅ **真实模型集成**: 使用HuggingFace和Ollama的真实模型，不是mock数据
- ✅ **智能缓存机制**: 自动缓存处理结果，避免重复计算
- ✅ **全面配置对比**: 2×2×2=8种配置组合，体现明显性能差异

## 🏗️ 系统架构

### 三大核心模块

本项目采用模块化设计，包含三个核心功能模块：

1. **QACG数据生成模块**: 基于LlamaIndex构建8种不同配置的RAG系统，生成标准化的测试数据
2. **DICE简化版评估模块**: 实现基于passage粒度的检索-证据双通道评估框架，集成soft win机制和Elo排名系统
3. **RAGAS评估模块**: 基于RAGAS框架的指标化评估，提供多维度系统性能评分

### 核心改进

根据改进方案，本项目已实现：

1. **独立知识库处理**: 每个RAG系统对JSONL中的每行文本分别做chunking和embedding
2. **专属向量存储**: 每个系统使用独立的Chroma向量数据库
3. **智能缓存系统**: 基于配置自动缓存和复用处理结果
4. **真实模型集成**: 使用LlamaIndex API调用真实的embedding和LLM模型
5. **多评估方法**: 支持DICE和RAGAS两种不同的评估方法，满足不同评估需求

### RAG系统配置 (2×2×2 = 8种组合)

1. **Embedding模型** (2种):
   - `bge-large-zh`: BAAI/bge-large-zh-v1.5 (中文大模型)
   - `bge-small-zh`: BAAI/bge-small-zh-v1.5 (中文小模型)

2. **Chunking策略** (2种):
   - `chunk_256`: 基于256字符长度的分块
   - `chunk_512`: 基于512字符长度的分块

3. **LLM模型** (2种):
   - `qwen2.5`: 通义千问2.5 (7B版本) - 高性能主力模型
   - `qwen2.5-mini`: 通义千问2.5 (0.5B版本) - 超轻量对比模型

### 生成的8个RAG系统

1. `bge-large-zh_chunk_256_qwen2.5` - **最高性能组合** (大模型+大embedding+256分块)
2. `bge-large-zh_chunk_256_qwen2.5-mini` - 大embedding+超小LLM+256分块
3. `bge-large-zh_chunk_512_qwen2.5` - 512分块+高性能LLM
4. `bge-large-zh_chunk_512_qwen2.5-mini` - 512分块+超小LLM
5. `bge-small-zh_chunk_256_qwen2.5` - 小embedding+高性能LLM+256分块
6. `bge-small-zh_chunk_256_qwen2.5-mini` - **最轻量组合** (小embedding+超小LLM+256分块)
7. `bge-small-zh_chunk_512_qwen2.5` - 小embedding+512分块+高性能LLM
8. `bge-small-zh_chunk_512_qwen2.5-mini` - **基线组合** (全小模型+512分块)

## API Key配置说明

### ⚠️ 需要配置的API Key位置：

1. **如果使用OpenAI模型**:
   - 文件: `src/rag_systems/llamaindex_rag.py` 第85-90行
   - 设置环境变量: `OPENAI_API_KEY`
   - 或在代码中直接配置API Key

2. **如果使用Ollama本地模型**:
   - 无需API Key
   - 需要本地安装Ollama服务
   - 下载所需模型: `qwen2.5:7b` 和 `qwen2.5:0.5b`

3. **HuggingFace模型**:
   - 无需API Key（免费使用）
   - 首次运行会自动下载模型到 `./models` 目录

## 安装和配置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置Ollama（推荐方式）

```bash
# 安装Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 下载模型
ollama pull qwen2.5:7b
ollama pull qwen2.5:0.5b

# 启动服务
ollama serve
```

### 3. 配置OpenAI（可选）

如果选择使用OpenAI模型，需要设置API Key：

```bash
export OPENAI_API_KEY="your-api-key-here"
```

或在 `src/rag_systems/llamaindex_rag.py` 中直接配置：

```python
self.llm_model = OpenAI(
    model=self.config.llm_model.replace("openai-", ""),
    temperature=self.config.temperature,
    max_tokens=self.config.max_tokens,
    api_key="your-api-key-here"  # 在这里添加API Key
)
```

## 使用方法

### 🚀 完整工作流程

#### 1. 生成QACG测试数据

首先为8个RAG系统生成标准化的测试数据：

```bash
# 小规模测试（推荐）
python generate_qacg.py --num_questions 5

# 完整生成
python generate_qacg.py --num_questions 20

# 自定义参数
python generate_qacg.py \
    --jsonl_path dice/知识源.jsonl \
    --output_dir qacg_output \
    --num_questions 15 \
    --llm_model qwen2.5:7b
```

#### 2. 运行DICE简化版评估

使用DICE简化版框架进行系统间的两两对比评估：

```bash
# 场景A: 八系统锦标赛（瑞士轮系统）
python dice_simplified_main.py --scenario tournament

# 场景B: 单系统vs锦标赛排名基线
python dice_simplified_main.py --scenario baseline --target_system bge-large-zh_chunk_256_qwen2.5

# 自定义参数
python dice_simplified_main.py --scenario tournament --max_questions 50 --max_workers 4
```

#### 3. 运行RAGAS评估

使用RAGAS框架进行指标化评估：

```bash
# 评估所有系统
python dice_ragas_main.py

# 指定输入和输出目录
python dice_ragas_main.py --input_dir qacg_output --output_dir ragas_dice_output

# 自定义评估指标
python dice_ragas_main.py --metrics answer_relevancy faithfulness context_precision
```

### 🔧 辅助工具

#### 验证知识库独立性

```bash
python verify_kb_independence.py
```

#### 测试模型可用性

```bash
python test_models.py
```

#### 使用OpenAI模型（可选）

```bash
python generate_qacg.py \
    --llm_model openai-gpt-3.5-turbo \
    --num_questions 10
```

## 参数说明

- `--jsonl_path`: 知识库JSONL文件路径（默认: dice/知识源.jsonl）
- `--output_dir`: 输出目录（默认: qacg_output）
- `--num_questions`: 每个RAG系统生成的问题数量（默认: 10）
- `--llm_model`: 用于生成问题的LLM模型（默认: qwen2.5:7b）

## 输出格式

每个RAG系统会生成一个JSON文件，包含QACG四元组：

```json
[
  {
    "question": "这件事发生在什么时间？",
    "rag_answer": "根据文本内容生成的答案",
    "context": ["检索到的相关文档片段"],
    "groundtruth": "ground truth答案",
    "expected_answer": "标准答案",
    "metadata": {
      "system_name": "bge-large-zh_chunk_256_qwen2.5",
      "embedding_model": "bge-large-zh",
      "llm_model": "qwen2.5",
      "chunking_strategy": "chunk_256",
      "retrieval_top_k": 5,
      "question_id": "q_1",
      "generated_at": "..."
    }
  }
]
```

## 故障排除

### 1. 模型下载失败
- 检查网络连接
- 设置HuggingFace镜像源
- 手动下载模型到 `./models` 目录

### 2. Ollama连接失败
- 确保Ollama服务已启动: `ollama serve`
- 检查模型是否已下载: `ollama list`
- 检查端口11434是否可用
- Qwen2.5:0.5b模型较小(~400MB)，下载速度较快

### 3. 内存不足
- 使用更小的模型 (bge-small-zh + qwen2.5-mini)
- 减少 `num_questions` 参数  
- 增加系统内存或使用云服务器
- Qwen2.5:0.5b仅需约1GB内存，适合低配置机器

### 4. API调用失败
- 检查API Key配置
- 确认网络连接
- 检查API调用频率限制

## 模型选择说明

### LLM模型对比
- **Qwen2.5-7B**: 阿里通义千问标准版本，7B参数，在中文理解和生成方面表现优秀
- **Qwen2.5-0.5B**: 阿里通义千问超轻量版本，仅0.5B参数，性能相对较弱但运行速度快

### 为什么选择这两个模型？
1. **参数规模巨大差异**: 7B vs 0.5B，14倍的参数差异，能明显体现模型容量对RAG效果的影响
2. **相同架构**: 同为Qwen2.5架构，排除架构差异，纯粹对比模型规模影响
3. **训练数据一致**: 使用相同的训练语料和优化目标，仅在模型大小上有差异
4. **显著性能梯度**: 7B版本在几乎所有任务上都显著优于0.5B版本，差异明显
5. **资源消耗对比**: 0.5B模型资源消耗极低，7B模型相对较高，体现效率vs性能权衡

## 性能优化建议

1. **使用本地模型**: Ollama + BGE模型，无需API Key且运行稳定
2. **批量处理**: 一次性生成所有系统的数据
3. **缓存机制**: 向量存储会自动缓存到 `./chroma_db/` 目录
4. **并行处理**: 可以修改代码支持多系统并行生成

## 📊 知识库独立性验证

### 验证结果
运行 `python verify_kb_independence.py` 可以验证：

- ✅ **独立处理**: 每个RAG系统独立加载和处理JSONL数据
- ✅ **专属存储**: 每个系统有独立的向量数据库路径 (`./chroma_db/{system_name}`)
- ✅ **智能缓存**: 处理结果缓存到 `./knowledge_cache/{system_name}`
- ✅ **策略差异**: 不同chunking策略生成不同数量和内容的文档块

### 文件结构
```
dice/
├── chroma_db/                          # 向量数据库（每个系统独立）
│   ├── bge-large-zh_sentence_qwen2.5/
│   ├── bge-small-zh_semantic_qwen2.5-mini/
│   └── ...
├── knowledge_cache/                    # 知识库缓存（每个系统独立）
│   ├── bge-large-zh_sentence_qwen2.5/
│   └── ...
├── qacg_output/                       # QACG四元组输出
│   ├── qacg_bge-large-zh_chunk_256_qwen2.5.json
│   ├── qacg_bge-large-zh_chunk_512_qwen2.5.json
│   └── ...
└── models/                           # HuggingFace模型缓存
```

## 🎯 预期性能差异

基于模型规模和策略的差异，预期性能梯度：

### 📊 模型性能排序（预期）
1. **bge-large-zh + chunk_512 + qwen2.5(7B)** - 最优性能
2. **bge-large-zh + chunk_256 + qwen2.5(7B)** - 大embedding+小分块
3. **bge-small-zh + chunk_512 + qwen2.5(7B)** - 轻量embedding+大分块
4. **bge-large-zh + chunk_512 + qwen2.5-mini(0.5B)** - 大embedding+小LLM
5. **bge-small-zh + chunk_256 + qwen2.5-mini(0.5B)** - 全轻量组合

### 🔍 关键差异维度
- **答案质量**: 7B模型 >> 0.5B模型
- **语义理解**: BGE-Large >> BGE-Small
- **分块策略**: 512字符分块提供更多上下文，256字符分块更精确
- **处理速度**: 小模型更快，大模型更准确

## 📁 项目文件说明

### 主要入口文件
- `generate_qacg.py` - QACG数据生成主脚本
- `dice_simplified_main.py` - DICE简化版评估主脚本（推荐使用）
- `dice_main.py` - DICE原版评估脚本（已弃用）
- `dice_ragas_main.py` - RAGAS评估主脚本

### 核心模块
- `src/qacg_generator.py` - QACG生成器核心实现
- `src/rag_systems/llamaindex_rag.py` - 基于LlamaIndex的RAG系统
- `src/rag_systems/base_rag.py` - RAG系统基础抽象类
- `src/dice/dice_simplified.py` - DICE简化版评估核心实现
- `src/dice/local_pairwise_judge.py` - 本地DeepSeek-R1判决器
- `src/ragas/ragas_dice_core.py` - RAGAS评估核心实现



### 数据文件
- `dice/知识源.jsonl` - 原始知识库数据
- `requirements.txt` - 项目依赖

### 输出目录
- `qacg_output/` - QACG生成结果
- `dice_simplified_output/` - DICE简化版评估结果
- `ragas_dice_output/` - RAGAS评估结果
- `chroma_db/` - 向量数据库存储
- `knowledge_cache/` - 知识库缓存

---

# 📊 RAGAS 评估系统

## 概述

**RAGAS（Retrieval-Augmented Generation Assessment）** 是一个基于指标化评估的RAG系统评估框架。与DICE的两两对比不同，RAGAS对每个系统单独评分，然后根据综合得分进行排名。

## 🔬 核心特性

### 1. **指标化评估**
- 基于预定义的评估指标对系统进行量化评分
- 支持多种评估维度的综合评估
- 提供标准化的评分体系

### 2. **多维度评估指标**
- **answer_relevancy** (25%): 答案相关性 - 评估生成答案与问题的相关程度
- **context_precision** (20%): 上下文精确度 - 评估检索到的上下文与问题的精确匹配度  
- **context_recall** (20%): 上下文召回率 - 评估检索到的上下文是否包含回答问题所需的信息
- **faithfulness** (20%): 忠实度 - 评估生成答案与提供上下文的一致性
- **answer_correctness** (15%): 答案正确性 - 评估答案与标准答案的匹配度

### 3. **灵活配置**
- 支持自定义评估指标组合
- 可调整指标权重
- 支持多种LLM和嵌入模型

### 4. **高效评估**
- 支持批量处理
- 并发评估优化
- 智能缓存机制

## 🚀 RAGAS使用方法

### 基础命令

```bash
# 评估所有系统
python dice_ragas_main.py

# 指定输入和输出目录
python dice_ragas_main.py --input_dir qacg_output --output_dir ragas_dice_output

# 只评估特定系统
python dice_ragas_main.py --target_system bge-large-zh_chunk_512_qwen2.5
```

### 高级配置

```bash
# 自定义评估指标
python dice_ragas_main.py --metrics answer_relevancy faithfulness context_precision

# 调整性能参数
python dice_ragas_main.py --batch_size 10 --max_workers 2

# 使用不同的模型
python dice_ragas_main.py --llm_model gpt-4 --embeddings_model text-embedding-ada-002
```

## 📈 RAGAS输出结果

### 1. **系统排名**
- 基于综合得分的系统排名
- 各指标详细得分
- 置信区间统计

### 2. **评估报告**
```
ragas_dice_output/
├── ragas_results.json     # 完整评估结果
├── ragas_summary.md       # 汇总报告
└── individual_scores.json # 各系统详细得分
```

### 3. **可视化图表**
- 系统性能雷达图
- 指标对比柱状图
- 排名变化趋势图

---

# 🎯 DICE简化版评估系统

## 概述

**DICE简化版** 是基于原版DICE框架的优化版本，专注于passage粒度的检索-证据双通道评估，集成了创新的soft win机制，提供更精准和高效的RAG系统对比评估。

## 🔬 核心创新

### 1. **Passage粒度专注评估**
- 专注于检索证据与回答质量的综合评估
- 简化评估流程，提高评估效率
- 基于DeepSeek-R1的深度思考判决

### 2. **Soft Win机制**
- **Hard Win**: 当概率差距≥0.1时，胜者得1分，败者得0分
- **Soft Win**: 当概率差距<0.1时，使用概率分布作为得分
- 保留判决的"强度"信息，避免简单0.5处理
- 更准确地反映系统间的细微差异

### 3. **检索-证据双通道判决**
- 将检索证据显式纳入评估过程
- 四元组比较：`(q, D_A, a_A)` vs `(q, D_B, a_B)`
- 天然区分"回答质量差"vs"检索错误"

### 4. **瑞士轮锦标赛系统**
- 智能配对：每轮选择Elo最接近的未对战过的两队
- 4轮比赛，每轮4场，共16场比赛
- 动态Elo更新，实时反映实力变化

### 5. **本地DeepSeek-R1判决器**
- 基于本地DeepSeek-R1-8B模型
- 支持深度思考模式和直接输出模式
- 强制A/B/T三选项输出，确保判决准确性

## 🚀 DICE简化版使用方法

### 场景A: 八系统锦标赛

```bash
# 标准锦标赛（推荐）
python dice_simplified_main.py --scenario tournament

# 自定义参数
python dice_simplified_main.py --scenario tournament --max_questions 50 --max_workers 4

# 指定输出目录
python dice_simplified_main.py --scenario tournament --output_dir my_tournament_results
```

### 场景B: 单系统vs锦标赛排名基线

```bash
# 与锦标赛排名基线对比
python dice_simplified_main.py --scenario baseline --target_system bge-large-zh_chunk_256_qwen2.5

# 指定锦标赛报告路径
python dice_simplified_main.py --scenario baseline \
  --target_system bge-large-zh_chunk_256_qwen2.5 \
  --tournament_report_path dice_simplified_output/tournament_report.md
```

### 高级配置

```bash
# 自定义LLM模型和参数
python dice_simplified_main.py --scenario tournament \
  --llm_model deepseek-chat \
  --max_questions 70 \
  --max_workers 4 \
  --batch_size 8
```

## 📈 DICE简化版输出结果

### 1. **锦标赛结果**
- 最终排名和Elo分数
- 瑞士轮比赛过程记录
- Soft Win统计信息
- 失败模式聚类分析

### 2. **基线对比结果**
- 与锦标赛排名的对比分析
- 详细QACG对比数据
- 胜率和置信度统计

### 3. **文件输出**
```
dice_simplified_output/
├── tournament_result.json     # 锦标赛完整结果
├── tournament_report.md       # 锦标赛汇总报告
├── baseline_comparison.json   # 基线对比结果
├── qacg_detailed_comparisons.json # 详细QACG对比数据
└── baseline_report.md         # 基线对比报告
```

## 📊 评估示例

### 锦标赛结果示例
```
🏆 DICE简化版锦标赛结果:
  1. bge-large-zh_chunk_512_qwen2.5: 1520.3 🥇
  2. bge-large-zh_chunk_256_qwen2.5: 1508.7 🥈
  3. bge-small-zh_chunk_512_qwen2.5: 1495.2 🥉
  4. bge-small-zh_chunk_256_qwen2.5: 1488.9

Soft Win统计:
- Hard Wins: 45场 (A硬胜: 23, B硬胜: 22)
- Soft Wins: 31场 (A软胜: 16, B软胜: 15)
- Ties: 8场
```

### 判决示例
```
🏆 判决: A soft wins
📈 Logits: A=2.34, B=2.18, T=1.95
📊 概率: A=0.45, B=0.38, T=0.17
🔥 概率差距: 0.07 (Soft win)
🎯 得分: A=0.52, B=0.48
💭 理由: 系统A在检索证据的完整性上略优于系统B...
```

## 🚀 完整工作流程

### 方法一：DICE简化版评估流程

1. **生成QACG数据**:
   ```bash
   python generate_qacg.py --num_questions 70
   ```

2. **运行DICE简化版锦标赛**:
   ```bash
   python dice_simplified_main.py --scenario tournament
   ```

3. **查看锦标赛结果**:
   ```bash
   cat dice_simplified_output/tournament_report.md
   ```

4. **运行基线对比**:
   ```bash
   python dice_simplified_main.py --scenario baseline --target_system bge-large-zh_chunk_256_qwen2.5
   ```

### 方法二：RAGAS评估流程

1. **生成QACG数据**:
   ```bash
   python generate_qacg.py --num_questions 70
   ```

2. **运行RAGAS评估**:
   ```bash
   python dice_ragas_main.py --input_dir qacg_output
   ```

3. **查看RAGAS结果**:
   ```bash
   cat ragas_dice_output/ragas_summary.md
   ```

### 方法三：混合评估流程

1. **生成QACG数据**:
   ```bash
   python generate_qacg.py --num_questions 70
   ```

2. **运行DICE简化版锦标赛**:
   ```bash
   python dice_simplified_main.py --scenario tournament
   ```

3. **运行RAGAS评估**:
   ```bash
   python dice_ragas_main.py --input_dir qacg_output
   ```

4. **对比两种评估结果**:
   ```bash
   # 查看DICE简化版排名
   cat dice_simplified_output/tournament_report.md
   
   # 查看RAGAS排名
   cat ragas_dice_output/ragas_summary.md
   ```

## 📊 评估方法对比

| 特性 | DICE简化版评估 | RAGAS评估 |
|------|----------|-----------|
| **评估方式** | 瑞士轮锦标赛，Elo排名 | 单系统评分，综合排名 |
| **评估维度** | Passage粒度+Soft Win机制 | 五指标加权评估 |
| **评估深度** | 深度语义对比+概率分布 | 标准化指标评分 |
| **适用场景** | 系统性能对比 | 系统性能量化 |
| **计算复杂度** | 中等（16场比赛） | 较低（O(n)） |
| **结果解释** | 相对性能差异+置信度 | 绝对性能得分 |
| **统计严谨性** | 高（Soft Win+置信区间） | 中（加权平均） |
| **创新特性** | Soft Win机制，本地DeepSeek-R1判决 | 标准化指标评估 |

## 🎯 推荐使用场景

### 使用DICE简化版评估的场景：
- 需要深入了解系统间的性能差异
- 要求统计严谨的对比结果
- 关注检索和生成质量的细粒度分析
- 需要Soft Win机制处理细微差异
- 希望使用本地DeepSeek-R1模型进行判决
- 需要瑞士轮锦标赛式的公平对比

### 使用RAGAS评估的场景：
- 需要标准化的系统性能评分
- 要求快速获得系统排名
- 关注特定指标的量化评估
- 需要与其他RAGAS评估结果对比
- 希望使用在线API模型进行评估

## 🆕 Soft Win机制详解

### 核心思想
传统的两两对比评估中，当两个系统性能接近时，简单的"胜/负/平"判决会丢失重要的细微差异信息。Soft Win机制通过概率分布保留这些信息。

### 工作原理
1. **DeepSeek-R1判决**: 模型输出A/B/T三个选项的概率分布
2. **概率差距判断**: 
   - 差距≥0.1 → Hard Win（明确胜负）
   - 差距<0.1 → Soft Win（细微差异）
3. **得分计算**:
   - Hard Win: 胜者1分，败者0分
   - Soft Win: 使用概率分布作为得分，T的概率按A/B比例分配

### 优势
- **保留细微差异**: 不丢失接近系统间的差异信息
- **更准确排名**: 基于概率分布的排名更精确
- **统计严谨**: 基于真实模型输出的概率分布
- **可解释性**: 每个判决都有明确的概率依据 
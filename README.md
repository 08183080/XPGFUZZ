# XPGFUZZ 

## 项目概述

XPGFUZZ是一个基于大语言模型（LLM）增强的网络协议模糊测试工具，集成了多个先进的模糊测试技术。


### XPGfuzz
XPGFUZZ是项目的核心创新模块，位于`xpgfuzz/protocol_seeds`目录中，包含两个关键组件：

#### 1. 种子丰富模块 (RAG + ReAct智能体)
- **核心文件**: `seed_enricher_with_rag_agent.py`
- **功能**: 使用RAG（检索增强生成）和ReAct智能体技术，智能分析协议种子中缺失的命令，并基于权威文档生成高质量的种子变体
- **特点**: 
  - 支持多种网络协议（SMTP、FTP、RTSP、SIP等）
  - 基于ChromaDB的向量检索系统
  - 智能命令补全和语法验证

#### 2. 细粒度语法模板约束变异模块
- **核心目录**: `template_mutate/`
- **核心文件**: `finegrained_mutator.py`
- **功能**: 实现细粒度的协议语法模板变异，支持多种数据类型的精确变异
- **特点**:
  - 支持整数、字符串、路径、IP地址、端口等多种数据类型的变异
  - 基于LLM的协议语法提取和解析
  - 结构化的变异策略和攻击向量生成

该项目配置在[ProfuzzBench](https://github.com/profuzzbench/profuzzbench)框架内，这是一个广泛使用的网络协议状态模糊测试基准测试平台。

## 文件夹结构

```
XPGFUZZ/
├── aflnet: AFLNet的修改版本，输出状态和状态转换
├── analyze.sh: 分析脚本
├── benchmark: ProfuzzBench的修改版本，仅包含基于文本的协议
├── clean.sh: 清理脚本
├── ChatAFL: chatafl的代码
├── deps.sh: 安装依赖的脚本，执行时会要求输入密码
├── README: 本文件
├── run.sh: 在目标上运行模糊测试器并收集数据的执行脚本
├── setup.sh: 设置docker镜像的准备脚本
└── xpgfuzz: XPGfuzz核心模块目录
    └── protocol_seeds: 协议种子处理核心模块
        ├── seed_enricher_with_rag_agent.py: RAG+ReAct智能体种子丰富器
        ├── template_mutate/: 细粒度语法模板约束变异模块
        │   ├── finegrained_mutator.py: 细粒度变异器核心实现
        │   ├── chatafl_simulation.py: ChatAFL模拟器
        │   ├── input_seeds/: 输入种子文件目录
        │   └── output_results/: 变异结果输出目录
        ├── chroma_db_protocols/: ChromaDB向量数据库存储
        ├── protocol_deepwiiki/: 协议deepwiki文档知识库
        ├── in-*/: 各协议的原始种子文件目录
        └── *.dict: 各协议的字典文件
```

## XPGfuzz核心模块详解

### 1. 种子丰富模块 (RAG + ReAct智能体)

XPGfuzz的种子丰富模块位于`xpgfuzz/protocol_seeds/seed_enricher_with_rag_agent.py`，是一个基于RAG（检索增强生成）和ReAct智能体技术的协议种子增强系统。

#### 核心功能
- **智能命令分析**: 自动分析协议种子中缺失的命令类型
- **权威文档检索**: 基于ChromaDB向量数据库检索相关协议文档
- **智能种子生成**: 使用ReAct智能体生成符合协议规范的种子变体
- **多协议支持**: 支持SMTP、FTP、RTSP、SIP等多种网络协议

#### 使用方法
```bash
python seed_enricher_with_rag_agent.py \
    --protocol-name "Exim SMTP" \
    --commands "HELO,EHLO,MAIL FROM,RCPT TO,DATA,QUIT" \
    --input-dir "in-smtp" \
    --output-dir "out-smtp" \
    --variations 10 \
    --collection-name "smtp_protocol"
```

#### 技术特点
- **RAG技术**: 结合向量检索和生成模型，确保生成的种子基于权威文档
- **ReAct智能体**: 使用思考-行动-观察循环，智能决策命令插入位置
- **类型感知变异**: 根据数据类型（整数、字符串、路径等）进行精确变异
- **批量处理**: 支持大规模种子文件的批量处理

### 2. 细粒度语法模板约束变异模块

细粒度语法模板约束变异模块位于`xpgfuzz/protocol_seeds/template_mutate/`目录，实现了基于LLM的协议语法提取和细粒度语法模板约束变异。

#### 核心组件

##### 2.1 细粒度变异器 (`finegrained_mutator.py`)
支持多种数据类型的精确变异：
- **整数变异**: 边界值测试、溢出攻击、负数攻击
- **字符串变异**: SQL注入、XSS攻击、路径遍历
- **IP地址变异**: 无效IP、特殊地址、IPv6支持
- **端口变异**: 边界端口、无效端口、特权端口
- **枚举变异**: 无效值、大小写变换、特殊字符

##### 2.2 协议语法提取器 (`GrammarExtractor`)
- **LLM驱动**: 使用GPT模型提取协议语法模板
- **结构化解析**: 将LLM输出解析为可执行的语法规则
- **正则表达式生成**: 自动生成匹配和变异用的正则表达式
- **一致性验证**: 多次调用LLM确保语法提取的一致性


#### 使用方法
```bash
cd xpgfuzz/protocol_seeds/template_mutate/
python finegrained_mutator.py
```

#### 配置说明
在`finegrained_mutator.py`的`main()`函数中可以配置：
- `protocol_name`: 目标协议名称
- `base_input_dir`: 输入种子目录
- `base_output_dir`: 输出结果目录
- `openai_api_key`: OpenAI API密钥

#### 输出结果
- `mutations/`: 包含所有变异后的种子文件
- `protocol-grammars/grammars.json`: 提取的协议语法规则
- 变异文件命名格式: `{原文件名}_mut_{序号}.raw`


#### 模板驱动的细粒度变异集成

在`xpgfuzz/afl-fuzz.c`调用基于模板、带类型约束的细粒度变异：

```bash
# Linux/macOS
export XPG_TEMPLATE_MUTATE=1

# Windows PowerShell
$Env:XPG_TEMPLATE_MUTATE = 1
```

要求：
- 语法文件位于`out/<protocol_name>/protocol-grammars/grammars.json`（可由内置LLM提取流程生成）。
- Python 可用：Unix 使用 `python3`；Windows 使用 `py` 启动器。

说明：
- 该外部变异以约5%的概率在每次havoc堆叠操作中触发，以控制开销。
- 临时文件写入到 `out/<session>/template_mutations/`，使用后会清理。

### 3. 协议知识库

XPGfuzz集成了丰富的协议知识库：
- **ChromaDB向量数据库**: 存储协议文档的向量表示
- **DeepWiki知识库**: 包含Exim、ProFTPD、Kamailio、Live555等协议的deepwiki文档

### 4. 支持的协议

XPGfuzz目前支持以下网络协议：
- **SMTP**: Exim邮件服务器
- **FTP**: ProFTPD、PureFTPD、LightFTPD
- **RTSP**: Live555流媒体服务器
- **SIP**: Kamailio SIP服务器

每个协议都有对应的种子文件、字典文件和知识库文档。


## 安装和使用

### 安装依赖

需要`Docker`、`Bash`、`Python3`以及`pandas`和`matplotlib`库。我们提供了一个辅助脚本`deps.sh`来运行所需的步骤，确保提供所有依赖：

```bash
./deps.sh
```

### 准备Docker镜像 [约40分钟]

运行以下命令来设置所有docker镜像，包括带有所有模糊测试器的目标：

```bash
KEY=<OPENAI_API_KEY> ./setup.sh
```

### 运行实验

使用`run.sh`脚本来运行实验。命令如下：

```bash
 ./run.sh <container_number> <fuzzed_time> <subjects> <fuzzers>
```

其中`container_number`指定创建多少个容器来在特定目标上运行单个模糊测试器（每个容器在一个目标上运行一个模糊测试器）。`fuzzed_time`表示模糊测试时间（分钟）。`subjects`是测试目标列表，`fuzzers`是用于模糊测试目标的模糊测试器列表。例如，命令（`run.sh 1 5 pure-ftpd xpgfuzz`）将为模糊测试器XPGFUZZ创建一个容器，对目标pure-ftpd进行5分钟的模糊测试。简而言之，可以通过在目标和模糊测试器列表的位置使用`all`来执行所有模糊测试器和所有目标。

当脚本完成时，在`benchmark`目录中将创建一个文件夹`result-<目标名称>`，包含每次运行的模糊测试结果。

### 分析结果

`analyze.sh`脚本用于分析数据并构建图表，说明每个目标上模糊测试器的平均代码和状态覆盖率随时间的变化。脚本使用以下命令执行：

```bash
./analyze.sh <subjects> <fuzzed_time> 
```

脚本接受2个参数 - `subjects`是测试目标列表，`fuzzed_time`是要分析的运行持续时间。注意，第二个参数是可选的，脚本默认假设执行时间为1440分钟，等于1天。例如，命令（`analyze.sh exim 240`）将分析exim目标执行结果的前4小时。

执行完成后，脚本将通过构建csv文件来处理归档，包含随时间覆盖的分支、状态和状态转换数量。此外，这些csv文件将被处理成PNG文件，这些文件是图表，说明每个目标上模糊测试器的平均代码和状态覆盖率随时间的变化（`cov_over_time...`用于代码和分支覆盖率，`state_over_time...`用于状态和状态转换覆盖率）。所有这些信息都移动到根目录中带有时间戳的`res_<目标名称>`文件夹中。

### 清理

当工件的评估完成时，运行`clean.sh`脚本将确保此目录中只留下剩余文件：

```bash
./clean.sh
```

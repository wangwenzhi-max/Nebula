# Nebula Graph LangChain集成项目

这是一个使用LangChain调用Nebula Graph图数据库的Python工程代码。该项目提供了完整的集成方案，包括数据库连接、查询执行和自然语言问答功能。

## 项目特性

- 完整的Nebula Graph连接和操作封装
- LangChain Chain集成，支持自然语言问答
- 支持图数据库的CRUD操作
- 提供丰富的示例代码
- 易于扩展和定制

## 项目结构

```
nebula/
├── src/                          # 源代码目录
│   ├── __init__.py              # 模块初始化
│   ├── nebula_graph_store.py    # Nebula Graph存储类
│   └── nebula_chain.py          # LangChain Chain集成
├── examples/                     # 示例代码目录
│   ├── basic_usage.py           # 基础使用示例
│   ├── langchain_integration.py # LangChain集成示例
│   └── setup_nebula.py          # 环境设置脚本
├── requirements.txt              # Python依赖
└── README.md                    # 项目文档
```

## 安装步骤

### 1. 安装Python依赖

```bash
pip install -r requirements.txt
```

### 2. 安装Nebula Graph

请参考[Nebula Graph官方文档](https://docs.nebula-graph.io/)安装和配置Nebula Graph数据库。

确保Nebula Graph服务正在运行在 `127.0.0.1:9669`。

### 3. 设置测试环境

运行设置脚本创建测试图空间和示例数据：

```bash
python examples/setup_nebula.py
```

或者使用重置命令（先清理再设置）：

```bash
python examples/setup_nebula.py --action reset
```

## 快速开始

### 基础使用

```python
from src.nebula_graph_store import NebulaGraphStore

# 创建连接
graph_store = NebulaGraphStore(
    addresses="127.0.0.1:9669",
    user="root",
    password="nebula",
    space="test"
)

# 执行查询
results = graph_store.execute_query("MATCH (v:person) RETURN v.name, v.age")
print(results)

# 获取schema信息
schema = graph_store.get_schema()
print(f"点类型: {schema['vertex_types']}")

# 关闭连接
graph_store.close()
```

### LangChain问答

```python
from src.nebula_graph_store import NebulaGraphStore
from src.nebula_chain import NebulaGraphQAChain
from langchain.llms import OpenAI

# 创建图存储
graph_store = NebulaGraphStore(
    addresses="127.0.0.1:9669",
    user="root",
    password="nebula",
    space="test"
)

# 创建LLM
llm = OpenAI(temperature=0)

# 创建问答链
qa_chain = NebulaGraphQAChain(llm=llm, graph_store=graph_store)

# 提问
answer = qa_chain.run("数据库中有哪些人？")
print(answer)

graph_store.close()
```

## 核心模块说明

### NebulaGraphStore

图数据库存储类，提供以下功能：

- 连接管理：自动建立和管理Nebula Graph连接
- 查询执行：执行nGQL查询并返回格式化结果
- Schema信息：获取图数据库的结构信息
- 数据操作：添加节点和边
- 图数据获取：获取完整的图结构数据

主要方法：
- `execute_query(query)`: 执行nGQL查询
- `get_schema()`: 获取schema信息
- `get_directed_graph()`: 获取图数据
- `add_node(tag, vid, properties)`: 添加节点
- `add_edge(edge_type, src_vid, dst_vid, properties)`: 添加边

### NebulaGraphChain

LangChain集成类，提供以下功能：

- 自然语言到nGQL的转换
- 查询执行和结果处理
- 自然语言回答生成
- Chain接口实现

主要方法：
- `run(question)`: 执行问答
- `from_llm(llm, graph_store)`: 从LLM创建Chain实例

### NebulaGraphQAChain

简化的问答链，提供更简单的接口：

- 自动化的问答流程
- 内置的查询生成和回答生成
- 可配置的详细输出

## 示例代码

### 1. 基础操作示例

```bash
python examples/basic_usage.py
```

演示内容：
- 基本连接和查询
- 添加节点和边
- 复杂查询
- 错误处理

### 2. LangChain集成示例

```bash
python examples/langchain_integration.py
```

演示内容：
- 使用模拟LLM的问答链
- 使用OpenAI的问答链
- 高级Chain用法
- 批量问答
- 自定义Chain

### 3. 环境设置脚本

```bash
# 设置测试环境
python examples/setup_nebula.py --action setup

# 清理测试环境
python examples/setup_nebula.py --action cleanup

# 重置测试环境
python examples/setup_nebula.py --action reset
```

## 配置说明

### Nebula Graph连接配置

```python
NebulaGraphStore(
    addresses="127.0.0.1:9669",  # 数据库地址
    user="root",                   # 用户名
    password="nebula",             # 密码
    space="test"                   # 图空间名称
)
```

### 环境变量（可选）

可以创建 `.env` 文件来配置环境变量：

```
NEBULA_ADDRESSES=127.0.0.1:9669
NEBULA_USER=root
NEBULA_PASSWORD=nebula
NEBULA_SPACE=test
OPENAI_API_KEY=your_openai_api_key
```

## LLM集成

项目支持多种LLM提供商：

### OpenAI

```python
from langchain.llms import OpenAI

llm = OpenAI(temperature=0)
```

### 其他LLM

```python
from langchain.llms import HuggingFacePipeline
from langchain.llms import Anthropic
from langchain.llms import Cohere

# 使用HuggingFace
llm = HuggingFacePipeline.from_model_id(
    model_id="gpt2",
    task="text-generation"
)
```

## 常见问题

### 1. 连接失败

确保Nebula Graph服务正在运行，并且地址、端口、用户名、密码配置正确。

### 2. 查询语法错误

检查nGQL语法是否正确，参考[Nebula Graph查询语言文档](https://docs.nebula-graph.io/3.5.0/3.ngql-guide/1.nGQL-overview/)。

### 3. 图空间不存在

运行设置脚本创建图空间：
```bash
python examples/setup_nebula.py
```

### 4. LLM API错误

确保已设置正确的API密钥，并且网络连接正常。

## 扩展开发

### 自定义Chain

可以基于提供的基类创建自定义Chain：

```python
from langchain.chains.base import Chain
from src.nebula_graph_store import NebulaGraphStore

class CustomChain(Chain):
    graph_store: NebulaGraphStore

    def _call(self, inputs):
        # 自定义逻辑
        pass
```

### 自定义查询生成

重写 `_generate_ngql_query` 方法实现自定义查询生成逻辑。

## 性能优化建议

1. 使用连接池管理连接
2. 批量操作时合理使用事务
3. 复杂查询考虑添加索引
4. 大数据查询使用分页

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request！

## 联系方式

如有问题，请提交Issue或联系项目维护者。

## 更新日志

### v0.1.0
- 初始版本发布
- 实现基础的Nebula Graph连接和查询功能
- 实现LangChain Chain集成
- 提供完整的示例代码

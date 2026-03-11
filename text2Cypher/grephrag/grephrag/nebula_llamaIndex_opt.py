import os
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
from contextlib import contextmanager
from tenacity import retry, stop_after_attempt, wait_exponential

from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config as Config_nebula
from llama_index.graph_stores.nebula import NebulaGraphStore
from llama_index.core import StorageContext, Settings
from llama_index.readers.wikipedia import WikipediaReader
from llama_index.core import KnowledgeGraphIndex
from llama_index.core.query_engine import KnowledgeGraphQueryEngine
from llama_index.core.prompts import PromptTemplate

from utils import CustomOpenAILLM, CustomOpenAIEmbedding, Config

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from typing import List, Dict, Any
from dataclasses import dataclass, field

@dataclass
class TagDef:
    name: str
    properties: List[str]          # 如 ["name", "age"]

@dataclass
class EdgeDef:
    name: str
    properties: List[str]           # 如 ["start_year", "end_year"]
    src_tag: str
    dst_tag: str

@dataclass
class ApplicationConfig:
    nebula_host: str = "10.78.212.222"
    nebula_port: int = 9669
    nebula_user: str = "root"
    nebula_password: str = "nebula"
    space_name: str = "demo_basketballplayer"
    llm_temperature: float = 0.1
    llm_timeout: int = 60

    # 图谱结构定义
    tags: List[TagDef] = field(default_factory=lambda: [
        TagDef("player", ["name", "age"]),
        TagDef("team", ["name"])
    ])
    edges: List[EdgeDef] = field(default_factory=lambda: [
        EdgeDef("serve", ["start_year", "end_year"], "player", "team"),
        EdgeDef("follow", ["degree"], "player", "player")
    ])

    @property
    def nebula_address(self) -> str:
        return f"{self.nebula_host}:{self.nebula_port}"

    def get_schema_description(self) -> str:
        """生成用于 Prompt 的 Schema 描述文本"""
        lines = ["**标签 (Tag) 及其属性**："]
        for tag in self.tags:
            props = ", ".join(tag.properties)
            lines.append(f"- {tag.name}: {props}")
        lines.append("\n**边类型 (Edge Type) 及其属性**：")
        for edge in self.edges:
            props = ", ".join(edge.properties)
            lines.append(f"- {edge.name}: {props} (起点: {edge.src_tag}, 终点: {edge.dst_tag})")
        return "\n".join(lines)

class NebulaManager:
    def __init__(self, config: ApplicationConfig):
        self.config = config
        self._connection_pool: Optional[ConnectionPool] = None
        self._graph_store: Optional[NebulaGraphStore] = None

    def _init_connection_pool(self):
        if not self._connection_pool:
            nebula_config = Config_nebula()
            self._connection_pool = ConnectionPool()
            self._connection_pool.init([(self.config.nebula_host, self.config.nebula_port)], nebula_config)

    @property
    def graph_store(self) -> NebulaGraphStore:
        if not self._graph_store:
            self._init_connection_pool()
            # 将标签/边定义转换为 LlamaIndex 要求的格式
            edge_types = [e.name for e in self.config.edges]
            # 注意：rel_prop_names 应为每个边类型的属性列表，以逗号分隔的字符串
            rel_prop_names = [",".join(e.properties) for e in self.config.edges]
            tags = [t.name for t in self.config.tags]
            tag_prop_names = [",".join(t.properties) for t in self.config.tags]

            os.environ.update({
                "NEBULA_USER": self.config.nebula_user,
                "NEBULA_PASSWORD": self.config.nebula_password,
                "NEBULA_ADDRESS": self.config.nebula_address
            })

            self._graph_store = NebulaGraphStore(
                space_name=self.config.space_name,
                edge_types=edge_types,
                rel_prop_names=rel_prop_names,
                tags=tags,
                tag_prop_names=tag_prop_names,
            )
        return self._graph_store

    def close(self):
        if self._connection_pool:
            self._connection_pool.close()
            self._connection_pool = None

class KnowledgeGraphBuilder:
    def __init__(self, nebula_manager: NebulaManager, config: ApplicationConfig):
        self.nebula_manager = nebula_manager
        self.config = config
        self._setup_llm()

    def _setup_llm(self):
        app_config = Config()  # 假设这是您的外部配置类
        Settings.llm = CustomOpenAILLM(
            api_url=app_config.llm_api_url,
            model=app_config.llm_model_name,
            api_key="EMPTY",
            temperature=self.config.llm_temperature,
            timeout=self.config.llm_timeout
        )
        Settings.embed_model = CustomOpenAIEmbedding(app_config)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def build_from_wikipedia(self, pages: list) -> KnowledgeGraphIndex:
        try:
            reader = WikipediaReader()
            documents = reader.load_data(pages=pages, auto_suggest=False)
            logger.info(f"加载了 {len(documents)} 个维基百科文档")

            storage_context = StorageContext.from_defaults(
                graph_store=self.nebula_manager.graph_store
            )

            kg_index = KnowledgeGraphIndex.from_documents(
                documents,
                storage_context=storage_context,
                max_triplets_per_chunk=10,
                space_name=self.config.space_name,
                edge_types=[e.name for e in self.config.edges],
                rel_prop_names=[",".join(e.properties) for e in self.config.edges],
                tags=[t.name for t in self.config.tags],
                max_knowledge_sequence=15
            )
            logger.info("知识图谱构建成功")
            return kg_index
        except Exception as e:
            logger.error(f"构建知识图谱失败: {e}")
            raise

    def verify_graph_data(self):
        """验证图谱数据，执行几条基础查询"""
        queries = [
            "SHOW TAGS;",
            "SHOW EDGES;",
            f"USE {self.config.space_name}; MATCH (n) RETURN n LIMIT 10;",
            f"USE {self.config.space_name}; MATCH ()-[r]->() RETURN r LIMIT 10;"
        ]
        for query in queries:
            try:
                result = self.nebula_manager.graph_store.query(query)
                logger.info(f"查询: {query}\n结果: {result}")
            except Exception as e:
                logger.error(f"查询失败: {query}, 错误: {e}")

class QueryEngine:
    def __init__(self, nebula_manager: NebulaManager):
        self.nebula_manager = nebula_manager
        self._engine = None
        self._prompt = self._build_prompt()

    def _build_prompt(self) -> PromptTemplate:
        schema_desc = self.nebula_manager.config.get_schema_description()
        space_name = self.nebula_manager.config.space_name

        template = f"""
        你是一个精通 Nebula Graph 的图数据库查询专家，严格遵循 nGQL 语法。你的任务是根据用户的自然语言问题，生成准确、可执行的 nGQL 查询语句。

        ### 图数据库 Schema 信息
        图空间名称：{space_name}
        {schema_desc}

        ### 输出格式要求
        - 仅输出 **纯 nGQL 查询语句**（一条或多条，以分号分隔），不要包含任何解释、注释、Markdown 代码块标记或额外文本。
        - 如果问题无法根据给定 Schema 生成查询，请输出：`-- 无法生成查询`

        ### 核心语法要求（严格遵守）

        #### 1. 基本结构
        - **USE 语句**：所有查询必须以 `USE {space_name};` 开头（单独一行）。
        - **查询语言**：必须使用 **nGQL**，禁止使用 openCypher 中 nGQL 不支持的语法（如 `UNWIND`、`FOREACH`、`CREATE`、`MERGE` 等）
        - **禁止在 WHERE 中使用子查询**（如 `WHERE p.age == (MATCH ... RETURN ...)`）。若需要基于聚合值过滤，请使用 `WITH` 分步完成，或拆分为多个独立查询

        #### 2. 查询构造
        - **匹配方式**：优先使用 `MATCH` 语句进行模式匹配。避免使用 `LOOKUP` 和 `GO`，除非明确要求使用索引或不支持 `MATCH` 的场景。
        - **变量与属性访问**：访问节点或边的属性时，必须**显式指定标签名**（若节点有多个标签，使用点号连接，如 `v.player.name`），不能简写为 `v.name`。
        - **标签匹配**：可以在 `MATCH` 中指定标签，如 `(v:player)`，也可以在 `WHERE` 中使用 `label(v) == "player"`。
        - **相等比较**：使用 `==` 表示相等，禁止使用单个 `=`（在 nGQL 中 `=` 用于赋值）。
        - **字符串匹配**：使用 `CONTAINS` 关键字（区分大小写），如 `v.player.name CONTAINS "James"`。若需忽略大小写，可使用 `v.player.name LOWER CONTAINS LOWER("James")`。
        - **聚合值跨 MATCH 限制：禁止在第一个 MATCH 中计算全局聚合（如 max()、min()、avg()），然后在后续 MATCH 的 WHERE 中直接使用这些聚合值进行过滤。这种写法会导致优化器错误。

        #### 3. 存在性检查（关键区别）
        - **检查属性是否存在**：使用 `EXISTS(v.player.age)`。
        - **检查边模式是否存在**：直接在 `WHERE` 子句中使用模式作为布尔表达式，**禁止使用 `EXISTS()` 包裹模式**。
            - ✅ 正确：`MATCH (p:player) WHERE (p)-[:follow]->(:player) RETURN p`
            - ✅ 正确：`MATCH (p:player) WHERE NOT (p)-[:follow]->() RETURN p`
            - ❌ 错误：`MATCH (p:player) WHERE EXISTS((p)-[:follow]->(:player)) RETURN p`
        - **检查边是否存在并绑定变量**：也可以结合 `OPTIONAL MATCH` 和条件判断，但优先使用上述简洁写法。

        #### 4. 排序与聚合
        - **ORDER BY**：只能对 `RETURN` 子句中的列名或别名进行排序。如果返回的是属性路径（如 `e.end_year`），必须为其指定别名，然后使用别名排序。不能直接使用属性路径或表达式排序。
        - ✅ 正确：`RETURN e.end_year AS year, count(*) AS cnt ORDER BY year`
        - ❌ 错误：`RETURN e.end_year, count(*) AS cnt ORDER BY e.end_year`
        - ✅ 正确（字符串长度排序）：`RETURN p.player.name AS name, size(p.player.name) AS len ORDER BY len DESC`
        - ❌ 错误：`RETURN p.player.name ORDER BY size(p.player.name) DESC`
        - **聚合函数**：支持 `COUNT(*)`、`COUNT(DISTINCT v)`、`AVG()`、`SUM()`、`MAX()`、`MIN()` 等。如需分组，非聚合字段会自动作为分组键，无需额外 `GROUP BY`。
        - **避免在同一个查询中先计算全局聚合（如 max、min），然后在后续 MATCH 的 WHERE 中直接使用这些聚合值。这可能导致优化器错误。
        - **聚合值跨 MATCH 限制**：禁止在第一个 MATCH 中计算全局聚合（如 max、min），然后在后续 MATCH 的 WHERE 中直接使用这些聚合值。这种写法会导致优化器错误。应拆分为多个独立查询，或用 `UNION` 合并（需测试）。

        #### 5. 路径与多跳查询
        - 使用变长路径表示法：`(p:player)-[:follow*1..3]->(f:player)` 表示 1 到 3 跳的 follow 关系。
        - 返回路径变量：`MATCH path = (p:player)-[:follow*1..3]->(f:player) RETURN path`

        #### 6. 注意事项
        - **大小写敏感**：标签名、边类型名、属性名在 nGQL 中默认区分大小写，请严格参照 Schema。
        - **索引提醒**：`MATCH` 中使用的属性过滤条件（如 `p.player.age > 30`）通常需要对应索引才能高效执行。若查询性能不佳，建议检查索引。但生成查询时，请按逻辑正常生成，不必回避。
        - **多语句支持**：可以在一个请求中发送多条 nGQL 语句，用分号分隔。但一般情况下只需生成一条主要查询。

        ### 常见错误与正确写法对照

        | 错误写法 (Cypher 风格) | 正确写法 (nGQL 风格) |
        |------------------------|----------------------|
        | `WHERE EXISTS((p)-[:follow]->())` | `WHERE (p)-[:follow]->()` |
        | `WHERE p.age > 30` (属性无标签) | `WHERE p.player.age > 30` |
        | `WHERE p.name = "Yao Ming"` | `WHERE p.player.name == "Yao Ming"` |
        | `RETURN ... ORDER BY size(p.name) DESC` | `RETURN p.player.name, size(p.player.name) AS len ORDER BY len DESC` |
        

        ### 示例查询（严格遵循上述规则）

        **用户问题**：姚明效力过哪些球队？
        **nGQL 查询**：
        USE {space_name};
        MATCH (p:player)-[e:serve]->(t:team)
        WHERE p.player.name == "Yao Ming"
        RETURN t.team.name AS team_name, e.start_year, e.end_year;

        **用户问题**：哪些球员没有被任何人关注？
        **nGQL 查询**：
        USE {space_name};
        MATCH (p:player)
        WHERE NOT (p)<-[:follow]-(:player)
        RETURN p.player.name;

        **用户问题**：年龄大于30岁的球员有多少？
        **nGQL 查询**：
        USE {space_name};
        MATCH (p:player)
        WHERE p.player.age > 30
        RETURN count(*) AS player_count;

        **用户问题**：名字最长的球员（按字符长度）？
        **nGQL 查询**：
        USE {space_name};
        MATCH (p:player)
        RETURN p.player.name AS name, size(p.player.name) AS name_length
        ORDER BY name_length DESC LIMIT 1;

        **用户问题**：关注关系超过2跳的所有球员对？
        **nGQL 查询**：
        USE {space_name};
        MATCH path = (p1:player)-[:follow*2..3]->(p2:player)
        RETURN p1.player.name AS follower, p2.player.name AS followee, length(path) AS hops;

        **用户问题**：{{query_str}}
        **nGQL 查询**：
            """
        return PromptTemplate(template)

    @property
    def engine(self) -> KnowledgeGraphQueryEngine:
        if not self._engine:
            storage_context = StorageContext.from_defaults(
                graph_store=self.nebula_manager.graph_store
            )
            self._engine = KnowledgeGraphQueryEngine(
                storage_context=storage_context,
                llm=Settings.llm,
                graph_query_synthesis_prompt=self._prompt,
                verbose=True
            )
        return self._engine

    def query(self, question: str) -> str:
        try:
            logger.info(f"执行查询: {question}")
            response = self.engine.query(question)
            return str(response)
        except Exception as e:
            logger.error(f"查询失败: {e}", exc_info=True)
            return f"查询失败: {e}"

# 主程序
def main():
    config = ApplicationConfig()
    nebula_manager = NebulaManager(config)

    try:
        # 可选：构建图谱（注释掉，按需启用）
        builder = KnowledgeGraphBuilder(nebula_manager, config)
        # kg_index = builder.build_from_wikipedia(pages=['Guardians of the Galaxy Vol. 3'])
        # builder.verify_graph_data()

        query_engine = QueryEngine(nebula_manager)
        questions = [
            "查找年龄最大和年龄最小的球员所效力的球队，并列出球队名",
        ]

        for question in questions:
            response = query_engine.query(question)
            print(f"\n问题: {question}")
            print(f"回答: {response}")
    finally:
        nebula_manager.close()  # 确保释放连接池

if __name__ == "__main__":
    main()
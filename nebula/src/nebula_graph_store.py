"""
Nebula Graph存储模块
提供与Nebula Graph数据库的连接和基础操作功能
"""

from typing import Any, Dict, List, Optional, Union
from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config
from langchain.graphs import GraphStore
from langchain.pydantic_v1 import BaseModel, Field


class NebulaConfig(BaseModel):
    """Nebula Graph连接配置"""
    addresses: List[str] = Field(default=["127.0.0.1:9669"], description="Nebula Graph地址列表")
    user: str = Field(default="root", description="用户名")
    password: str = Field(default="nebula", description="密码")
    space: str = Field(default="default", description="图空间名称")
    config: Optional[Config] = None

    class Config:
        """配置类"""
        arbitrary_types_allowed = True


class NebulaGraphStore(GraphStore):
    """
    Nebula Graph存储类
    实现LangChain的GraphStore接口，提供图数据库操作功能
    """

    def __init__(
        self,
        addresses: Union[str, List[str]],
        user: str = "root",
        password: str = "nebula",
        space: str = "default",
        config: Optional[Config] = None,
    ):
        """
        初始化Nebula Graph存储

        Args:
            addresses: Nebula Graph地址，可以是单个字符串或地址列表
            user: 用户名
            password: 密码
            space: 图空间名称
            config: 连接配置对象
        """
        # 处理地址格式
        if isinstance(addresses, str):
            self.addresses = [address.strip() for address in addresses.split(",")]
        else:
            self.addresses = addresses

        self.user = user
        self.password = password
        self.space = space
        self.config = config or Config()

        # 初始化连接池
        self.connection_pool = ConnectionPool()
        self.session = None

        self._initialize_connection()

    def _initialize_connection(self):
        """初始化数据库连接"""
        try:
            # 连接到Nebula Graph
            for address in self.addresses:
                host, port = address.split(":") if ":" in address else (address, "9669")
                self.connection_pool.init([(host, int(port))], self.config)

            # 创建会话
            self.session = self.connection_pool.get_session(self.user, self.password)

            # 使用指定的图空间
            self.session.execute(f"USE {self.space}")

            print(f"成功连接到Nebula Graph: {self.addresses}, 空间: {self.space}")

        except Exception as e:
            raise ConnectionError(f"连接Nebula Graph失败: {str(e)}")

    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        """
        执行nGQL查询

        Args:
            query: nGQL查询语句

        Returns:
            查询结果列表，每个元素是一行数据的字典
        """
        try:
            result = self.session.execute(query)

            if not result.is_succeeded():
                raise Exception(f"查询执行失败: {result.error_msg()}")

            # 将结果转换为字典列表
            columns = result.keys()
            data = []

            for record in result:
                row = {}
                for idx, value in enumerate(record.values()):
                    column_name = columns[idx] if idx < len(columns) else f"col_{idx}"
                    row[column_name] = self._convert_value(value)
                data.append(row)

            return data

        except Exception as e:
            raise Exception(f"执行查询时出错: {str(e)}")

    def _convert_value(self, value: Any) -> Any:
        """
        转换Nebula Graph的值类型为Python原生类型

        Args:
            value: Nebula Graph的值

        Returns:
            转换后的Python值
        """
        if value is None:
            return None

        # 处理不同类型的值
        value_type = type(value).__name__

        if value_type == "ValueWrapper":
            return self._convert_value(value.as_node())
        elif value_type == "Vertex":
            # 处理顶点
            return {
                "id": value.get_id().as_string(),
                "tags": [tag for tag in value.tags()],
                "props": {prop: self._convert_value(value.properties(prop)) for prop in value.tags()}
            }
        elif value_type == "Edge":
            # 处理边
            return {
                "src": value.src_vid.as_string(),
                "dst": value.dst_vid.as_string(),
                "type": value.edge_name,
                "rank": value.ranking,
                "props": self._convert_value(value.properties)
            }
        elif value_type == "Relationship":
            # 处理关系
            return {
                "start": value.start_node_id,
                "end": value.end_node_id,
                "type": value.type,
                "properties": value.properties
            }
        elif value_type == "Node":
            # 处理节点
            return {
                "id": value.id,
                "labels": value.labels,
                "properties": value.properties
            }
        elif isinstance(value, (int, float, str, bool)):
            return value
        elif value_type == "list":
            return [self._convert_value(item) for item in value]
        elif value_type == "set":
            return [self._convert_value(item) for item in value]
        elif value_type == "map":
            return {k: self._convert_value(v) for k, v in value.items()}
        else:
            return str(value)

    def get_schema(self) -> Dict[str, Any]:
        """
        获取图数据库的schema信息

        Returns:
            包含点类型、边类型和属性的字典
        """
        schema = {
            "vertex_types": [],
            "edge_types": [],
            "properties": {}
        }

        try:
            # 获取点类型
            tag_result = self.session.execute("SHOW TAGS")
            if tag_result.is_succeeded():
                for tag in tag_result:
                    tag_name = tag.values()[0].as_string()
                    schema["vertex_types"].append(tag_name)

                    # 获取tag的属性
                    desc_result = self.session.execute(f"DESCRIBE TAG {tag_name}")
                    if desc_result.is_succeeded():
                        schema["properties"][tag_name] = []
                        for prop in desc_result:
                            schema["properties"][tag_name].append({
                                "name": prop.values()[0].as_string(),
                                "type": prop.values()[1].as_string()
                            })

            # 获取边类型
            edge_result = self.session.execute("SHOW EDGES")
            if edge_result.is_succeeded():
                for edge in edge_result:
                    edge_name = edge.values()[0].as_string()
                    schema["edge_types"].append(edge_name)

                    # 获取edge的属性
                    desc_result = self.session.execute(f"DESCRIBE EDGE {edge_name}")
                    if desc_result.is_succeeded():
                        schema["properties"][edge_name] = []
                        for prop in desc_result:
                            schema["properties"][edge_name].append({
                                "name": prop.values()[0].as_string(),
                                "type": prop.values()[1].as_string()
                            })

        except Exception as e:
            print(f"获取schema时出错: {str(e)}")

        return schema

    def get_directed_graph(self) -> Dict[str, Any]:
        """
        获取有向图数据

        Returns:
            包含节点和边的字典
        """
        graph_data = {
            "nodes": [],
            "edges": []
        }

        try:
            # 获取所有节点
            nodes_query = "MATCH (v) RETURN v LIMIT 1000"
            nodes_data = self.execute_query(nodes_query)
            graph_data["nodes"] = nodes_data

            # 获取所有边
            edges_query = "MATCH (v)-[e]->(u) RETURN e LIMIT 1000"
            edges_data = self.execute_query(edges_query)
            graph_data["edges"] = edges_data

        except Exception as e:
            print(f"获取图数据时出错: {str(e)}")

        return graph_data

    def add_node(self, tag: str, vid: str, properties: Optional[Dict[str, Any]] = None) -> bool:
        """
        添加节点

        Args:
            tag: 节点标签
            vid: 节点ID
            properties: 节点属性

        Returns:
            是否成功
        """
        try:
            if properties:
                props_str = ", ".join([f"{k}: {self._format_value(v)}" for k, v in properties.items()])
                query = f"INSERT VERTEX {tag}({', '.join(properties.keys())}) VALUES '{vid}': ({props_str})"
            else:
                query = f"INSERT VERTEX {tag} VALUES '{vid}': ()"

            result = self.session.execute(query)
            return result.is_succeeded()

        except Exception as e:
            print(f"添加节点时出错: {str(e)}")
            return False

    def add_edge(
        self,
        edge_type: str,
        src_vid: str,
        dst_vid: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        添加边

        Args:
            edge_type: 边类型
            src_vid: 源节点ID
            dst_vid: 目标节点ID
            properties: 边属性

        Returns:
            是否成功
        """
        try:
            if properties:
                props_str = ", ".join([f"{k}: {self._format_value(v)}" for k, v in properties.items()])
                query = f"INSERT EDGE {edge_type}({', '.join(properties.keys())}) VALUES '{src_vid}'->'{dst_vid}': ({props_str})"
            else:
                query = f"INSERT EDGE {edge_type} VALUES '{src_vid}'->'{dst_vid}': ()"

            result = self.session.execute(query)
            return result.is_succeeded()

        except Exception as e:
            print(f"添加边时出错: {str(e)}")
            return False

    def _format_value(self, value: Any) -> str:
        """
        格式化值为nGQL格式

        Args:
            value: 待格式化的值

        Returns:
            格式化后的字符串
        """
        if value is None:
            return "NULL"
        elif isinstance(value, str):
            return f'"{value}"'
        elif isinstance(value, bool):
            return str(value).lower()
        else:
            return str(value)

    def close(self):
        """关闭数据库连接"""
        try:
            if self.session:
                self.session.release()
            if self.connection_pool:
                self.connection_pool.close()
            print("Nebula Graph连接已关闭")
        except Exception as e:
            print(f"关闭连接时出错: {str(e)}")

    def __del__(self):
        """析构函数，确保连接被正确关闭"""
        self.close()

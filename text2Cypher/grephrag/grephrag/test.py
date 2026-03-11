"""
Text to Cypher Query Generator for Nebula Graph
使用 LangChain 和自部署的 Qwen3-32B 模型将自然语言转换为 Cypher 查询
"""

from langchain_openai import ChatOpenAI
from langchain_community.graphs import NebulaGraph
from langchain.chains import GraphCypherQAChain
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import re
from typing import Optional, Dict, List


class NebulaText2Cypher:
    """
    将自然语言文本转换为 Nebula Graph Cypher 查询的类
    """
    
    def __init__(
        self,
        nebula_host: str = "localhost",
        nebula_port: int = 9669,
        nebula_username: str = "root",
        nebula_password: str = "nebula",
        nebula_space: str = "your_space",
        llm_model_url: str = "http://localhost:8080/v1",
        llm_model_name: str = "Qwen3-32B"
    ):
        """
        初始化 Text2Cypher 转换器
        
        Args:
            nebula_host: Nebula Graph 主机地址
            nebula_port: Nebula Graph 端口
            nebula_username: Nebula Graph 用户名
            nebula_password: Nebula Graph 密码
            nebula_space: Nebula Graph 空间名称
            llm_model_url: 自部署模型的 API URL
            llm_model_name: 模型名称
        """
        self.nebula_host = nebula_host
        self.nebula_port = nebula_port
        self.nebula_space = nebula_space
        
        # 初始化 Nebula Graph
        self.graph = NebulaGraph(
            space=nebula_space,
            username=nebula_username,
            password=nebula_password,
            address=f"{nebula_host}:{nebula_port}",
        )
        
        # 初始化自部署的 LLM
        self.llm = ChatOpenAI(
            model=llm_model_name,
            base_url=llm_model_url,
            temperature=0,
            max_tokens=2048,
        )
        
        # 初始化 QA 链
        self.qa_chain = self._create_qa_chain()
    
    def _create_qa_chain(self) -> GraphCypherQAChain:
        """
        创建 GraphCypherQAChain，利用 LangChain 的 Nebula 支持
        
        Returns:
            GraphCypherQAChain: 配置好的 QA 链
        """
        
        # 自定义 Cypher 生成提示词
        cypher_prompt = PromptTemplate(
            input_variables=["schema", "question"],
            template="""
你是一个 Nebula Graph Cypher 查询专家。
使用给定的图数据库 schema 将用户的自然语言问题转换为可执行的 Cypher 查询。

Schema:
{schema}

问题: {question}

请仅输出 Cypher 查询代码，不要包含其他说明文字。
Cypher 查询:"""
        )
        
        # 创建 GraphCypherQAChain
        chain = GraphCypherQAChain.from_llm(
            llm=self.llm,
            graph=self.graph,
            cypher_prompt=cypher_prompt,
            verbose=True,
            validate_cypher=True,
            return_direct=True,
        )
        
        return chain
    
    def text2cypher(self, question: str) -> Dict:
        """
        将自然语言转换为 Cypher 查询并执行
        
        Args:
            question: 用户的自然语言问题
            
        Returns:
            包含查询结果的字典
        """
        try:
            result = self.qa_chain.invoke({"query": question})
            return {
                "success": True,
                "question": question,
                "result": result,
            }
        except Exception as e:
            return {
                "success": False,
                "question": question,
                "error": str(e),
            }
    
    def get_cypher_only(self, question: str) -> Optional[str]:
        """
        仅获取转换后的 Cypher 查询，不执行查询
        
        Args:
            question: 用户的自然语言问题
            
        Returns:
            生成的 Cypher 查询语句
        """
        
        # 获取图 schema
        schema = self.graph.get_schema
        
        # 创建单独的 Cypher 生成链
        cypher_prompt = PromptTemplate(
            input_variables=["schema", "question"],
            template="""
你是一个 Nebula Graph Cypher 查询专家。
使用给定的图数据库 schema 将用户的自然语言问题转换为可执行的 Cypher 查询。

Schema:
{schema}

问题: {question}

请仅输出 Cypher 查询代码，不要包含其他说明文字。
Cypher 查询:"""
        )
        
        cypher_chain = (
            cypher_prompt
            | self.llm
            | StrOutputParser()
        )
        
        try:
            cypher_query = cypher_chain.invoke({
                "schema": schema,
                "question": question
            })
            return cypher_query.strip()
        except Exception as e:
            print(f"生成 Cypher 查询出错: {e}")
            return None
    
    def execute_cypher(self, cypher_query: str) -> List:
        """
        执行 Cypher 查询
        
        Args:
            cypher_query: Cypher 查询语句
            
        Returns:
            查询结果列表
        """
        try:
            result = self.graph.query(cypher_query)
            return result
        except Exception as e:
            print(f"执行 Cypher 查询出错: {e}")
            return []
    
    def get_graph_schema(self) -> str:
        """
        获取 Nebula Graph 的 schema 信息
        
        Returns:
            Graph schema 字符串
        """
        return self.graph.get_schema
    
    def close(self):
        """
        关闭数据库连接
        """
        self.graph.close()


class NebulaText2CypherAdvanced(NebulaText2Cypher):
    """
    高级版本的 Text2Cypher，支持更多功能
    """
    
    def multi_turn_conversation(self, questions: List[str]) -> List[Dict]:
        """
        支持多轮对话
        
        Args:
            questions: 多个自然语言问题列表
            
        Returns:
            多个查询结果列表
        """
        results = []
        for question in questions:
            result = self.text2cypher(question)
            results.append(result)
        return results
    
    def refine_cypher(self, question: str, cypher_query: str) -> Optional[str]:
        """
        根据反馈优化 Cypher 查询
        
        Args:
            question: 原始问题
            cypher_query: 原始查询
            
        Returns:
            优化后的 Cypher 查询
        """
        refinement_prompt = PromptTemplate(
            input_variables=["question", "cypher", "schema"],
            template="""
你是一个 Nebula Graph Cypher 查询专家。
请检查以下 Cypher 查询是否正确，如果有问题请优化它。

Schema:
{schema}

原始问题: {question}
原始查询: {cypher}

请输出优化后的 Cypher 查询，如果查询没有问题，直接输出原始查询。
仅输出查询代码，不要包含其他说明文字。
优化后的查询:"""
        )
        
        refinement_chain = (
            refinement_prompt
            | self.llm
            | StrOutputParser()
        )
        
        try:
            refined_query = refinement_chain.invoke({
                "question": question,
                "cypher": cypher_query,
                "schema": self.get_graph_schema()
            })
            return refined_query.strip()
        except Exception as e:
            print(f"优化 Cypher 查询出错: {e}")
            return cypher_query


# 使用示例
if __name__ == "__main__":
    
    # 初始化 Text2Cypher
    text2cypher = NebulaText2Cypher(
        nebula_host="localhost",
        nebula_port=9669,
        nebula_username="root",
        nebula_password="nebula",
        nebula_space="your_space_name",  # 请替换为实际的 space 名称
        llm_model_url="http://localhost:8080/v1",
        llm_model_name="Qwen3-32B"
    )
    
    # 示例 1: 直接执行查询
    print("=" * 50)
    print("示例 1: 直接执行查询")
    print("=" * 50)
    result = text2cypher.text2cypher("查找所有的用户")
    print(f"结果: {result}")
    
    # 示例 2: 仅获取 Cypher 查询
    print("\n" + "=" * 50)
    print("示例 2: 仅获取 Cypher 查询")
    print("=" * 50)
    cypher = text2cypher.get_cypher_only("找出所有连接的节点")
    print(f"生成的 Cypher: {cypher}")
    
    # 示例 3: 执行自定义 Cypher 查询
    if cypher:
        print("\n" + "=" * 50)
        print("示例 3: 执行自定义查询")
        print("=" * 50)
        exec_result = text2cypher.execute_cypher(cypher)
        print(f"执行结果: {exec_result}")
    
    # 示例 4: 获取图 schema
    print("\n" + "=" * 50)
    print("示例 4: 获取图 Schema")
    print("=" * 50)
    schema = text2cypher.get_graph_schema()
    print(f"Graph Schema:\n{schema}")
    
    # 关闭连接
    text2cypher.close()
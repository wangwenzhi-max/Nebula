"""
Nebula Graph Chain模块
提供LangChain的Chain集成，实现基于Nebula Graph的问答和对话功能
"""

from typing import Any, Dict, List, Optional, Tuple
from langchain.chains.base import Chain
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.prompts import PromptTemplate
from langchain.pydantic_v1 import BaseModel, Field
from langchain.schema import BasePromptTemplate
from .nebula_graph_store import NebulaGraphStore


class NebulaGraphChainInput(BaseModel):
    """Nebula Graph Chain的输入模型"""
    question: str = Field(description="用户的问题")


class NebulaGraphChain(Chain):
    """
    Nebula Graph Chain类
    实现基于Nebula Graph的问答功能
    """

    graph_store: NebulaGraphStore = Field(description="Nebula Graph存储实例")
    llm: Any = Field(description="语言模型实例")
    prompt: BasePromptTemplate = Field(description="提示模板")
    return_direct: bool = Field(default=False, description="是否直接返回查询结果")
    input_key: str = "question"  #: :meta private:
    output_key: str = "result"  #: :meta private:

    class Config:
        """配置类"""
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """返回输入键列表"""
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """返回输出键列表"""
        return [self.output_key]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """
        执行链的主要逻辑

        Args:
            inputs: 输入字典
            run_manager: 回调管理器

        Returns:
            输出字典
        """
        question = inputs[self.input_key]

        if run_manager:
            run_manager.on_text(f"用户问题: {question}\n", color="blue")

        # 步骤1: 使用LLM将自然语言问题转换为nGQL查询
        schema_info = self._get_schema_info()
        ngql_query = self._generate_ngql_query(question, schema_info)

        if run_manager:
            run_manager.on_text(f"生成的nGQL查询: {ngql_query}\n", color="green")

        # 步骤2: 执行nGQL查询
        try:
            query_results = self.graph_store.execute_query(ngql_query)

            if run_manager:
                run_manager.on_text(f"查询结果: {query_results}\n", color="yellow")

            # 如果没有结果
            if not query_results:
                return {self.output_key: "没有找到相关的数据。"}

            # 步骤3: 使用LLM生成自然语言回答
            answer = self._generate_answer(question, ngql_query, query_results)

            return {self.output_key: answer}

        except Exception as e:
            error_msg = f"查询执行出错: {str(e)}"
            if run_manager:
                run_manager.on_text(error_msg + "\n", color="red")
            return {self.output_key: error_msg}

    def _get_schema_info(self) -> str:
        """
        获取图数据库的schema信息

        Returns:
            Schema信息的字符串表示
        """
        schema = self.graph_store.get_schema()

        info = []
        info.append("图数据库Schema信息:")
        info.append(f"点类型: {', '.join(schema['vertex_types'])}")
        info.append(f"边类型: {', '.join(schema['edge_types'])}")
        info.append("属性详情:")

        for entity, props in schema['properties'].items():
            info.append(f"  {entity}:")
            for prop in props:
                info.append(f"    - {prop['name']}: {prop['type']}")

        return "\n".join(info)

    def _generate_ngql_query(self, question: str, schema_info: str) -> str:
        """
        使用LLM生成nGQL查询

        Args:
            question: 用户问题
            schema_info: Schema信息

        Returns:
            nGQL查询语句
        """
        query_prompt = PromptTemplate(
            template="""你是一个Nebula Graph数据库查询专家。根据以下schema信息和用户问题，生成正确的nGQL查询语句。

{schema_info}

用户问题: {question}

请生成一个nGQL查询语句来回答这个问题。只返回查询语句，不要包含任何解释或额外内容。

注意事项:
1. 使用MATCH语句进行图查询
2. 使用RETURN返回结果
3. 限制结果数量(使用LIMIT)
4. 确保语法正确
5. 如果问题无法用图查询回答，返回一个简单的查询

nGQL查询:""",
            input_variables=["schema_info", "question"]
        )

        prompt_value = query_prompt.format(schema_info=schema_info, question=question)

        # 调用LLM生成查询
        response = self.llm.predict(prompt_value)
        query = response.strip()

        # 清理生成的查询
        if "```" in query:
            query = query.split("```")[1]
        if query.startswith("nGQL:") or query.startswith("nGQL"):
            query = query.split(":", 1)[1].strip()
        if query.startswith("Query:") or query.startswith("Query"):
            query = query.split(":", 1)[1].strip()

        return query

    def _generate_answer(
        self,
        question: str,
        query: str,
        results: List[Dict[str, Any]]
    ) -> str:
        """
        使用LLM基于查询结果生成自然语言回答

        Args:
            question: 用户问题
            query: nGQL查询语句
            results: 查询结果

        Returns:
            自然语言回答
        """
        answer_prompt = PromptTemplate(
            template="""你是一个友好的助手。根据以下查询结果，用自然语言回答用户的问题。

用户问题: {question}

执行的查询: {query}

查询结果: {results}

请基于查询结果给出一个清晰、准确的回答。如果结果为空，说明没有找到相关信息。回答要简洁明了，不要重复查询结果中的技术细节。

回答:""",
            input_variables=["question", "query", "results"]
        )

        prompt_value = answer_prompt.format(
            question=question,
            query=query,
            results=str(results)
        )

        # 调用LLM生成回答
        response = self.llm.predict(prompt_value)
        return response.strip()

    @classmethod
    def from_llm(
        cls,
        llm: Any,
        graph_store: NebulaGraphStore,
        return_direct: bool = False,
        **kwargs: Any,
    ) -> "NebulaGraphChain":
        """
        从LLM和图存储创建Chain实例

        Args:
            llm: 语言模型实例
            graph_store: Nebula Graph存储实例
            return_direct: 是否直接返回查询结果
            **kwargs: 其他参数

        Returns:
            NebulaGraphChain实例
        """
        default_prompt = PromptTemplate(
            template="请基于图数据库信息回答问题: {question}",
            input_variables=["question"]
        )

        return cls(
            llm=llm,
            graph_store=graph_store,
            prompt=default_prompt,
            return_direct=return_direct,
            **kwargs
        )


class NebulaGraphQAChain(Chain):
    """
    Nebula Graph问答链的简化版本
    提供更简单的接口进行图数据库问答
    """

    graph_store: NebulaGraphStore = Field(description="Nebula Graph存储实例")
    llm: Any = Field(description="语言模型实例")
    verbose: bool = Field(default=True, description="是否显示详细信息")
    input_key: str = "question"
    output_key: str = "answer"

    class Config:
        """配置类"""
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """返回输入键列表"""
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """返回输出键列表"""
        return [self.output_key]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """
        执行问答链

        Args:
            inputs: 输入字典
            run_manager: 回调管理器

        Returns:
            输出字典
        """
        question = inputs[self.input_key]

        # 获取schema信息
        schema_info = self._get_schema_info()

        if self.verbose and run_manager:
            run_manager.on_text(f"Schema信息:\n{schema_info}\n\n", color="blue")

        # 生成查询并执行
        query = self._generate_query(question, schema_info)

        if self.verbose and run_manager:
            run_manager.on_text(f"生成的查询: {query}\n", color="green")

        try:
            results = self.graph_store.execute_query(query)

            if self.verbose and run_manager:
                run_manager.on_text(f"查询结果: {results}\n", color="yellow")

            # 生成回答
            answer = self._generate_answer(question, query, results)

            return {self.output_key: answer}

        except Exception as e:
            return {self.output_key: f"查询出错: {str(e)}"}

    def _get_schema_info(self) -> str:
        """获取schema信息"""
        schema = self.graph_store.get_schema()
        return str(schema)

    def _generate_query(self, question: str, schema_info: str) -> str:
        """生成查询语句"""
        prompt = f"""基于以下图数据库schema信息，为问题生成nGQL查询语句：

Schema: {schema_info}

问题: {question}

生成nGQL查询（只返回查询语句）:"""

        return self.llm.predict(prompt).strip()

    def _generate_answer(self, question: str, query: str, results: List[Dict[str, Any]]) -> str:
        """生成回答"""
        prompt = f"""基于查询结果回答问题：

问题: {question}
查询: {query}
结果: {results}

请用自然语言回答:"""

        return self.llm.predict(prompt).strip()

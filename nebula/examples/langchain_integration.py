"""
LangChain集成示例
演示如何使用Nebula Graph Chain进行问答和对话
"""

import os
import sys

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from nebula_graph_store import NebulaGraphStore
from nebula_chain import NebulaGraphChain, NebulaGraphQAChain


def example_qa_chain_with_mock_llm():
    """示例1: 使用模拟LLM的问答链"""
    print("=" * 60)
    print("示例1: 使用模拟LLM的问答链")
    print("=" * 60)

    # 创建图存储
    graph_store = NebulaGraphStore(
        addresses="127.0.0.1:9669",
        user="root",
        password="nebula",
        space="test"
    )

    try:
        # 创建模拟LLM类（实际使用时替换为真实的LLM）
        class MockLLM:
            """模拟语言模型，用于演示目的"""

            def predict(self, prompt: str) -> str:
                """简单的预测方法"""
                if "nGQL" in prompt or "查询" in prompt:
                    return "MATCH (v:person) RETURN v.name, v.age LIMIT 5"
                elif "回答" in prompt:
                    return "根据查询结果，数据库中有几个人物信息，包括他们的名字和年龄。"
                else:
                    return "这是一个模拟的回答"

        llm = MockLLM()

        # 创建问答链
        qa_chain = NebulaGraphQAChain(llm=llm, graph_store=graph_store, verbose=True)

        # 提问
        question = "数据库中有哪些人？"
        print(f"\n问题: {question}")

        result = qa_chain.run(question)
        print(f"回答: {result}")

    finally:
        graph_store.close()


def example_qa_chain_with_openai():
    """示例2: 使用OpenAI的问答链"""
    print("\n" + "=" * 60)
    print("示例2: 使用OpenAI的问答链")
    print("=" * 60)

    try:
        from langchain.llms import OpenAI

        # 创建图存储
        graph_store = NebulaGraphStore(
            addresses="127.0.0.1:9669",
            user="root",
            password="nebula",
            space="test"
        )

        # 创建OpenAI LLM
        # 注意: 需要设置OPENAI_API_KEY环境变量
        llm = OpenAI(temperature=0)

        # 创建问答链
        qa_chain = NebulaGraphQAChain(llm=llm, graph_store=graph_store, verbose=True)

        # 多个问题示例
        questions = [
            "数据库中有哪些人？",
            "张三有哪些朋友？",
            "有多少个朋友关系？"
        ]

        for question in questions:
            print(f"\n问题: {question}")
            result = qa_chain.run(question)
            print(f"回答: {result}")

        graph_store.close()

    except ImportError:
        print("OpenAI库未安装，跳过此示例")
        print("要使用OpenAI，请运行: pip install openai")
    except Exception as e:
        print(f"执行OpenAI示例时出错: {e}")


def example_advanced_chain():
    """示例3: 高级Chain用法"""
    print("\n" + "=" * 60)
    print("示例3: 高级Chain用法")
    print("=" * 60)

    # 创建图存储
    graph_store = NebulaGraphStore(
        addresses="127.0.0.1:9669",
        user="root",
        password="nebula",
        space="test"
    )

    try:
        # 创建模拟LLM
        class MockLLM:
            def predict(self, prompt: str) -> str:
                if "nGQL" in prompt or "查询" in prompt:
                    return "MATCH (v:person)-[e:friend]->(u:person) RETURN v.name, u.name LIMIT 10"
                elif "回答" in prompt:
                    return "数据库中存在多个朋友关系，显示了人物之间的社交网络。"
                return "模拟回答"

        llm = MockLLM()

        # 使用from_llm方法创建链
        chain = NebulaGraphChain.from_llm(
            llm=llm,
            graph_store=graph_store,
            return_direct=False
        )

        # 执行查询
        question = "有哪些朋友关系？"
        print(f"\n问题: {question}")

        result = chain.run({chain.input_key: question})
        print(f"回答: {result[chain.output_key]}")

    finally:
        graph_store.close()


def example_batch_questions():
    """示例4: 批量问答"""
    print("\n" + "=" * 60)
    print("示例4: 批量问答")
    print("=" * 60)

    # 创建图存储
    graph_store = NebulaGraphStore(
        addresses="127.0.0.1:9669",
        user="root",
        password="nebula",
        space="test"
    )

    try:
        # 创建模拟LLM
        class MockLLM:
            def predict(self, prompt: str) -> str:
                if "nGQL" in prompt or "查询" in prompt:
                    return "MATCH (v:person) RETURN v.name AS name, v.age AS age"
                elif "回答" in prompt:
                    return "数据库中存储了多个人物信息，包括他们的姓名和年龄。"
                return "模拟回答"

        llm = MockLLM()

        # 创建问答链
        qa_chain = NebulaGraphQAChain(llm=llm, graph_store=graph_store, verbose=False)

        # 批量问题
        questions = [
            "数据库中有多少条记录？",
            "有哪些人？",
            "统计年龄分布"
        ]

        print(f"\n处理 {len(questions)} 个问题:\n")

        for i, question in enumerate(questions, 1):
            print(f"[{i}/{len(questions)}] 问题: {question}")
            try:
                result = qa_chain.run(question)
                print(f"回答: {result}\n")
            except Exception as e:
                print(f"错误: {e}\n")

    finally:
        graph_store.close()


def example_custom_chain():
    """示例5: 自定义Chain"""
    print("\n" + "=" * 60)
    print("示例5: 自定义Chain")
    print("=" * 60)

    from langchain.chains.base import Chain
    from langchain.callbacks.manager import CallbackManagerForChainRun
    from langchain.pydantic_v1 import Field

    class CustomNebulaChain(Chain):
        """自定义Nebula Chain"""

        graph_store: NebulaGraphStore = Field(...)
        llm: Any = Field(...)

        class Config:
            arbitrary_types_allowed = True

        @property
        def input_keys(self) -> list:
            return ["question"]

        @property
        def output_keys(self) -> list:
            return ["answer", "query", "results"]

        def _call(
            self,
            inputs: dict,
            run_manager: Optional[CallbackManagerForChainRun] = None,
        ) -> dict:
            question = inputs["question"]

            # 生成查询
            query = "MATCH (v:person) RETURN v.name, v.age LIMIT 5"

            # 执行查询
            results = self.graph_store.execute_query(query)

            # 生成回答
            answer = f"根据查询，找到了 {len(results)} 条记录"

            return {
                "answer": answer,
                "query": query,
                "results": results
            }

    # 创建图存储
    graph_store = NebulaGraphStore(
        addresses="127.0.0.1:9669",
        user="root",
        password="nebula",
        space="test"
    )

    try:
        # 创建模拟LLM
        class MockLLM:
            def predict(self, prompt: str) -> str:
                return "模拟回答"

        llm = MockLLM()

        # 创建自定义链
        custom_chain = CustomNebulaChain(graph_store=graph_store, llm=llm)

        # 执行
        question = "查询所有人员信息"
        print(f"问题: {question}")

        result = custom_chain.run(question)
        print(f"回答: {result['answer']}")
        print(f"查询: {result['query']}")
        print(f"结果: {result['results']}")

    finally:
        graph_store.close()


if __name__ == "__main__":
    print("Nebula Graph LangChain集成 - Chain示例\n")

    # 运行各个示例
    try:
        example_qa_chain_with_mock_llm()
        example_qa_chain_with_openai()
        example_advanced_chain()
        example_batch_questions()
        example_custom_chain()

        print("\n" + "=" * 60)
        print("所有示例执行完成!")
        print("=" * 60)

    except Exception as e:
        print(f"\n执行示例时出错: {e}")
        print("\n请确保:")
        print("1. Nebula Graph数据库正在运行")
        print("2. 已创建图空间 'test' 和相应的schema")
        print("3. 连接参数配置正确")
        print("4. 如需使用OpenAI，请设置OPENAI_API_KEY环境变量")

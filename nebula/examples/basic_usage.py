"""
基础使用示例
演示如何使用Nebula Graph LangChain集成进行基本操作
"""

import os
import sys

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from nebula_graph_store import NebulaGraphStore


def example_basic_connection():
    """示例1: 基本连接和查询"""
    print("=" * 60)
    print("示例1: 基本连接和查询")
    print("=" * 60)

    # 创建Nebula Graph连接
    graph_store = NebulaGraphStore(
        addresses="127.0.0.1:9669",  # Nebula Graph地址
        user="root",                   # 用户名
        password="nebula",             # 密码
        space="test"                   # 图空间名称
    )

    try:
        # 获取schema信息
        schema = graph_store.get_schema()
        print("\n图数据库Schema:")
        print(f"点类型: {schema['vertex_types']}")
        print(f"边类型: {schema['edge_types']}")

        # 执行简单的查询
        print("\n执行查询: MATCH (v) RETURN v LIMIT 5")
        results = graph_store.execute_query("MATCH (v) RETURN v LIMIT 5")
        print(f"查询结果: {results}")

        # 获取图数据
        print("\n获取图数据:")
        graph_data = graph_store.get_directed_graph()
        print(f"节点数量: {len(graph_data['nodes'])}")
        print(f"边数量: {len(graph_data['edges'])}")

    finally:
        # 关闭连接
        graph_store.close()


def example_add_data():
    """示例2: 添加节点和边"""
    print("\n" + "=" * 60)
    print("示例2: 添加节点和边")
    print("=" * 60)

    graph_store = NebulaGraphStore(
        addresses="127.0.0.1:9669",
        user="root",
        password="nebula",
        space="test"
    )

    try:
        # 添加节点
        print("\n添加节点...")
        graph_store.add_node("person", "p1", {"name": "张三", "age": 30})
        graph_store.add_node("person", "p2", {"name": "李四", "age": 25})
        graph_store.add_node("person", "p3", {"name": "王五", "age": 28})
        print("节点添加成功")

        # 添加边
        print("\n添加边...")
        graph_store.add_edge("friend", "p1", "p2", {"since": 2020})
        graph_store.add_edge("friend", "p2", "p3", {"since": 2021})
        graph_store.add_edge("friend", "p1", "p3", {"since": 2019})
        print("边添加成功")

        # 验证数据
        print("\n验证添加的数据:")
        results = graph_store.execute_query(
            "MATCH (v:person) RETURN v.name AS name, v.age AS age"
        )
        print(f"人员列表: {results}")

    finally:
        graph_store.close()


def example_complex_query():
    """示例3: 复杂查询"""
    print("\n" + "=" * 60)
    print("示例3: 复杂查询")
    print("=" * 60)

    graph_store = NebulaGraphStore(
        addresses="127.0.0.1:9669",
        user="root",
        password="nebula",
        space="test"
    )

    try:
        # 查询所有朋友关系
        print("\n查询所有朋友关系:")
        query = """
        MATCH (v1:person)-[e:friend]->(v2:person)
        RETURN v1.name AS person1, v2.name AS person2, e.since AS since
        """
        results = graph_store.execute_query(query)
        print(f"朋友关系: {results}")

        # 查询某个人的朋友的朋友
        print("\n查询张三的朋友的朋友:")
        query = """
        MATCH (v1:person {name: '张三'})-[:friend]->(v2:person)-[:friend]->(v3:person)
        RETURN v1.name AS person1, v2.name AS person2, v3.name AS person3
        """
        results = graph_store.execute_query(query)
        print(f"二度朋友关系: {results}")

        # 统计查询
        print("\n统计每个年龄的人数:")
        query = """
        MATCH (v:person)
        RETURN v.age AS age, count(v) AS count
        ORDER BY age
        """
        results = graph_store.execute_query(query)
        print(f"年龄统计: {results}")

    finally:
        graph_store.close()


def example_error_handling():
    """示例4: 错误处理"""
    print("\n" + "=" * 60)
    print("示例4: 错误处理")
    print("=" * 60)

    try:
        # 尝试连接到不存在的地址
        graph_store = NebulaGraphStore(
            addresses="127.0.0.1:9999",  # 错误的端口
            user="root",
            password="nebula",
            space="test"
        )
    except ConnectionError as e:
        print(f"捕获到连接错误: {e}")

    try:
        graph_store = NebulaGraphStore(
            addresses="127.0.0.1:9669",
            user="root",
            password="nebula",
            space="test"
        )

        # 尝试执行错误的查询
        print("\n尝试执行错误的查询:")
        results = graph_store.execute_query("INVALID QUERY")
    except Exception as e:
        print(f"捕获到查询错误: {e}")
    finally:
        if 'graph_store' in locals():
            graph_store.close()


if __name__ == "__main__":
    print("Nebula Graph LangChain集成 - 基础使用示例\n")

    # 运行各个示例
    # 注意: 这些示例需要Nebula Graph数据库正在运行
    # 并且已经创建了相应的图空间和schema

    try:
        example_basic_connection()
        example_add_data()
        example_complex_query()
        example_error_handling()

        print("\n" + "=" * 60)
        print("所有示例执行完成!")
        print("=" * 60)

    except Exception as e:
        print(f"\n执行示例时出错: {e}")
        print("\n请确保:")
        print("1. Nebula Graph数据库正在运行")
        print("2. 已创建图空间 'test' 和相应的schema")
        print("3. 连接参数配置正确")

"""
快速开始示例
演示最简单的使用方法
"""

import os
import sys

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from nebula_graph_store import NebulaGraphStore


def quick_start():
    """快速开始示例"""
    print("Nebula Graph LangChain集成 - 快速开始\n")

    # 1. 连接到Nebula Graph
    print("1. 连接到Nebula Graph...")
    graph_store = NebulaGraphStore(
        addresses="127.0.0.1:9669",
        user="root",
        password="nebula",
        space="test"
    )

    try:
        # 2. 查看schema
        print("\n2. 查看图数据库schema...")
        schema = graph_store.get_schema()
        print(f"   点类型: {schema['vertex_types']}")
        print(f"   边类型: {schema['edge_types']}")

        # 3. 执行简单查询
        print("\n3. 执行简单查询...")
        query = "MATCH (v:person) RETURN v.name AS name, v.age AS age LIMIT 5"
        results = graph_store.execute_query(query)
        print(f"   查询: {query}")
        print(f"   结果: {results}")

        # 4. 获取图数据
        print("\n4. 获取图数据...")
        graph_data = graph_store.get_directed_graph()
        print(f"   节点数量: {len(graph_data['nodes'])}")
        print(f"   边数量: {len(graph_data['edges'])}")

        print("\n✅ 快速开始完成！")

    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        print("\n请确保:")
        print("   1. Nebula Graph数据库正在运行")
        print("   2. 已运行设置脚本创建测试环境:")
        print("      python examples/setup_nebula.py")

    finally:
        # 5. 关闭连接
        print("\n5. 关闭连接...")
        graph_store.close()


if __name__ == "__main__":
    quick_start()

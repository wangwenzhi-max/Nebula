"""
Nebula Graph环境设置脚本
用于创建测试图空间和示例数据
"""

import os
import sys

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from nebula_graph_store import NebulaGraphStore


def setup_test_space():
    """设置测试图空间和schema"""
    print("=" * 60)
    print("设置Nebula Graph测试环境")
    print("=" * 60)

    # 连接到Nebula Graph
    print("\n连接到Nebula Graph...")
    graph_store = NebulaGraphStore(
        addresses="127.0.0.1:9669",
        user="root",
        password="nebula",
        space=""  # 连接时不指定空间
    )

    try:
        # 创建图空间
        print("\n创建图空间 'test'...")
        create_space_query = """
        CREATE SPACE IF NOT EXISTS test (
            partition_num = 10,
            replica_factor = 1,
            vid_type = FIXED_STRING(32)
        )
        """
        result = graph_store.session.execute(create_space_query)
        if result.is_succeeded():
            print("图空间创建成功")
        else:
            print(f"图空间创建失败: {result.error_msg()}")

        # 使用图空间
        print("\n使用图空间 'test'...")
        result = graph_store.session.execute("USE test")
        if result.is_succeeded():
            print("图空间切换成功")
        else:
            print(f"图空间切换失败: {result.error_msg()}")
            return

        # 等待图空间就绪
        print("\n等待图空间就绪...")
        import time
        time.sleep(2)

        # 创建Tag (点类型)
        print("\n创建Tag 'person'...")
        create_tag_query = """
        CREATE TAG IF NOT EXISTS person (
            name string,
            age int,
            gender string,
            occupation string
        )
        """
        result = graph_store.session.execute(create_tag_query)
        if result.is_succeeded():
            print("Tag 'person' 创建成功")
        else:
            print(f"Tag创建失败: {result.error_msg()}")

        # 创建Edge Type (边类型)
        print("\n创建Edge Type 'friend'...")
        create_edge_query = """
        CREATE EDGE IF NOT EXISTS friend (
            since int,
            relationship string
        )
        """
        result = graph_store.session.execute(create_edge_query)
        if result.is_succeeded():
            print("Edge Type 'friend' 创建成功")
        else:
            print(f"Edge Type创建失败: {result.error_msg()}")

        # 等待schema生效
        print("\n等待schema生效...")
        time.sleep(2)

        # 插入示例数据 - 节点
        print("\n插入示例节点数据...")
        nodes_data = [
            ("p1", {"name": "张三", "age": 30, "gender": "男", "occupation": "工程师"}),
            ("p2", {"name": "李四", "age": 25, "gender": "女", "occupation": "设计师"}),
            ("p3", {"name": "王五", "age": 28, "gender": "男", "occupation": "医生"}),
            ("p4", {"name": "赵六", "age": 32, "gender": "女", "occupation": "教师"}),
            ("p5", {"name": "钱七", "age": 27, "gender": "男", "occupation": "律师"}),
        ]

        for vid, props in nodes_data:
            props_str = ", ".join([f'{k}: "{v}"' if isinstance(v, str) else f'{k}: {v}'
                                   for k, v in props.items()])
            query = f'INSERT VERTEX person(name, age, gender, occupation) VALUES "{vid}": ({props_str})'
            result = graph_store.session.execute(query)
            if result.is_succeeded():
                print(f"  节点 {vid} 插入成功")
            else:
                print(f"  节点 {vid} 插入失败: {result.error_msg()}")

        # 插入示例数据 - 边
        print("\n插入示例边数据...")
        edges_data = [
            ("p1", "p2", {"since": 2020, "relationship": "大学同学"}),
            ("p2", "p3", {"since": 2021, "relationship": "工作伙伴"}),
            ("p1", "p3", {"since": 2019, "relationship": "高中同学"}),
            ("p3", "p4", {"since": 2022, "relationship": "朋友"}),
            ("p4", "p5", {"since": 2020, "relationship": "邻居"}),
            ("p1", "p4", {"since": 2018, "relationship": "同事"}),
            ("p2", "p5", {"since": 2023, "relationship": "健身伙伴"}),
        ]

        for src, dst, props in edges_data:
            props_str = ", ".join([f'{k}: {v}' if k == "since" else f'{k}: "{v}"'
                                   for k, v in props.items()])
            query = f'INSERT EDGE friend(since, relationship) VALUES "{src}"->"{dst}": ({props_str})'
            result = graph_store.session.execute(query)
            if result.is_succeeded():
                print(f"  边 {src}->{dst} 插入成功")
            else:
                print(f"  边 {src}->{dst} 插入失败: {result.error_msg()}")

        # 验证数据
        print("\n验证插入的数据...")

        # 查询节点数量
        query = "MATCH (v) RETURN count(v) AS count"
        result = graph_store.execute_query(query)
        print(f"节点数量: {result}")

        # 查询边数量
        query = "MATCH ()-[e]->() RETURN count(e) AS count"
        result = graph_store.execute_query(query)
        print(f"边数量: {result}")

        # 查询所有人员
        query = "MATCH (v:person) RETURN v.name, v.age, v.occupation"
        result = graph_store.execute_query(query)
        print(f"\n人员列表:")
        for person in result:
            print(f"  {person}")

        # 查询所有朋友关系
        query = """
        MATCH (v1:person)-[e:friend]->(v2:person)
        RETURN v1.name AS person1, v2.name AS person2, e.since AS since, e.relationship AS relationship
        """
        result = graph_store.execute_query(query)
        print(f"\n朋友关系:")
        for relation in result:
            print(f"  {relation}")

        print("\n" + "=" * 60)
        print("测试环境设置完成!")
        print("=" * 60)

    except Exception as e:
        print(f"\n设置过程中出错: {e}")
        import traceback
        traceback.print_exc()

    finally:
        graph_store.close()


def cleanup_test_space():
    """清理测试图空间"""
    print("\n" + "=" * 60)
    print("清理Nebula Graph测试环境")
    print("=" * 60)

    graph_store = NebulaGraphStore(
        addresses="127.0.0.1:9669",
        user="root",
        password="nebula",
        space=""
    )

    try:
        # 删除图空间
        print("\n删除图空间 'test'...")
        result = graph_store.session.execute("DROP SPACE IF EXISTS test")
        if result.is_succeeded():
            print("图空间删除成功")
        else:
            print(f"图空间删除失败: {result.error_msg()}")

        print("\n测试环境清理完成!")

    except Exception as e:
        print(f"\n清理过程中出错: {e}")

    finally:
        graph_store.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Nebula Graph测试环境管理")
    parser.add_argument(
        "--action",
        choices=["setup", "cleanup", "reset"],
        default="setup",
        help="操作类型: setup(设置), cleanup(清理), reset(重置)"
    )

    args = parser.parse_args()

    if args.action == "setup":
        setup_test_space()
    elif args.action == "cleanup":
        cleanup_test_space()
    elif args.action == "reset":
        cleanup_test_space()
        import time
        time.sleep(2)
        setup_test_space()

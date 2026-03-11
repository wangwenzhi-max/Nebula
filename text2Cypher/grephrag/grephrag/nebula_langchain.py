"""
优化的GraphRAG Demo using LangChain
"""

import os
import re
import time
import json
import logging
from datetime import datetime
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Optional, Dict, Any, List

from langchain_openai import ChatOpenAI
from langchain.chains import NebulaGraphQAChain
from langchain_community.graphs import NebulaGraph
from tenacity import retry, stop_after_attempt, wait_exponential

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== 配置类 ====================

@dataclass
class Config:
    """应用配置"""
    # LLM配置
    llm_base_url: str = "http://7.242.97.82:8080/v1"
    llm_model: str = "Qwen3-Coder-30B-A3B-Instruct"
    llm_temperature: float = 0.3
    llm_api_key: str = "EMPTY"
    
    # Nebula配置
    nebula_space: str = "rag_workshop"
    nebula_user: str = "root"
    nebula_password: str = "nebula"
    nebula_address: str = "10.78.212.222"
    nebula_port: int = 9669
    session_pool_size: int = 30
    
    # 代理配置
    proxy_enabled: bool = False
    
    def setup_environment(self):
        """设置环境变量"""
        if not self.proxy_enabled:
            proxy_vars = ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']
            for var in proxy_vars:
                os.environ[var] = ''
        
        # 设置其他环境变量
        os.environ['no_proxy'] = 'localhost,127.0.0.1'
        logger.info("环境变量设置完成")

# ==================== 连接管理器 ====================

class NebulaConnectionManager:
    """Nebula连接管理器"""
    
    def __init__(self, config: Config):
        self.config = config
        self._graph: Optional[NebulaGraph] = None
        self._connected = False
    
    def connect(self) -> NebulaGraph:
        """建立连接"""
        if self._connected and self._graph:
            return self._graph
        
        try:
            logger.info(f"连接到Nebula: {self.config.nebula_address}:{self.config.nebula_port}")
            self._graph = NebulaGraph(
                space=self.config.nebula_space,
                username=self.config.nebula_user,
                password=self.config.nebula_password,
                address=self.config.nebula_address,
                port=self.config.nebula_port,
                session_pool_size=self.config.session_pool_size,
            )
            
            # 测试连接
            self._graph.query("SHOW SPACES;")
            self._connected = True
            logger.info("Nebula连接成功")
            return self._graph
            
        except Exception as e:
            logger.error(f"连接失败: {e}")
            raise ConnectionError(f"Nebula连接失败: {e}")
    
    def disconnect(self):
        """断开连接"""
        if self._graph:
            self._graph = None
            self._connected = False
            logger.info("Nebula连接已断开")
    
    @contextmanager
    def get_connection(self):
        """获取连接的上下文管理器"""
        try:
            yield self.connect()
        finally:
            self.disconnect()
    
    @property
    def is_connected(self):
        return self._connected

# ==================== 查询优化器 ====================

class QueryOptimizer:
    """查询优化器"""
    
    def __init__(self):
        self.entity_patterns = [
            (r'(?:who|what|where|when|why|how)\s+(?:is|are)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', 1),
            (r'(?:tell|ask)\s+(?:me\s+)?about\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', 1),
        ]
    
    def preprocess(self, query: str) -> str:
        """预处理查询"""
        if not query:
            return query
        
        # 清理
        query = ' '.join(query.split())
        query = query.strip('?').strip()
        
        return query
    
    def extract_entity(self, query: str) -> Optional[str]:
        """提取主要实体"""
        for pattern, group_idx in self.entity_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(group_idx)
        return None
    
    def generate_variations(self, query: str) -> List[str]:
        """生成查询变体"""
        entity = self.extract_entity(query)
        if not entity:
            return [query]
        
        variations = [
            query,
            f"Tell me about {entity}",
            f"What do you know about {entity}",
            f"Describe {entity}",
            f"Information about {entity}"
        ]
        
        return list(set(variations))

# ==================== 查询引擎 ====================

class GraphRAGQueryEngine:
    """GraphRAG查询引擎"""
    
    def __init__(self, config: Config):
        self.config = config
        self.connection_manager = NebulaConnectionManager(config)
        self.optimizer = QueryOptimizer()
        self.llm = None
        self.chain = None
        self.stats = {
            'success': 0,
            'failure': 0,
            'total_time': 0,
            'queries': []
        }
        
        self._initialize()
    
    def _initialize(self):
        """初始化LLM"""
        logger.info(f"初始化LLM: {self.config.llm_model}")
        self.llm = ChatOpenAI(
            openai_api_key=self.config.llm_api_key,
            base_url=self.config.llm_base_url,
            model=self.config.llm_model,
            temperature=self.config.llm_temperature
        )
    
    def _create_chain(self, graph: NebulaGraph) -> NebulaGraphQAChain:
        """创建查询链"""
        return NebulaGraphQAChain.from_llm(
            llm=self.llm,
            graph=graph,
            verbose=True,
            allow_dangerous_requests=True,  # 注意安全风险
        )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def query(self, question: str) -> Dict[str, Any]:
        """执行查询"""
        start_time = time.time()
        result = {
            'question': question,
            'processed_question': None,
            'answer': None,
            'success': False,
            'error': None,
            'duration': 0,
            'entity': None
        }
        
        try:
            # 1. 预处理查询
            processed = self.optimizer.preprocess(question)
            result['processed_question'] = processed
            
            # 2. 提取实体
            entity = self.optimizer.extract_entity(processed)
            result['entity'] = entity
            logger.info(f"提取到实体: {entity}")
            
            # 3. 获取连接并执行查询
            with self.connection_manager.get_connection() as graph:
                if not self.chain:
                    self.chain = self._create_chain(graph)
                
                # 执行查询
                response = self.chain.invoke(processed)
                result['answer'] = response
                result['success'] = True
                
                # 如果结果为空，尝试查询变体
                if not response or len(str(response)) < 20:
                    variations = self.optimizer.generate_variations(processed)
                    for var in variations[1:3]:  # 尝试前2个变体
                        alt_response = self.chain.invoke(var)
                        if alt_response and len(str(alt_response)) > len(str(response)):
                            result['answer'] = alt_response
                            logger.info(f"使用变体查询成功: {var}")
                            break
                
                self.stats['success'] += 1
                
        except Exception as e:
            result['error'] = str(e)
            result['success'] = False
            self.stats['failure'] += 1
            logger.error(f"查询失败: {e}")
            
        finally:
            result['duration'] = time.time() - start_time
            self.stats['total_time'] += result['duration']
            self.stats['queries'].append(result)
        
        return result
    
    def batch_query(self, questions: List[str]) -> List[Dict[str, Any]]:
        """批量查询"""
        results = []
        for q in questions:
            logger.info(f"执行查询 [{len(results)+1}/{len(questions)}]: {q}")
            result = self.query(q)
            results.append(result)
            time.sleep(0.5)  # 避免请求过快
        return results
    
    def print_stats(self):
        """打印统计信息"""
        total = self.stats['success'] + self.stats['failure']
        if total == 0:
            print("暂无统计数据")
            return
        
        success_rate = (self.stats['success'] / total) * 100
        avg_time = self.stats['total_time'] / max(self.stats['success'], 1)
        
        print("\n" + "-"*30)
        print("性能统计")
        print("-"*30)
        print(f"✅ 成功查询: {self.stats['success']}")
        print(f"❌ 失败查询: {self.stats['failure']}")
        print(f"📈 成功率: {success_rate:.1f}%")
        print(f"⏱️  平均耗时: {avg_time:.2f}秒")
        print(f"📊 总查询数: {total}")
        print("-"*30)

# ==================== 主程序 ====================

def format_result(result: Dict[str, Any]) -> str:
    """格式化结果输出"""
    output = []
    output.append("\n" + "="*60)
    output.append(f"📊 查询结果 [{datetime.now().strftime('%H:%M:%S')}]")
    output.append("="*60)
    
    if result['success']:
        output.append(f"✅ 状态: 成功")
        output.append(f"❓ 原始问题: {result['question']}")
        
        if result['processed_question'] != result['question']:
            output.append(f"⚙️  优化问题: {result['processed_question']}")
        
        if result['entity']:
            output.append(f"🔍 识别实体: {result['entity']}")
        
        # 格式化答案
        answer = result['answer']
        if isinstance(answer, dict):
            if 'result' in answer:
                answer = answer['result']
            else:
                answer = json.dumps(answer, indent=2, ensure_ascii=False)
        
        output.append(f"📝 答案:")
        output.append(str(answer))
        output.append(f"⏱️  耗时: {result['duration']:.2f}秒")
    else:
        output.append(f"❌ 状态: 失败")
        output.append(f"❓ 问题: {result['question']}")
        output.append(f"⚠️  错误: {result['error']}")
    
    output.append("="*60)
    return '\n'.join(output)

def main():
    """主函数"""
    print("\n" + "*"*30)
    print("GraphRAG Demo 启动")
    print("*"*30)
    
    # 1. 初始化配置
    config = Config()
    config.setup_environment()
    
    # 2. 创建查询引擎
    engine = GraphRAGQueryEngine(config)
    
    # 3. 测试查询
    test_queries = [
        '查找年龄最大和年龄最小的球员所效力的球队，并列出球队名',
        "Who is Rocket?",
        "What are the relationships of Rocket?",
        "Tell me about Groot",
        "Who is the leader of Guardians?"
    ]
    
    # 4. 执行查询
    results = engine.batch_query(test_queries)
    
    # 5. 打印结果
    for i, result in enumerate(results, 1):
        print(format_result(result))
    
    # 6. 打印统计信息
    engine.print_stats()


if __name__ == "__main__":
    main()
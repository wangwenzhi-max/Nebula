"""
优化的RAG Demo
"""
import os
import asyncio
import logging
import time
from dataclasses import dataclass
from typing import List, Optional
from contextlib import asynccontextmanager
from functools import wraps

from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.milvus import MilvusVectorStore
from utils import CustomOpenAILLM, CustomOpenAIEmbedding, Config

import nest_asyncio
nest_asyncio.apply()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class RAGConfig:
    """RAG应用配置"""
    milvus_uri: str = "http://10.78.212.222:19530"
    milvus_dim: int = 4096
    milvus_overwrite: bool = True
    input_file: str = "/home/cheleiping/workspace/GraphRAG/demos/data/paul_graham_essay.txt"
    llm_temperature: float = 0.1
    llm_timeout: int = 60
    chunk_size: int = 512
    chunk_overlap: int = 50
    similarity_top_k: int = 3
    
    @classmethod
    def from_env(cls):
        """从环境变量创建配置"""
        return cls(
            milvus_uri=os.getenv("MILVUS_URI", cls.milvus_uri),
            input_file=os.getenv("INPUT_FILE", cls.input_file)
        )

def retry(max_attempts=3, delay=1):
    """重试装饰器"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        logger.error(f"最终失败: {func.__name__}")
                        raise
                    logger.warning(f"第{attempt+1}次失败，{delay}秒后重试: {e}")
                    await asyncio.sleep(delay)
        return wrapper
    return decorator

class MilvusConnection:
    """Milvus连接管理器"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.vector_store = None
        self.storage_context = None
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        logger.info(f"连接到Milvus: {self.config.milvus_uri}")
        self.vector_store = MilvusVectorStore(
            uri=self.config.milvus_uri,
            dim=self.config.milvus_dim,
            overwrite=self.config.milvus_overwrite
        )
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        logger.info("关闭Milvus连接")
        if self.vector_store:
            # 如果有清理方法，在这里调用
            pass

class RAGApplication:
    """RAG应用主类"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self._setup_llm()
        self.index = None
        self.query_engine = None
    
    def _setup_llm(self):
        """设置LLM"""
        app_config = Config()
        Settings.llm = CustomOpenAILLM(
            api_url=app_config.llm_api_url,
            model=app_config.llm_model_name,
            api_key="EMPTY",
            temperature=self.config.llm_temperature,
            timeout=self.config.llm_timeout
        )
        Settings.embed_model = CustomOpenAIEmbedding(app_config)
        Settings.chunk_size = self.config.chunk_size
        Settings.chunk_overlap = self.config.chunk_overlap
        logger.info(f"LLM配置完成: {app_config.llm_model_name}")
    
    @retry(max_attempts=2)
    async def load_documents(self) -> List:
        """加载文档"""
        if not os.path.exists(self.config.input_file):
            raise FileNotFoundError(f"文件不存在: {self.config.input_file}")
        
        logger.info(f"加载文档: {self.config.input_file}")
        documents = SimpleDirectoryReader(
            input_files=[self.config.input_file]
        ).load_data()
        
        logger.info(f"成功加载 {len(documents)} 个文档")
        if documents:
            logger.info(f"文档ID: {documents[0].doc_id}")
            logger.info(f"文档长度: {len(documents[0].text)} 字符")
        
        return documents
    
    async def create_index(self, documents: List, milvus: MilvusConnection):
        """创建向量索引"""
        logger.info("开始创建向量索引...")
        start = time.time()
        
        self.index = VectorStoreIndex.from_documents(
            documents,
            storage_context=milvus.storage_context,
            embed_model=Settings.embed_model,
            show_progress=True
        )
        
        elapsed = time.time() - start
        logger.info(f"索引创建完成，耗时: {elapsed:.2f}秒")
        return self.index
    
    async def initialize_query_engine(self):
        """初始化查询引擎"""
        if not self.index:
            raise RuntimeError("请先创建索引")
        
        self.query_engine = self.index.as_query_engine(
            llm=Settings.llm,
            similarity_top_k=self.config.similarity_top_k
        )
    
    async def query(self, question: str) -> str:
        """执行查询"""
        if not self.query_engine:
            await self.initialize_query_engine()
        
        logger.info(f"执行查询: {question}")
        start = time.time()
        
        response = self.query_engine.query(question)
        
        elapsed = time.time() - start
        logger.info(f"查询完成，耗时: {elapsed:.2f}秒")
        
        return str(response)
    
    async def batch_query(self, questions: List[str]) -> List[str]:
        """批量查询"""
        results = []
        for q in questions:
            results.append(await self.query(q))
        return results

async def main():
    """主函数"""
    # 清除代理
    os.environ.update({
        "http_proxy": "",
        "https_proxy": "",
        "NO_PROXY": "localhost,127.0.0.1"
    })
    
    # 创建配置
    config = RAGConfig.from_env()
    
    # 创建应用
    app = RAGApplication(config)
    
    # 使用Milvus连接
    async with MilvusConnection(config) as milvus:
        try:
            # 1. 加载文档
            documents = await app.load_documents()
            
            # 2. 创建索引
            await app.create_index(documents, milvus)
            
            # 3. 执行查询
            questions = [
                "What did the author learn?",
                "What challenges did the disease pose for the author?"
            ]
            
            results = await app.batch_query(questions)
            
            # 4. 打印结果
            for i, (q, r) in enumerate(zip(questions, results)):
                print(f"\n{'='*50}")
                print(f"问题 {i+1}: {q}")
                print(f"{'='*50}")
                print(f"回答: {r}")
                print(f"{'='*50}")
                
        except Exception as e:
            logger.error(f"应用执行失败: {e}")
            raise

if __name__ == "__main__":
    asyncio.run(main())
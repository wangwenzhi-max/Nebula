"""
优化的RAG Demo using LangChain + Milvus - 修复连接问题
"""

import asyncio
import os
import time
import socket
import logging
from typing import List, Optional
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModel
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_milvus import Milvus
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_environment():
    """设置环境变量，禁用代理"""
    proxy_vars = [
        'http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY',
        'ftp_proxy', 'FTP_PROXY', 'all_proxy', 'ALL_PROXY'
    ]
    
    for var in proxy_vars:
        os.environ[var] = ''
    
    # 设置不经过代理的地址
    os.environ['NO_PROXY'] = 'localhost,127.0.0.1,10.78.212.222'
    
    logger.info("代理设置已清除")
    logger.info(f"NO_PROXY: {os.environ.get('NO_PROXY')}")


@dataclass
class RAGConfig:
    """RAG配置类"""
    # 文件配置
    file_path: str = "/home/cheleiping/workspace/GraphRAG/demos/data/paul_graham_essay.txt"
    chunk_size: int = 2000
    chunk_overlap: int = 200
    
    # LLM配置
    llm_base_url: str = "http://7.242.97.82:8081/v1"
    llm_model: str = "Qwen3-VL-30B-A3B-Instruct"
    llm_temperature: float = 0.3
    llm_api_key: str = "EMPTY"
    llm_timeout: int = 60
    
    # Embedding配置
    embedding_model_path: str = "/mnt/esfs/llm/Qwen3-VL-Embedding-2B"
    
    # Milvus配置
    milvus_uri: str = "http://10.78.212.222:19530"
    milvus_timeout: int = 60
    milvus_secure: bool = False
    milvus_retry_count: int = 3
    milvus_retry_delay: int = 5
    milvus_collection_name: str = "rag_demo"
    
    # 检索配置
    top_k: int = 4
    
    # 提示词模板
    prompt_template: str = """
    Human: You are an AI assistant, and provides answers to questions by using fact based and statistical information when possible.
    Use the following pieces of information to provide a concise answer to the question enclosed in <question> tags.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    <context>
    {context}
    </context>

    <question>
    {question}
    </question>

    The response should be specific and use statistics or numbers when possible.

    Assistant:"""
    
    def validate(self):
        """验证配置"""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"文件不存在: {self.file_path}")
        if self.chunk_size < self.chunk_overlap:
            raise ValueError("chunk_size必须大于chunk_overlap")
        return True


class QwenEmbedding(Embeddings):
    """Qwen Embedding类"""
    
    def __init__(self, config: RAGConfig, device: Optional[str] = None):
        self.config = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self._initialize_model()
    
    def _initialize_model(self):
        """初始化模型"""
        try:
            logger.info(f"加载Embedding模型: {self.config.embedding_model_path}")
            logger.info(f"使用设备: {self.device}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.embedding_model_path, 
                trust_remote_code=True
            )
            
            self.model = AutoModel.from_pretrained(
                self.config.embedding_model_path, 
                trust_remote_code=True, 
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
            ).to(self.device)
            
            self.model.eval()
            logger.info("Embedding模型加载成功")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise RuntimeError(f"无法加载Embedding模型: {e}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入文档"""
        if not texts:
            return []
        
        try:
            batch_size = 32
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                
                inputs = self.tokenizer(
                    batch, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                    all_embeddings.extend(embeddings.tolist())
                
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"文档嵌入失败: {e}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """嵌入查询"""
        if not text:
            return []
        
        return self.embed_documents([text])[0]
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class DocumentProcessor:
    """文档处理器"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
    
    def load_documents(self, file_path: Optional[str] = None) -> List:
        """加载文档"""
        target_file = file_path or self.config.file_path
        
        if not os.path.exists(target_file):
            raise FileNotFoundError(f"文件不存在: {target_file}")
        
        try:
            logger.info(f"加载文档: {target_file}")
            
            loader = TextLoader(target_file, encoding='utf-8')
            documents = loader.load()
            logger.info(f"成功加载 {len(documents)} 个文档")
            
            return documents
            
        except Exception as e:
            logger.error(f"文档加载失败: {e}")
            raise
    
    def split_documents(self, documents: List) -> List:
        """分块处理"""
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
            )
            
            docs = text_splitter.split_documents(documents)
            logger.info(f"文档分块完成: {len(documents)} -> {len(docs)} 块")
            
            return docs
            
        except Exception as e:
            logger.error(f"文档分块失败: {e}")
            raise


class MilvusConnectionManager:
    """Milvus连接管理器"""
    
    def __init__(self, config: RAGConfig, embedding_model: Embeddings):
        self.config = config
        self.embedding_model = embedding_model
        self.vectorstore = None
        self.retriever = None
    
    def test_connection(self) -> bool:
        """测试Milvus连接"""
        try:
            # 解析URI
            uri = self.config.milvus_uri
            if uri.startswith('http://'):
                host_port = uri[7:]
            else:
                host_port = uri
            
            if ':' in host_port:
                host, port = host_port.split(':')
                port = int(port)
            else:
                host = host_port
                port = 19530
            
            # 测试TCP连接
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((host, port))
            sock.close()
            
            if result == 0:
                logger.info(f"Milvus服务器 {host}:{port} 可达")
                return True
            else:
                logger.error(f"Milvus服务器 {host}:{port} 不可达，错误码: {result}")
                return False
                
        except Exception as e:
            logger.error(f"连接测试失败: {e}")
            return False
    
    def create_from_documents(self, docs: List) -> Milvus:
        """从文档创建向量存储"""
        
        # 测试连接
        self.test_connection()
        
        for attempt in range(self.config.milvus_retry_count):
            try:
                logger.info(f"尝试连接Milvus (尝试 {attempt + 1}/{self.config.milvus_retry_count})...")
                
                # 生成唯一的集合名称
                collection_name = f"{self.config.milvus_collection_name}_{int(time.time())}"
                
                self.vectorstore = Milvus.from_documents(
                    documents=docs,
                    embedding=self.embedding_model,
                    connection_args={
                        "uri": self.config.milvus_uri,
                        "timeout": self.config.milvus_timeout,
                        "secure": self.config.milvus_secure,
                    },
                    collection_name=collection_name,
                    drop_old=True,
                )
                
                logger.info(f"向量存储创建成功，集合: {collection_name}")
                
                # 创建检索器
                self.retriever = self.vectorstore.as_retriever(
                    search_kwargs={"k": self.config.top_k}
                )
                
                return self.vectorstore
                
            except Exception as e:
                logger.error(f"尝试 {attempt + 1} 失败: {e}")
                
                if attempt < self.config.milvus_retry_count - 1:
                    logger.info(f"{self.config.milvus_retry_delay}秒后重试...")
                    time.sleep(self.config.milvus_retry_delay)
                else:
                    logger.error("所有重试都失败了")
                    raise
        
        return None
    
    def get_retriever(self):
        """获取检索器"""
        if not self.retriever:
            raise RuntimeError("请先创建向量存储")
        return self.retriever


class RAGApplication:
    """RAG应用主类"""
    
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self.config.validate()
        
        # 设置环境
        setup_environment()
        
        self.doc_processor = DocumentProcessor(self.config)
        self.embedding_model = None
        self.milvus_manager = None
        self.llm = None
        self.chain = None
        
        logger.info("RAG应用初始化完成")
    
    def initialize(self):
        """初始化所有组件"""
        # 1. 初始化Embedding模型
        self.embedding_model = QwenEmbedding(self.config)
        
        # 2. 初始化LLM
        self.llm = ChatOpenAI(
            openai_api_key=self.config.llm_api_key,
            base_url=self.config.llm_base_url,
            model=self.config.llm_model,
            temperature=self.config.llm_temperature,
            timeout=self.config.llm_timeout
        )
        logger.info(f"LLM初始化完成: {self.config.llm_model}")
        
        # 3. 初始化Milvus管理器
        self.milvus_manager = MilvusConnectionManager(self.config, self.embedding_model)
        
        return self
    
    async def setup_vectorstore(self):
        """设置向量存储"""
        # 1. 加载文档
        documents = self.doc_processor.load_documents()
        
        # 2. 分块处理
        docs = self.doc_processor.split_documents(documents)
        
        # 3. 创建向量存储
        self.milvus_manager.create_from_documents(docs)
        
        # 4. 构建RAG链
        self._build_chain()
        
        return self
    
    def _build_chain(self):
        """构建RAG链"""
        prompt = PromptTemplate(
            template=self.config.prompt_template,
            input_variables=["context", "question"]
        )
        
        retriever = self.milvus_manager.get_retriever()
        
        def format_docs(docs):
            return "\n\n---\n\n".join(doc.page_content for doc in docs)
        
        self.chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        logger.info("RAG链构建完成")
    
    async def query(self, question: str) -> dict:
        """执行查询"""
        result = {
            'query': question,
            'answer': None,
            'success': False,
            'error': None,
            'duration': 0
        }
        
        start_time = time.time()
        
        try:
            if not self.chain:
                raise RuntimeError("请先调用setup_vectorstore()")
            
            answer = await self.chain.ainvoke(question)
            
            result['answer'] = answer
            result['success'] = True
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"查询失败: {e}")
            
        finally:
            result['duration'] = time.time() - start_time
        
        return result
    
    def print_result(self, result: dict):
        """打印结果"""
        print("\n" + "="*60)
        print(f"📊 查询结果")
        print("="*60)
        
        print(f"❓ 问题: {result['query']}")
        
        if result['success']:
            print(f"✅ 状态: 成功")
            print(f"📝 答案: {result['answer']}")
            print(f"⏱️  耗时: {result['duration']:.2f}秒")
        else:
            print(f"❌ 状态: 失败")
            print(f"⚠️  错误: {result['error']}")
        
        print("="*60)


async def main():
    """主函数"""
    print("\n" + "-"*30)
    print("RAG Demo 启动")
    print("-"*30)
    
    try:
        # 1. 创建应用
        app = RAGApplication()
        
        # 2. 初始化
        app.initialize()
        
        # 3. 设置向量存储
        await app.setup_vectorstore()
        
        # 4. 执行查询
        queries = [
            "What did the author learn?",
            "What challenges did the disease pose for the author?"
        ]
        
        for i, query in enumerate(queries, 1):
            print(f"\n📌 查询 {i}/{len(queries)}")
            result = await app.query(query)
            app.print_result(result)
        
    except Exception as e:
        logger.error(f"应用执行失败: {e}")
        raise
    
    print("\n" + "-"*30)
    print("Demo 运行完成")
    print("-"*30)


if __name__ == "__main__":
    asyncio.run(main())
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from utils.build_graph import build_knowledge_graph
from rank_bm25 import BM25Okapi
import jieba
from pathlib import Path
from langchain_core.documents import Document
from langchain.text_splitter import SpacyTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import pdfplumber  # 更可靠的PDF解析库
from langchain_community.document_loaders import Docx2txtLoader, TextLoader
import tempfile
import spacy


# 配置常量
SUPPORTED_EXT = ['.pdf', '.docx', '.txt']
TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True, parents=True)  # 确保临时目录存在

def process_uploaded_files(uploaded_files):
    """统一处理上传文件的主函数"""
    documents = []
    
    for file in uploaded_files:
        file_name = file.name
        file_ext = Path(file_name).suffix.lower()
        
        # 创建临时文件安全上下文
        with tempfile.NamedTemporaryFile(
            dir=TEMP_DIR,
            suffix=file_ext,
            delete=False  # 手动控制删除
        ) as temp_file:
            try:
                # 写入临时文件
                temp_file.write(file.getbuffer())
                temp_path = Path(temp_file.name)
                
                # 根据文件类型选择处理器
                if file_ext == '.pdf':
                    # 使用pdfplumber提取文本（更好的中文支持）
                    with pdfplumber.open(temp_path) as pdf:
                        text = "\n\n".join(
                            f"Page {i+1}:\n{p.extract_text()}" 
                            for i, p in enumerate(pdf.pages)
                        )
                    documents.append(Document(page_content=text))
                    
                elif file_ext == '.docx':
                    loader = Docx2txtLoader(str(temp_path))
                    documents.extend(loader.load())
                    
                elif file_ext == '.txt':
                    # 使用自动检测编码的加载器
                    loader = TextLoader(
                        str(temp_path),
                        autodetect_encoding=True
                    )
                    documents.extend(loader.load())
                    
                else:
                    st.warning(f"跳过不支持的文件类型: {file_name}")
                    continue
                    
            except pdfplumber.PDFSyntaxError as e:
                st.error(f"PDF解析失败 [{file_name}]: 文件可能已损坏")
                continue
            except UnicodeDecodeError as e:
                st.error(f"编码解析失败 [{file_name}]: 请检查文件编码格式")
                continue
            except Exception as e:
                st.error(f"处理文件 [{file_name}] 失败: {str(e)}")
                continue
            finally:
                # 确保删除临时文件
                try:
                    temp_path.unlink()
                except Exception as e:
                    st.error(f"临时文件清理失败: {str(e)}")

    return documents

def chinese_text_split(documents):
    """增强型中文文本分割"""
    try:
        # 方案1：使用Spacy语义分割
        nlp = spacy.load("zh_core_web_sm")
        text_splitter = SpacyTextSplitter(
            pipeline="zh_core_web_sm",
            chunk_size=100,
            chunk_overlap=20,
            separator=["\n\n", "。", "！", "？"]  # 多级分隔符
        )
    except:
        # 方案2：回退到递归字符分割
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=20,
            separators=["\n\n", "。", "！", "？", "\n", "，", ""]
        )
    
    texts = text_splitter.split_documents(documents)
    
    # 后处理
    text_contents = [
        doc.page_content.strip() 
        for doc in texts
        if len(doc.page_content.strip()) > 20  # 过滤空内容
    ]
    
    return (texts, text_contents)

def process_documents(uploaded_files, reranker, embedding_model, device):
    if st.session_state.documents_loaded:
        return

    st.session_state.processing = True
    documents = process_uploaded_files(uploaded_files=uploaded_files)
    
    texts, text_contents = chinese_text_split(documents)

    # 🚀 中文嵌入模型
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # 向量存储
    vector_store = FAISS.from_documents(texts, embeddings)
    
    # BM25中文检索
    bm25_retriever = BM25Retriever.from_texts(
        text_contents, 
        bm25_impl=BM25Okapi,
        preprocess_func=lambda text: [word for word in jieba.lcut(text) if word.strip()]
    )

    # 混合检索（调整权重）
    ensemble_retriever = EnsembleRetriever(
        retrievers=[
            bm25_retriever,
            vector_store.as_retriever(search_kwargs={"k": 8})  # 增加召回数量
        ],
        weights=[0.3, 0.7]  # 提高向量检索权重
    )

    # 存储会话状态
    st.session_state.retrieval_pipeline = {
        "ensemble": ensemble_retriever,
        "reranker": reranker,
        "texts": text_contents,
        "knowledge_graph": build_knowledge_graph(texts)
    }

    st.session_state.documents_loaded = True
    st.session_state.processing = False

    # ✅ 调试信息中文化
    if "knowledge_graph" in st.session_state.retrieval_pipeline:
        G = st.session_state.retrieval_pipeline["knowledge_graph"]
        st.write(f"🔗 总节点数: {len(G.nodes)}")
        st.write(f"🔗 总边数: {len(G.edges)}")
        st.write(f"🔗 示例节点: {list(G.nodes)[:10]}")
        st.write(f"🔗 示例关系: {list(G.edges(data=True))[:5]}")

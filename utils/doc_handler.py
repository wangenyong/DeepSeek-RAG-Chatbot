import streamlit as st
import pandas as pd
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
import re
import logging


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
        logging.info(f"开始处理文件 | 文件名：{file_name} | 类型：{file_ext}")

        # 创建临时文件安全上下文
        with tempfile.NamedTemporaryFile(
            dir=TEMP_DIR,
            suffix=file_ext,
            delete=False
        ) as temp_file:
            try:
                # 写入临时文件
                temp_file.write(file.getbuffer())
                temp_path = Path(temp_file.name)
                logging.info(f"创建临时文件成功 | 路径：{temp_path}")

                # 根据文件类型选择处理器
                if file_ext == '.pdf':
                    logging.info(f"开始解析PDF文件 | 文件名：{file_name}")
                    with pdfplumber.open(temp_path) as pdf:
                        text = "\n\n".join(
                            f"Page {i+1}:\n{p.extract_text()}" 
                            for i, p in enumerate(pdf.pages)
                        )
                        logging.info(f"PDF解析完成 | 页数：{len(pdf.pages)} | 字符数：{len(text)}")
                    documents.append(Document(page_content=text))
                    
                elif file_ext == '.docx':
                    logging.info(f"开始解析DOCX文件 | 文件名：{file_name}")
                    loader = Docx2txtLoader(str(temp_path))
                    docs = loader.load()
                    logging.info(f"DOCX解析完成 | 段落数：{len(docs)}")
                    documents.extend(docs)
                    
                elif file_ext == '.txt':
                    logging.info(f"开始解析TXT文件 | 文件名：{file_name}")
                    loader = TextLoader(
                        str(temp_path),
                        autodetect_encoding=True
                    )
                    docs = loader.load()
                    logging.info(f"TXT解析完成 | 字符数：{len(docs[0].page_content) if docs else 0}")
                    documents.extend(docs)
                    
                # 新增Excel处理分支
                elif file_ext in ('.xls', '.xlsx'):
                    logging.info(f"开始解析Excel文件 | 文件名：{file_name}")
                    
                    # 读取Excel文件
                    df_dict = pd.read_excel(temp_path, sheet_name=None)
                    text_content = []
                    
                    # 遍历所有工作表
                    for sheet_name, df in df_dict.items():
                        sheet_text = [
                            f"工作表【{sheet_name}】",
                            "表头：" + ", ".join(df.columns.astype(str)),
                            "数据：\n" + df.to_string(index=False)
                        ]
                        text_content.append("\n".join(sheet_text))
                    
                    full_text = "\n\n".join(text_content)
                    documents.append(Document(page_content=full_text))
                    logging.info(f"Excel解析完成 | 工作表数：{len(df_dict)} | 总行数：{sum(len(df) for df in df_dict.values())}")
                    
                else:
                    logging.warning(f"跳过不支持的文件类型 | 文件名：{file_name}")
                    continue
                    
            except pdfplumber.PDFSyntaxError as e:
                logging.error(f"PDF解析失败 | 文件名：{file_name}", exc_info=True)
                continue
            except UnicodeDecodeError as e:
                logging.error(f"编码解析失败 | 文件名：{file_name} | 错误：{str(e)}")
                continue
            except Exception as e:
                logging.error(f"文件处理异常 | 文件名：{file_name}", exc_info=True)
                continue
            finally:
                try:
                    if temp_path.exists():
                        temp_path.unlink()
                        logging.info(f"临时文件已清理 | 路径：{temp_path}")
                except Exception as e:
                    logging.error(f"临时文件清理失败 | 路径：{temp_path}", exc_info=True)

    logging.info(f"文件处理完成 | 总处理文件数：{len(uploaded_files)} | 有效文档数：{len(documents)}")
    return documents

def chinese_text_split(documents):
    logging.info("开始文本分割 | 原始文档数：%d", len(documents))
    
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "。", "！", "？"],  # 按段落、句子分割
        chunk_size=800,
        chunk_overlap=160,
        is_separator_regex=False  # 明确关闭正则匹配，直接按字符匹配
    )
    
    texts = text_splitter.split_documents(documents)
    logging.info("初步分割完成 | 块数：%d", len(texts))
    
    # 后处理
    text_contents = [
        doc.page_content.strip() 
        for doc in texts
        if len(doc.page_content.strip()) > 20  # 过滤空内容
    ]
    
    return (texts, text_contents)

def process_documents(uploaded_files, reranker, embedding_model, device):
    if st.session_state.documents_loaded:
        logging.info("文档已加载，跳过处理流程")
        return

    try:
        st.session_state.processing = True
        logging.info("📂 开始文档处理流程")
        
        # 文件处理
        logging.info(f"收到上传文件 | 数量：{len(uploaded_files)}")
        documents = process_uploaded_files(uploaded_files=uploaded_files)
        logging.info(f"原始文档处理完成 | 有效文档数：{len(documents)}")
        
        # 文本分割
        logging.info("开始中文文本分割")
        texts, text_contents = chinese_text_split(documents)
        logging.info(f"文本分割完成 | 总段落数：{len(texts)} | 平均长度：{sum(len(t) for t in text_contents)//len(text_contents)}字符")

        # 嵌入模型初始化
        logging.info(f"初始化嵌入模型 | 模型：{embedding_model} | 设备：{device}")
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # 向量存储
        logging.info("创建FAISS向量存储")
        vector_store = FAISS.from_documents(texts, embeddings)
        logging.info(f"向量存储创建完成 | 文档数：{vector_store.index.ntotal} | 维度：{vector_store.index.d}")

        # BM25检索器
        logging.info("初始化BM25检索器")
        bm25_retriever = BM25Retriever.from_texts(
            text_contents, 
            bm25_impl=BM25Okapi,
            preprocess_func=lambda text: [word for word in jieba.lcut(text) if word.strip()]
        )
        logging.info(f"BM25检索器就绪 | 文档数：{len(text_contents)}")

        # 混合检索器
        logging.info("配置混合检索器")
        ensemble_retriever = EnsembleRetriever(
            retrievers=[
                bm25_retriever,
                vector_store.as_retriever(search_kwargs={"k": 8})
            ],
            weights=[0.3, 0.7]
        )
        logging.info(f"混合检索器配置完成 | 权重：BM25(30%) + 向量(70%) | 召回数量：8")

        # 知识图谱构建
        # logging.info("开始构建知识图谱")
        # knowledge_graph = build_knowledge_graph(texts)
        # logging.info(f"知识图谱构建完成 | 节点数：{len(knowledge_graph.nodes)} | 边数：{len(knowledge_graph.edges)}")

        # 存储会话状态
        st.session_state.retrieval_pipeline = {
            "ensemble": ensemble_retriever,
            "reranker": reranker,
            "texts": text_contents,
        }
        logging.info("检索管道配置完成")

        st.session_state.documents_loaded = True
        logging.info("✅ 文档处理流程完成")

    except Exception as e:
        logging.error("文档处理流程异常终止", exc_info=True)
        st.error(f"文档处理失败: {str(e)}")
    finally:
        st.session_state.processing = False
        logging.info("清理处理状态")

    # 知识图谱调试信息
    if "knowledge_graph" in st.session_state.retrieval_pipeline:
        try:
            G = st.session_state.retrieval_pipeline["knowledge_graph"]
            logging.info(f"知识图谱统计 | 节点示例：{list(G.nodes)[:5]}... | 边示例：{list(G.edges(data=True))[:3]}...")
            
            logging.info(f"🔗 总节点数: {len(G.nodes)}")
            logging.info(f"🔗 总边数: {len(G.edges)}")
            logging.info(f"🔗 示例节点: {list(G.nodes)[:10]}")
            logging.info(f"🔗 示例关系: {list(G.edges(data=True))[:5]}")
        except Exception as e:
            logging.warning("知识图谱调试信息显示失败", exc_info=True)

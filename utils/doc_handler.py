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
# 在顶部导入新增
from pandas.api.types import is_numeric_dtype
from tabulate import tabulate
import logging
import re

# 配置常量
SUPPORTED_EXT = ['.pdf', '.docx', '.txt', '.xls', '.xlsx']
TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True, parents=True)  # 确保临时目录存在

def process_excel_sheet(sheet_name: str, df: pd.DataFrame) -> Document:
    """优化单个工作表的处理逻辑"""
    # 元数据增强
    metadata = {
        "source_type": "excel",
        "sheet_name": sheet_name,
        "columns": df.columns.astype(str).tolist(),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "shape": f"{len(df)}x{len(df.columns)}"
    }
    
    # 表格内容优化处理
    table_str = tabulate(
        df.head(1000),  # 限制最大行数避免内存问题
        headers='keys',
        tablefmt='pipe',
        showindex=False,
        maxcolwidths=30,
        stralign="left",
        numalign="center"
    )
    
    # 结构化文档格式
    content = f"""
# EXCEL工作表 [{sheet_name}]

## 元数据
- 列数: {len(df.columns)}
- 行数: {len(df)}
- 列类型: {metadata['dtypes']}

## 数据预览
{table_str}
"""
    return Document(page_content=content.strip(), metadata=metadata)

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
                    try:
                        # 增强的Excel读取参数
                        df_dict = pd.read_excel(
                            temp_path,
                            sheet_name=None,
                            dtype=str,  # 统一转为字符串避免类型问题
                            na_filter=False,  # 禁用自动空值过滤
                            engine='openpyxl' if file_ext == '.xlsx' else None
                        )
                        
                        # 处理每个工作表
                        sheet_docs = []
                        for sheet_name, df in df_dict.items():
                            logging.debug(f"工作表【{sheet_name}】样例数据：\n{df.head(3).to_markdown()}")
                            try:
                                # 清理列名中的特殊字符
                                df.columns = [re.sub(r'[\\/:*?"<>|]', '_', str(col)) for col in df.columns]
                                
                                # 数值列特殊处理
                                num_cols = [col for col in df.columns if is_numeric_dtype(df[col])]
                                if num_cols:
                                    df[num_cols] = df[num_cols].apply(lambda x: x.round(6))
                                
                                # 生成结构化文档
                                doc = process_excel_sheet(sheet_name, df)
                                sheet_docs.append(doc)
                            except Exception as e:
                                logging.error(f"工作表处理失败 | 文件：{file_name} | 工作表：{sheet_name}", exc_info=True)
                                continue
                        
                        documents.extend(sheet_docs)
                        logging.info(f"Excel解析完成 | 工作表数：{len(sheet_docs)} | 总行数：{sum(len(df) for df in df_dict.values())}")
                        
                    except Exception as e:
                        logging.error(f"Excel文件解析失败 | 文件名：{file_name}", exc_info=True)
                        continue
                    
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


def table_aware_splitter(documents: list) -> tuple:
    """优化后的表格分块逻辑"""
    # 动态调整分块大小
    table_docs = [doc for doc in documents if doc.metadata.get('source_type') == 'excel']
    avg_cols = sum(len(doc.metadata['columns']) for doc in table_docs) / (len(table_docs) or 1)
    
    chunk_size = max(800, int(avg_cols * 150))  # 每列约150字符
    chunk_overlap = min(160, int(chunk_size * 0.3))  # 提高重叠比例

    # 表格专用分割器
    table_splitter = RecursiveCharacterTextSplitter(
        separators=[
            r'\n## 数据预览\n',  # 保留完整表格
            r'\n# EXCEL工作表 $$(.*?)$$'  # 按工作表分割
        ],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        is_separator_regex=True  # 启用正则表达式
    )
    
    # 分块后添加表格标识
    split_docs = []
    for doc in table_docs:
        for chunk in table_splitter.split_documents([doc]):
            chunk.metadata['is_table'] = True
            split_docs.append(chunk)
    
    return split_docs, [doc.page_content for doc in split_docs]


def chinese_text_split(documents):
    logging.info("开始文本分割 | 原始文档数：%d", len(documents))
    
    # 检测是否包含表格文档
    if any(doc.metadata.get('source_type') == 'excel' for doc in documents):
        return table_aware_splitter(documents)
    
    # 原始分割逻辑
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "。", "！", "？"],
        chunk_size=800,
        chunk_overlap=160,
        is_separator_regex=False
    )
    
    texts = text_splitter.split_documents(documents)
    text_contents = [
        doc.page_content.strip() 
        for doc in texts
        if len(doc.page_content.strip()) > 20
    ]
    return (texts, text_contents)


# 修改BM25初始化部分
def table_tokenizer(text: str) -> list:
    """表格敏感型分词器"""
    # 保留数字、货币符号、百分比等
    tokens = re.findall(r'\d+\.?\d*|[$¥€%]|\w+[\u4e00-\u9fff]', text)
    return [t.lower() for t in tokens if t.strip()]


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
            preprocess_func=table_tokenizer  # 使用定制分词器
        )
        logging.info(f"BM25检索器就绪 | 文档数：{len(text_contents)}")

        # 混合检索器
        logging.info("配置混合检索器")
        # 在process_documents中修改混合检索器配置
        ensemble_retriever = EnsembleRetriever(
            retrievers=[
                bm25_retriever,
                vector_store.as_retriever(search_kwargs={
                    "k": 12,  # 增加向量召回量
                    "score_threshold": 0.65  # 降低阈值
                })
            ],
            weights=[0.5, 0.5]  # 平衡权重
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

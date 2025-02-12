import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from utils.build_graph import build_knowledge_graph
from rank_bm25 import BM25Okapi
import os
import jieba
from langchain_core.documents import Document
from langchain.text_splitter import SpacyTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings



def process_documents(uploaded_files, reranker, embedding_model, device):
    if st.session_state.documents_loaded:
        return

    st.session_state.processing = True
    documents = []
    
    # 创建临时目录（增加中文路径支持）
    if not os.path.exists("temp"):
        os.makedirs("temp", exist_ok=True)
    
    # 文件处理（增加编码处理）
    for file in uploaded_files:
        try:
            file_path = os.path.join("temp", file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())

            # 增加中文PDF支持（解决PDF解析乱码）
            if file.name.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                loader._pdf_parser.text = lambda x: x.get_text().encode('utf-8', errors='replace').decode('utf-8')
            elif file.name.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
            elif file.name.endswith(".txt"):
                with open(file_path, "r", encoding='utf-8', errors='replace') as f:
                    text = f.read()
                documents.append(Document(page_content=text))
                continue
            else:
                continue
                
            documents.extend(loader.load())
            os.remove(file_path)
        except Exception as e:
            st.error(f"文件处理失败 {file.name}: {str(e)}")
            return

    # 中文文本分割
    text_splitter = SpacyTextSplitter(
        pipeline="zh_core_web_sm",
        chunk_size=400,
        chunk_overlap=80,
        separator="\n"
    )
    texts = text_splitter.split_documents(documents)
    text_contents = [doc.page_content for doc in texts]

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

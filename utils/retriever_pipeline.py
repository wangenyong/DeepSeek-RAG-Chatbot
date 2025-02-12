import streamlit as st
from utils.build_graph import retrieve_from_graph
from langchain_core.documents import Document
import requests
import jieba  # 🌟 新增中文分词库
import re

# 🌟 中文停用词列表
STOP_WORDS = set(["的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一个"])

def chinese_text_preprocess(text):
    """🌟 中文文本预处理"""
    # 去除特殊字符
    text = re.sub(r'[^\w\s\u4e00-\u9fa5]', '', text)
    # 分词处理
    words = jieba.cut(text)
    # 去除停用词
    words = [w for w in words if w not in STOP_WORDS]
    return ' '.join(words)

# 🚀 Query Expansion with HyDE (中文优化版)
def expand_query(query, uri, model):
    try:
        # 🌟 添加中文提示词模板
        prompt = f"""请根据以下问题生成一个假设性的中文回答，用于改进检索效果。保持回答简洁，用书面中文。
        
        问题：{query}
        假设回答："""
        
        response = requests.post(uri, json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "temperature": 0.7  # 🌟 调整生成多样性
        }, timeout=10).json()
        
        # 🌟 结果清洗
        generated = response.get('response', '').strip()
        generated = re.sub(r'\n+', ' ', generated)  # 去除多余换行
        return f"{query}\n{generated}"
    except Exception as e:
        st.error(f"查询扩展失败: {str(e)}")
        return query

# 🚀 Advanced Retrieval Pipeline (中文优化版)
def retrieve_documents(query, uri, model, chat_history=""):
    # 🌟 中文预处理
    processed_query = chinese_text_preprocess(query)
    
    if st.session_state.enable_hyde:
        expanded_query = expand_query(f"{chat_history}\n{processed_query}", uri, model)
        expanded_query = chinese_text_preprocess(expanded_query)  # 🌟 扩展查询也预处理
    else:
        expanded_query = processed_query

    # 🔍 使用中文优化的检索流程
    docs = st.session_state.retrieval_pipeline["ensemble"].invoke(
        expanded_query,
        search_kwargs={"k": st.session_state.max_contexts*2}  # 🌟 扩大初始检索范围
    )
    
    # 🚀 GraphRAG 中文优化
    if st.session_state.enable_graph_rag:
        # 🌟 中文图查询预处理
        graph_query = chinese_text_preprocess(query)
        graph_results = retrieve_from_graph(
            graph_query, 
            st.session_state.retrieval_pipeline["knowledge_graph"],
            lang="zh"  # 🌟 指定中文模式
        )
        
        # 🌟 处理图检索结果
        graph_docs = []
        for node in graph_results:
            if isinstance(node, dict):
                content = node.get("text_zh", "") or node.get("text", "")  # 🌟 优先取中文内容
            else:
                content = str(node)
            graph_docs.append(Document(
                page_content=content,
                metadata={"source": "knowledge_graph"}
            ))
        
        if graph_docs:
            docs = graph_docs + docs

    # 🚀 中文重排序优化
    if st.session_state.enable_reranking:
        # 🌟 使用中文优化的reranker
        reranker = st.session_state.retrieval_pipeline["reranker"]
        pairs = [[processed_query, chinese_text_preprocess(doc.page_content)] for doc in docs]
        
        # 🌟 批量处理提升性能
        batch_size = 32
        scores = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i+batch_size]
            scores.extend(reranker.predict(batch))
        
        # 按分数排序
        ranked_docs = [doc for _, doc in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)]
    else:
        ranked_docs = docs

    # 🌟 结果后处理
    final_docs = []
    for doc in ranked_docs[:st.session_state.max_contexts]:
        # 🌟 确保内容为中文
        content = doc.page_content
        if not any('\u4e00' <= c <= '\u9fff' in content for c in content):
            continue  # 过滤非中文内容
        final_docs.append(doc)
    
    return final_docs

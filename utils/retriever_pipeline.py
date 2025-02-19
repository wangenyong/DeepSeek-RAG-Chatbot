import streamlit as st
from utils.build_graph import retrieve_from_graph
from utils.chinese_tools import chinese_text_preprocess
from langchain_core.documents import Document
import requests
import re
import logging
from datetime import datetime

# 🚀 Query Expansion with HyDE (中文优化版)
def expand_query(query, uri, model):
    logging.info("🔍 开始查询扩展处理 | 原始查询：%s", query[:50]+"..." if len(query)>50 else query)
    start_time = datetime.now()
    
    try:
        # 🌟 添加中文提示词模板
        prompt = f"""请根据以下问题生成一个假设性的中文回答，用于改进检索效果。保持回答简洁，用书面中文。
        
        问题：{query}
        假设回答："""
        
        logging.info("生成提示模板 | 长度：%d 字符 | 示例：%s", 
                    len(prompt), prompt[:100].replace('\n', ' ')+"...")
        
        logging.info("调用语言模型 | 服务地址：%s | 模型：%s", uri, model)
        response = requests.post(uri, json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "temperature": 0.7  # 🌟 调整生成多样性
        }, timeout=10).json()
        logging.info("模型响应接收 | 状态码：%d | 响应长度：%d", 
                     response.status_code, len(response.text))
        
        # 🌟 结果清洗
        generated = response.get('response', '').strip()
        logging.info("原始生成内容 | 长度：%d 字符 | 示例：%s",
                    len(generated), generated[:50].replace('\n', ' ')+"...")
        
        # 结果处理
        cleaned = re.sub(r'\n+', ' ', generated)
        logging.info("查询扩展完成 | 耗时：%.2fs | 扩展后长度：%d→%d 字符",
                    (datetime.now()-start_time).total_seconds(),
                    len(query), len(cleaned))
        return f"{query}\n{cleaned}"
    except Exception as e:
        logging.error("查询扩展失败 | 错误类型：%s | 详情：%s", 
                     type(e).__name__, str(e), exc_info=True)
        st.error(f"查询扩展失败: {str(e)}")
        return query

# 🚀 Advanced Retrieval Pipeline (中文优化版)
def retrieve_documents(query, uri, model, chat_history=""):
    logging.info("🚀 开始文档检索流程 | 原始查询：%s", query[:100]+"..." if len(query)>100 else query)
    start_time = datetime.now()
    
    try:
        # 🌟 中文预处理
        processed_query = chinese_text_preprocess(query)
        logging.info("查询预处理完成 | 原始长度：%d → 处理后：%d 字符",
                     len(query), len(processed_query))
        # HyDE扩展
        if st.session_state.enable_hyde:
            logging.info("启用HyDE扩展 | 历史上下文长度：%d 字符", len(chat_history))
            expanded_query = expand_query(f"{chat_history}\n{processed_query}", uri, model)
            expanded_query = chinese_text_preprocess(expanded_query)  # 🌟 扩展查询也预处理
            logging.info("扩展后查询 | 长度：%d 字符 | 示例：%s",
                         len(expanded_query), expanded_query[:100]+"...")
        else:
            logging.info("HyDE扩展未启用")
            expanded_query = processed_query

        # 🔍 使用中文优化的检索流程
        # 混合检索
        logging.info("执行混合检索 | 最大召回数：%d", st.session_state.max_contexts*2)
        docs = st.session_state.retrieval_pipeline["ensemble"].invoke(
            expanded_query,
            search_kwargs={"k": st.session_state.max_contexts*2}  # 🌟 扩大初始检索范围
        )
        logging.info("初步召回结果 | 文档数：%d", len(docs))
        
        # 🚀 GraphRAG 中文优化
        if st.session_state.enable_graph_rag:
            logging.info("启用GraphRAG增强")
            # 🌟 中文图查询预处理
            graph_query = chinese_text_preprocess(query)
            graph_results = retrieve_from_graph(
                graph_query, 
                st.session_state.retrieval_pipeline["knowledge_graph"]
            )
            logging.info("图谱匹配结果 | 匹配实体数：%d", len(graph_results))
            if graph_results:
                logging.info("TOP3图谱实体：%s", graph_results[:3])
            
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
                logging.info("增强后文档总数：%d", len(docs))

        # 🚀 中文重排序优化
        if st.session_state.enable_reranking:
            logging.info("执行重排序 | 文档数：%d | 批处理大小：%d", len(docs), batch_size)
            # 🌟 使用中文优化的reranker
            reranker = st.session_state.retrieval_pipeline["reranker"]
            pairs = [[processed_query, chinese_text_preprocess(doc.page_content)] for doc in docs]
            
            # 🌟 批量处理提升性能
            batch_size = 32
            scores = []
            logging.info("执行重排序 | 文档数：%d | 批处理大小：%d", len(docs), batch_size)
            for i in range(0, len(pairs), batch_size):
                batch = pairs[i:i+batch_size]
                scores.extend(reranker.predict(batch))
                logging.info("已处理批次：%d/%d", i//batch_size+1, len(pairs)//batch_size+1)
            logging.info("重排序完成 | 最高分：%.2f | 最低分：%.2f", 
                        max(scores) if scores else 0, min(scores) if scores else 0)
            # 按分数排序
            ranked_docs = [doc for _, doc in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)]
        else:
            ranked_docs = docs

        # 🌟 结果后处理
        original_count = len(ranked_docs)
        final_docs = []
        for doc in ranked_docs[:st.session_state.max_contexts]:
            content = doc.page_content
            # 添加中文内容比例阈值控制
            MIN_CHINESE_RATIO = 0.3  # 至少30%字符是中文
            chinese_chars = sum(1 for c in content if '\u4e00' <= c <= '\u9fff')
            if chinese_chars / len(content) < MIN_CHINESE_RATIO:
                continue
            final_docs.append(doc)
        
        logging.info("结果过滤 | 原始数：%d → 最终数：%d | 过滤比例：%.1f%%",
                    original_count, len(final_docs), 
                    (original_count-len(final_docs))/original_count*100 if original_count else 0)

        duration = (datetime.now()-start_time).total_seconds()
        logging.info("✅ 检索流程完成 | 总耗时：%.2fs | 最终返回数：%d", duration, len(final_docs))
        return final_docs
    
    except Exception as e:
        logging.error("检索流程异常终止 | 阶段：%s", "未知", exc_info=True)
        raise


import streamlit as st
import networkx as nx
import jieba
import jieba.posseg as pseg
from text2vec import Similarity

# 加载中文NLP资源
jieba.initialize()
SIM_MODEL = Similarity()

def build_knowledge_graph(docs):
    """优化后的中文知识图谱构建"""
    G = nx.Graph()
    
    for doc in docs:
        # 使用词性标注提取实体
        words = pseg.cut(doc.page_content)
        entities = [
            word.word for word in words 
            if word.flag in ['nr', 'ns', 'nt', 'nz']  # 人名/地名/机构名/其他专名
        ]
        
        # 添加语义关系（使用Text2Vec计算相似度）
        for i in range(len(entities)):
            for j in range(i+1, len(entities)):
                score = SIM_MODEL.get_score(entities[i], entities[j])  # 补上score定义
                if score > 0.6:
                    G.add_edge(entities[i], entities[j], weight=score)  # 使用实际计算值
    
    return G

def retrieve_from_graph(query, G, top_k=5):
    """优化后的中文图谱检索"""
    st.write(f"🔎 知识图谱检索: {query}")
    
    # 中文查询解析
    query_entities = [
        word.word for word in pseg.cut(query)
        if word.flag in ['nr', 'ns', 'nt', 'nz']
    ]
    
    if not query_entities:
        return []

    # 语义扩展检索
    results = []
    for node in G.nodes:
        sim_score = max(SIM_MODEL.get_score(node, ent) for ent in query_entities)
        if sim_score > 0.5:
            results.append((node, sim_score))
    
    # 按相似度排序
    results = sorted(results, key=lambda x: -x[1])[:top_k]
    
    if results:
        st.write(f"🟢 匹配实体: {[x[0] for x in results]}")
        return [x[0] for x in results]
    
    return []


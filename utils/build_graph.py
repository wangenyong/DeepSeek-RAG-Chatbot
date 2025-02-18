import streamlit as st
import networkx as nx
import jieba
import jieba.posseg as pseg
from text2vec import Similarity
import logging
from datetime import datetime

# 加载中文NLP资源
jieba.initialize()
SIM_MODEL = Similarity()

def build_knowledge_graph(docs):
    """优化后的中文知识图谱构建"""
    G = nx.Graph()
    logging.info("🏗️ 开始构建知识图谱 | 输入文档数：%d", len(docs))
    
    total_entities = 0
    total_relations = 0
    start_time = datetime.now()
    
    try:
        for idx, doc in enumerate(docs):
            # 实体提取
            try:
                words = pseg.cut(doc.page_content)
                entities = [
                    word.word for word in words 
                    if word.flag in ['nr', 'ns', 'nt', 'nz']
                ]
                logging.debug("文档[%d/%d] 实体提取 | 原始文本长度：%d | 提取实体数：%d",
                             idx+1, len(docs), len(doc.page_content), len(entities))
                total_entities += len(entities)
                
                if not entities:
                    logging.debug("文档[%d/%d] 未检测到有效实体", idx+1, len(docs))
                    continue

                # 记录示例实体（避免泄露敏感信息）
                sample_entities = [e[:min(5, len(e))]+"..." if len(e)>5 else e for e in entities[:3]]
                logging.debug("文档[%d/%d] 示例实体：%s", idx+1, len(docs), sample_entities)

            except Exception as e:
                logging.error("文档[%d/%d] 实体提取失败", idx+1, len(docs), exc_info=True)
                continue

            # 关系构建
            relation_count = 0
            for i in range(len(entities)):
                for j in range(i+1, len(entities)):
                    try:
                        score = SIM_MODEL.get_score(entities[i], entities[j])
                        if score > 0.6:
                            G.add_edge(entities[i], entities[j], weight=score)
                            relation_count += 1
                            total_relations += 1
                            
                            # 记录高分关系
                            if score > 0.8:
                                logging.debug("发现强关联 | %s ↔ %s | 分数：%.2f",
                                            entities[i], entities[j], score)
                    except Exception as e:
                        logging.error("相似度计算失败 | 实体对：%s-%s",
                                     entities[i], entities[j], exc_info=True)

            logging.debug("文档[%d/%d] 添加关系数：%d", idx+1, len(docs), relation_count)

        duration = (datetime.now() - start_time).total_seconds()
        logging.info("知识图谱构建完成 | 总耗时：%.2fs | 总实体数：%d | 总关系数：%d",
                    duration, total_entities, total_relations)
        logging.debug("图谱统计 | 节点度示例：%s", 
                     dict(list(G.degree())[:5]))  # 显示前5个节点的度

    except Exception as e:
        logging.error("知识图谱构建流程异常终止", exc_info=True)
        raise  # 重新抛出异常以便上层处理

    return G

def retrieve_from_graph(query, G, top_k=5):
    """优化后的中文图谱检索"""
    logging.info("🔍 开始图谱检索 | 查询：%s | 最大结果数：%d", query, top_k)
    start_time = datetime.now()
    
    try:
        # 实体提取
        query_entities = [
            word.word for word in pseg.cut(query)
            if word.flag in ['nr', 'ns', 'nt', 'nz']
        ]
        logging.info("查询解析 | 提取实体数：%d | 实体列表：%s", 
                    len(query_entities), query_entities[:5])  # 限制显示数量

        if not query_entities:
            logging.warning("查询未包含可识别实体 | 查询内容：%s", query[:50]+"..." if len(query)>50 else query)
            return []

        # 语义扩展检索
        results = []
        logging.debug("开始节点遍历 | 总节点数：%d", len(G.nodes))
        for node in G.nodes:
            try:
                sim_score = max(SIM_MODEL.get_score(node, ent) for ent in query_entities)
                if sim_score > 0.5:
                    results.append((node, sim_score))
                    logging.debug("节点匹配 | %s ↔ %s | 最高分：%.2f",
                                 node, query_entities, sim_score)
            except Exception as e:
                logging.error("节点相似度计算失败 | 节点：%s", node, exc_info=True)

        # 结果排序
        sorted_results = sorted(results, key=lambda x: -x[1])[:top_k]
        logging.info("检索完成 | 候选结果数：%d | 最高分：%.2f", 
                    len(sorted_results), sorted_results[0][1] if sorted_results else 0)

        if sorted_results:
            logging.debug("TOP结果示例：%s", ["%s(%.2f)"%(x[0],x[1]) for x in sorted_results[:3]])
            st.write(f"🟢 匹配实体: {[x[0] for x in sorted_results]}")
        else:
            logging.info("无符合阈值的结果")

        duration = (datetime.now() - start_time).total_seconds()
        logging.info("图谱检索完成 | 总耗时：%.2fs", duration)
        
        return [x[0] for x in sorted_results]

    except Exception as e:
        logging.error("图谱检索流程异常终止", exc_info=True)
        raise



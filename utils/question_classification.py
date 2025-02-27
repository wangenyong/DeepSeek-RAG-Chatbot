from transformers import pipeline
import re

# 加载本地部署的轻量级分类模型
classifier = pipeline(
    "text-classification",
    model="facebook/bart-large-mnli",
    device=0  # GPU加速
)

def detect_query_type(query: str) -> str:
    candidate_labels = ["structured_data", "document_content", "hybrid"]
    result = classifier(query, candidate_labels)
    return result[0]['label']

def analyze_query_features(query: str) -> dict:
    return {
        "has_table_mention": any(w in query.lower() for w in ["表", "table"]),
        "has_column_terms": len(re.findall(r"\b(金额|日期|客户编号)\b", query)) > 0,
        "contains_numeric_filter": bool(re.search(r"\d+", query)),
        "vector_score": retrieve_vector_similarity(query)  # 原始RAG的检索置信度
    }
    
def route_decision(query: str) -> str:
    features = analyze_query_features(query)
    intent = detect_query_type(query)
    
    # 优先处理明确的结构化特征
    if features['has_table_mention'] or features['has_column_terms']:
        return "structured"
    
    # 混合意图处理
    if intent == "hybrid":
        return "parallel"
    
    # 数值过滤条件优先走SQL
    if features['contains_numeric_filter'] and features['vector_score'] < 0.7:
        return "structured"
        
    return "vector"  # 默认走向量检索
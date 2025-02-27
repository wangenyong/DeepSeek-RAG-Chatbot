import requests
import os

OLLAMA_BASE_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
OLLAMA_API_URL = f"{OLLAMA_BASE_URL}/api/generate"
MODEL = os.getenv("MODEL", "deepseek-r1:1.5b")  # 🌟 改用中文模型

def is_structured_query(prompt):
    # 基于规则的关键词匹配
    keywords = ['查询', '统计', '数据', '表', '记录', '销售额', '用户数']
    if any(kw in prompt for kw in keywords):
        return True
    
    # 使用小型分类模型（示例）
    classifier_prompt = f"""判断以下问题是否需要查询数据库：
问题：{prompt}
答案（只需回答是或否）："""
    
    response = requests.post(
        OLLAMA_API_URL,
        json={"model": MODEL, "prompt": classifier_prompt, "temperature": 0}
    )
    return "是" in response.json()["response"].strip()
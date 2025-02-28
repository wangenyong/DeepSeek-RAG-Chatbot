import requests
import os
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.agent_toolkits import create_sql_agent
from langchain_ollama import OllamaLLM
import logging


DATABASE_URI = "mysql+pymysql://root:Marmot123@localhost:3306/gmall?charset=utf8mb4"
OLLAMA_BASE_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
OLLAMA_API_URL = f"{OLLAMA_BASE_URL}/api/generate"
MODEL = os.getenv("MODEL", "deepseek-r1:1.5b")  # 🌟 改用中文模型

class DBAgent:
    def __init__(self):
        self.db = SQLDatabase.from_uri(DATABASE_URI)
        self.toolkit = SQLDatabaseToolkit(db=self.db, llm=OllamaLLM(model=MODEL))
        self.agent = create_sql_agent(
            llm=OllamaLLM(model=MODEL),
            toolkit=self.toolkit,
            verbose=True,
            handle_parsing_errors=True
        )
    
    def query(self, question):
        logging.info(f"Querying database with question: {question}")
        try:
            prompt_template = """请按以下步骤操作：
            1. 生成用于解决此问题的 SQL 查询。
            2. 执行查询并返回结果。

            问题：{question}
            """
            result = self.agent.invoke({
                "input": prompt_template.format(question=question)
            })
            logging.info(f"Query result: {result}")
            return {
                "data": result["output"]
            }
        except Exception as e:
            logging.error(f"Query failed: {str(e)}")
            return {"error": f"查询失败：{str(e)}"}
    
    def summarize_result(self, result):
        summary_prompt = f"""将以下SQL结果转换为自然语言：
        原始数据：{result}
        用中文总结要点："""
        
        response = requests.post(
            OLLAMA_API_URL,
            json={"model": MODEL, "prompt": summary_prompt}
        )
        return response.json()["response"]
import requests
import os
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.llms import ollama

# 初始化数据库连接（以 PostgreSQL 为例）
db = SQLDatabase.from_uri("postgresql://user:password@localhost/mydb")

# 执行 SQL 查询
result = db.run("SELECT name FROM users LIMIT 5")
print(result)

DATABASE_URI = "mysql+pymysql://user:password@localhost:3306/dbname?charset=utf8mb4"
OLLAMA_BASE_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
OLLAMA_API_URL = f"{OLLAMA_BASE_URL}/api/generate"
MODEL = os.getenv("MODEL", "deepseek-r1:1.5b")  # 🌟 改用中文模型

class DBAgent:
    def __init__(self):
        self.db = SQLDatabase.from_uri(DATABASE_URI)
        self.toolkit = SQLDatabaseToolkit(db=self.db, llm=ollama(model=MODEL))
        self.agent = create_sql_agent(
            llm=ollama(model=MODEL),
            toolkit=self.toolkit,
            verbose=True,
            handle_parsing_errors=True
        )
    
    def query(self, question):
        try:
            result = self.agent.invoke({
                "input": f"请生成SQL查询并解释结果。问题：{question}"
            })
            return {
                "data": result["output"],
                "summary": self.summarize_result(result)
            }
        except Exception as e:
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
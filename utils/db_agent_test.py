import requests
import os
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit, create_sql_agent
from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor
import logging

DATABASE_URI = "mysql+pymysql://root:Marmot123@localhost:3306/gmall?charset=utf8mb4"
OLLAMA_BASE_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
MODEL = "deepseek-r1:7b"  # 修正2：确认正确的模型名称

class DBAgent:
    def __init__(self):
        # 初始化数据库连接
        self.db = SQLDatabase.from_uri(DATABASE_URI)
        
        # 初始化Ollama模型（修正3：使用正确的Chat模型）
        self.llm = ChatOllama(
            model=MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.1,
            format="json"
        )
        
        # 创建SQL Toolkit（修正4：正确的toolkit初始化方式）
        self.toolkit = SQLDatabaseToolkit(
            db=self.db,
            llm=self.llm
        )
        
        # 创建SQL Agent（修正5：使用新的agent创建方式）
        self.agent: AgentExecutor = create_sql_agent(
            llm=self.llm,
            toolkit=self.toolkit,
            verbose=True,
            handle_parsing_errors=True,
            agent_type="openai-tools"  # 使用更稳定的agent类型
        )
    
    def query(self, question: str):
        logging.info(f"Querying database with question: {question}")
        try:
            # 强化格式要求的提示词
            structured_prompt = f"""请严格按照以下步骤执行：
            1. 使用中文思考问题：'{question}'
            2. 生成SQL查询（必须使用sql_db_query工具）
            3. 返回结果时严格使用以下格式：
            
            Action: 要执行的动作
            Action Input: 输入的参数
            Observation: 执行结果分析
            Final Answer: 最终自然语言结论（中文）

            示例：
            用户问：用户数量是多少？
            你应输出：
            Action: sql_db_query
            Action Input: SELECT COUNT(*) FROM user_table
            Observation: 结果为 1500
            Final Answer: 系统中共有1500名注册用户。

            现在请处理：{question}
            """
            
            result = self.agent.invoke({"input": structured_prompt})
            
            return {
                "data": result["output"]
            }
        except Exception as e:
            logging.exception("Database query failed")  # 记录完整堆栈信息
            return {"error": f"查询失败：{str(e)}"}

# 使用示例
if __name__ == "__main__":
    agent = DBAgent()
    print(agent.query("查询表users_info有多少记录"))
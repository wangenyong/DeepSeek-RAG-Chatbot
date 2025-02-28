from smolagents import Tool, CodeAgent
import sqlalchemy
from typing import Dict, Any
import logging
from smolagents import LiteLLMModel  # 使用正确的Ollama模型类

# 数据库配置（根据网页[3]硬件要求）
DATABASE_URI = "mysql+pymysql://root:Marmot123@localhost:3306/gmall?charset=utf8mb4"

class SQLQueryTool(Tool):
    """自定义数据库查询工具（参考网页3的Tool类定义）"""
    def __init__(self):
        self.engine = sqlalchemy.create_engine(DATABASE_URI)
    
    @property
    def name(self) -> str:
        return "database_query"
    
    @property
    def description(self) -> str:
        return """执行SQL查询并返回结构化结果"""
    
    @property
    def inputs(self) -> Dict:
        return {
            "query": {
                "type": "string",
                "description": "SQL查询语句",
                "nullable": True  # 关键修正点1
            },
            "params": {
                "type": "object",
                "description": "查询参数",
                "default": {},
                "nullable": True
            }
        }
    
    @property
    def output_type(self) -> str:  # 必须返回字符串类型（网页3[3](@ref)）
        return "object"  # 对应Python的字典类型
    

    def forward(self, query: str = None, params: dict = None) -> Dict:  # 关键修正点2
        """关键修正点：参数名与inputs定义一致"""
        params = params or {}
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    sqlalchemy.text(query), 
                    params
                )
                return {
                    "columns": list(result.keys()),
                    "rows": [dict(row) for row in result.mappings()]
                }
        except Exception as e:
            logging.error(f"SQL执行失败: {str(e)}")
            return {"error": str(e)}

class DBAgent:
    """数据库查询智能体（根据网页4的Ollama集成最佳实践优化）"""
    def __init__(self):
        self.agent = CodeAgent(
            tools=[],
            model=self._create_ollama_model()
        )
        
    def _create_ollama_model(self):
        """创建本地Ollama模型实例"""
        
        return LiteLLMModel(
            model_id="ollama_chat/deepseek-r1:7b",
            api_base="http://127.0.0.1:11434",
            api_key="YOUR_API_KEY", # replace with API key if necessary
            num_ctx=8192,
        )
        
    def run(self, question: str) -> str:
        return self.agent.run(question)
    

    def _parse_response(self, response: Dict) -> str:
        """结果解析（增强错误处理）"""
        try:
            if "error" in response:
                return f"❌ 错误：{response['error']}"
            
            # 调整字段映射关系
            action_input = response.get("action_input", {})
            return f"{response.get('answer', '')}\n\n🔍 生成SQL：\n{action_input.get('query', '')}"
        except Exception as e:
            return f"⚠️ 结果解析失败：{str(e)}"

# 初始化测试（参考网页[3]验证流程）
if __name__ == "__main__":
    agent = DBAgent()
    
    agent.run("写一篇春节祝福语，使用中文回答")
    
    # 测试用例
    # test_cases = [
    #     "查询用户数量",
    # ]
    
    # for case in test_cases:
    #     print(f"### 查询：{case}")
    #     print(agent.query(case))
    #     print("\n" + "="*50 + "\n")
import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils.db_conn import get_db

tokenizer = AutoTokenizer.from_pretrained("tscholak/nl2sql-spider")
model = AutoModelForSeq2SeqLM.from_pretrained("tscholak/nl2sql-spider")

def nl2sql_converter(query: str, schema: dict) -> str:
    inputs = f"question: {query} context: {schema}"
    input_ids = tokenizer(inputs, return_tensors="pt").input_ids
    outputs = model.generate(input_ids)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def safe_query_executor(sql: str):
    # 添加安全限制
    if "DROP" in sql.upper() or "DELETE" in sql.upper():
        return "危险操作被阻止"
    
    # 执行查询
    with get_db() as db:
        try:
            result = db.execute(sql).fetchall()
            return str(result[:10])  # 限制返回结果数量
        except Exception as e:
            return f"查询错误: {str(e)}"

def handle_structured_query(query: str) -> str:
    """处理纯结构化数据查询"""
    try:
        # 生成SQL
        generated_sql = nl2sql_converter(query, get_cached_schema())
        
        # 安全执行
        if validate_sql(generated_sql):
            result = safe_query_executor(generated_sql)
            return format_sql_result(result)
        else:
            return "SQL验证未通过，请重新表述问题"
            
    except Exception as e:
        logging.error(f"SQL处理失败: {str(e)}")
        return f"数据库查询错误: {str(e)}"
    
    
def handle_hybrid_query(query: str):
    """处理混合型查询"""
    from concurrent.futures import ThreadPoolExecutor
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        # 并行执行文档检索和SQL生成
        future_docs = executor.submit(retrieve_documents, query, OLLAMA_API_URL, MODEL, "")
        future_sql = executor.submit(generate_sql_background, query)
        
        docs = future_docs.result(timeout=10)
        sql_data = future_sql.result(timeout=8)
    
    return docs, sql_data

def generate_sql_background(query: str):
    """后台生成SQL并执行"""
    try:
        sql = nl2sql_converter(query, get_cached_schema())
        return safe_query_executor(sql)
    except:
        return None
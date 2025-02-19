import streamlit as st
import requests
import json
import jieba  # 🌟 新增中文分词
from utils.retriever_pipeline import retrieve_documents
from utils.doc_handler import process_documents
from utils.chinese_tools import chinese_text_preprocess
from sentence_transformers import CrossEncoder
import torch
import os
from dotenv import load_dotenv, find_dotenv
import logging
from logging.handlers import RotatingFileHandler
import sys
import time

# 在文件头部添加请求ID用于追踪
import uuid
current_request_id = str(uuid.uuid4())[:8]

def setup_logging():
    root_logger = logging.getLogger()
    
    # 关键修复：清理现有handler
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 创建日志格式
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s @ %(module)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 文件处理器（自动轮转）
    file_handler = RotatingFileHandler(
        'logs/app.log',
        maxBytes=1024*1024*5,  # 5MB
        backupCount=3,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)

    # 控制台处理器
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    # 配置根日志
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)

# 初始化日志系统
setup_logging()

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]
load_dotenv(find_dotenv())

OLLAMA_BASE_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
OLLAMA_API_URL = f"{OLLAMA_BASE_URL}/api/generate"
MODEL = os.getenv("MODEL", "deepseek-r1:1.5b")  # 🌟 改用中文模型
EMBEDDINGS_MODEL = "moka-ai/m3e-base"  # 🌟 中文嵌入模型
CROSS_ENCODER_MODEL = "BAAI/bge-reranker-base"  # 🌟 中文重排序模型

device = "cuda" if torch.cuda.is_available() else "cpu"

# 🌟 初始化中文模型
if "jieba_initialized" not in st.session_state:
    jieba.initialize()
    jieba.load_userdict("data/custom_words.txt")  # 自定义词典
    st.session_state.jieba_initialized = True

reranker = None
try:
    # 🌟 使用中文重排序器
    # 初始化模型
    reranker = CrossEncoder(
        CROSS_ENCODER_MODEL,
        device=device,
        max_length=512
    )
    
except OSError as e:
    st.error(f"模型加载失败: {str(e)} (请检查网络或模型路径)")
except RuntimeError as e:
    if "CUDA out of memory" in str(e):
        st.error("显存不足，尝试减小 batch size 或使用 CPU 模式")
    else:
        st.error(f"运行时错误: {str(e)}")
except Exception as e:
    st.error(f"未知错误: {str(e)}")

# 🌟 汉化界面
st.set_page_config(page_title="深度图谱智能检索系统", layout="wide")

# 🌟 中文CSS样式
st.markdown("""
    <style>
        .stApp { background-color: #f4f4f9; }
        h1 { color: #00FF99; text-align: center; }
        .stChatMessage { border-radius: 10px; padding: 10px; margin: 10px 0; }
        .stChatMessage.user { background-color: #e8f0fe; }
        .stChatMessage.assistant { background-color: #d1e7dd; }
        .stButton>button { background-color: #00AAFF; color: white; }
    </style>
""", unsafe_allow_html=True)

# 🌟 中文会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retrieval_pipeline" not in st.session_state:
    st.session_state.retrieval_pipeline = None
if "rag_enabled" not in st.session_state:
    st.session_state.rag_enabled = False
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False

# 🌟 侧边栏汉化
with st.sidebar:
    st.header("📁 文档管理")
    uploaded_files = st.file_uploader(
        "上传文档（支持PDF/DOCX/TXT）",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )
    
    # 在文档处理部分添加：
    if uploaded_files and not st.session_state.documents_loaded:
        try:
            logging.info(f"开始处理文档上传：收到{len(uploaded_files)}个文件")
            with st.spinner("文档处理中..."):
                process_documents(uploaded_files, reranker, EMBEDDINGS_MODEL, device)
                logging.info(f"文档处理完成，文件名：{[f.name for f in uploaded_files]}")
                st.success("文档处理完成！")
        except Exception as e:
            logging.error("文档处理失败", exc_info=True)
            st.error(f"文档处理失败: {str(e)}")
    
    st.markdown("---")
    st.header("⚙️ 检索设置")
    
    st.session_state.rag_enabled = st.checkbox("启用智能检索", value=True)
    st.session_state.enable_hyde = st.checkbox("启用查询扩展", value=True)
    st.session_state.enable_reranking = st.checkbox("启用神经重排序", value=True)
    st.session_state.enable_graph_rag = st.checkbox("启用知识图谱", value=True)
    st.session_state.temperature = st.slider("生成温度", 0.0, 1.0, 0.3, 0.05)
    st.session_state.max_contexts = st.slider("最大上下文", 1, 5, 3)
    
    if st.button("清空对话历史"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("""
        <div style="font-size: 12px; color: gray;">
            <b>开发者：</b>wangenyong &copy; 版权所有 2025
        </div>
    """, unsafe_allow_html=True)

# 🌟 主界面汉化
st.title("🤖 深度图谱智能检索系统")
st.caption("集成知识图谱、混合检索与神经重排序的先进问答系统")


# 对话显示
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("请输入您的问题..."):
    # 🌟 中文预处理
    processed_prompt = chinese_text_preprocess(prompt)
    
    chat_history = "\n".join([msg["content"] for msg in st.session_state.messages[-5:]])
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        # 🌟 中文优化上下文构建
        context = ""
        # 在检索过程添加日志：
        if st.session_state.rag_enabled and st.session_state.retrieval_pipeline:
            try:
                logging.info(f"开始文档检索 | 查询：{processed_prompt}")
                docs = retrieve_documents(processed_prompt, OLLAMA_API_URL, MODEL, chat_history)
                logging.info(f"检索完成 | 获得{docs and len(docs) or 0}条相关文档")
                context = "\n".join(
                    f"[来源 {i+1}]: {doc.page_content}" 
                    for i, doc in enumerate(docs)
                )
            except Exception as e:
                logging.error("文档检索失败", exc_info=True)
                st.error(f"检索错误: {str(e)}")
        
        try:
            # 🌟 初始化关键变量
            full_response = ""
            response_buffer = b""
            token_count = 0
            start_time = time.time()
            
            logging.info(f"[{current_request_id}] 开始处理请求 | 温度={st.session_state.temperature} | 最大上下文={st.session_state.max_contexts}")
            
            # 🌟 增强提示词结构
            system_prompt = f"""基于以下上下文用中文回答问题(步骤):
                1️⃣【实体识别】提取关键人物、地点、时间
                2️⃣【一致性分析】对比{st.session_state.max_contexts}个来源的异同
                3️⃣【综合推理】结合历史对话分析：
                {chat_history[-3:] if chat_history else "无历史"}
                4️⃣【结构化输出】按要点分项说明

                📚上下文：{context[:1000]}...（共{len(context)}字）
                ❓问题：{prompt}
                🖋答案："""
                
            logging.info(f"[{current_request_id}] 完整提示词:\n{system_prompt}")

            # 🌟 增强请求超时设置
            response = requests.post(
                OLLAMA_API_URL,
                json={
                    "model": MODEL,
                    "prompt": system_prompt,
                    "stream": True,
                    "options": {
                        "temperature": max(0.1, min(st.session_state.temperature, 1.0)),  # 温度值安全限制
                        "num_ctx": 4096,
                        "stop": ["\n\n", "<|endoftext|>", "答案："]  # 防止模型自重复
                    }
                },
                stream=True
            )
            logging.info(f"[{current_request_id}] API请求成功 | 状态码: {response.status_code}")
            
            # 🌟 改进的流式处理
            for raw_chunk in response.iter_content(chunk_size=512):
                if raw_chunk:
                    response_buffer += raw_chunk
                    # 处理分块可能包含多个JSON的情况
                    while b'\n' in response_buffer:
                        line, response_buffer = response_buffer.split(b'\n', 1)
                        if not line:
                            continue
                        # 记录原始数据（调试用）
                        line_debug = line.decode('utf-8', errors='replace')
                        logging.info(f"原始数据: {line_debug}")
                        try:
                            data = json.loads(line.decode('utf-8'))
                            # 多字段兼容
                            token = data.get("response") or data.get("content") or data.get("text", "")
                            
                            if token:
                                token_count += 1
                                full_response += token
                                # 流式更新频率控制（每3个token或0.5秒更新一次）
                                if token_count % 3 == 0 or (time.time() - start_time) > 0.5:
                                    response_placeholder.markdown(full_response + "▌")
                                    start_time = time.time()
                            
                            # 结束条件判断
                            if data.get("done", False):
                                logging.info(f"[{current_request_id}] 收到结束标记 | 最后数据: {data}")
                                break
                                
                        except json.JSONDecodeError as e:
                            logging.warning(f"[{current_request_id}] JSON解析异常 | 数据块: {line} | 错误: {str(e)}")
                        except Exception as e:
                            logging.error(f"[{current_request_id}] 流处理异常 | 类型: {type(e).__name__} | 错误: {str(e)} | 原始数据: {line}", exc_info=True)

        except StopIteration:
            logging.info(f"[{current_request_id}] 流式响应正常结束")
        except requests.exceptions.Timeout:
            logging.error(f"[{current_request_id}] 请求超时 | 已接收数据: {len(full_response)}字")
            st.error("响应超时，请简化问题重试")
        except Exception as e:
            logging.critical(f"[{current_request_id}] 未捕获异常", exc_info=True)
            st.error("系统内部错误，请联系管理员")
        finally:
            # 🌟 最终处理
            if full_response:
                response_placeholder.markdown(full_response)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "request_id": current_request_id  # 用于后续追踪
                })
                logging.info(f"[{current_request_id}] 响应完成 | 总耗时: {time.time()-start_time:.2f}s | 总token数: {token_count}")
            else:
                st.error("未能生成有效响应")
                logging.warning(f"[{current_request_id}] 空响应 | 缓冲区大小: {len(response_buffer)}")

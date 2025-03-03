import streamlit as st
from utils.doc_handler import process_documents
import logging

st.set_page_config(page_title="文档管理 - PEACOCK", layout="wide")

st.title("📁 文档管理中心")

# 显示已加载文档
if "uploaded_files" in st.session_state and st.session_state.uploaded_files:
    st.subheader("已加载文档列表")
    cols = st.columns([2, 1, 1])
    with cols[0]:
        st.write("**文件名**")
    with cols[1]:
        st.write("**状态**")
    with cols[2]:
        st.write("**操作**")
    
    for idx, file in enumerate(st.session_state.uploaded_files):
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.write(file.name)
        with col2:
            st.success("已就绪" if st.session_state.documents_loaded else "待处理")
        with col3:
            if st.button("删除", key=f"del_{idx}"):
                del st.session_state.uploaded_files[idx]
                st.session_state.documents_loaded = False
                st.rerun()
    
    # 文档处理控制
    if st.button("重新处理所有文档"):
        try:
            with st.spinner("文档重新处理中..."):
                process_documents(
                    st.session_state.uploaded_files,
                    st.session_state.retrieval_pipeline["reranker"],
                    "moka-ai/m3e-base",
                    "cuda" if st.session_state.retrieval_pipeline else "cpu"
                )
                st.success("文档处理完成！")
        except Exception as e:
            logging.error("文档处理失败", exc_info=True)
            st.error(f"文档处理失败: {str(e)}")
else:
    st.info("当前没有已加载的文档")

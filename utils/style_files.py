

app_style = """
    <style>
        /* 主标题居中 */
        h1 {
            text-align: center;
            font-family: 'Arial Black', sans-serif;
            color: #2D4263;
            text-shadow: 2px 2px 4px rgba(45,66,99,0.1);
        }
        
        /* 增加顶部间距 */
        .stApp {
            margin-top: -50px;
            padding-top: 80px;
        }
    </style>
    """

thinking_loading_style = """
    <div style="display: flex; align-items: center; gap: 0.8rem; color: #4a4a4a; position: relative; top: -6px;">
        <div class="loader"></div>
        <div>正在思考中...</div>
    </div>
    <style>
    .loader {
        border: 3px solid #f3f3f3;
        border-radius: 50%;
        border-top: 3px solid #409EFF;
        width: 24px;
        height: 24px;
        animation: spin 1s linear infinite;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    </style>
    """

thinking_style = """
    <div style="
        background: #f8f9fa;
        border-left: 4px solid #6c757d;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
        color: #495057;
    ">
    💡 思考过程：<br>
    {}
    </div>
    """

answer_style = """
    <div style="
        background: #e9f5e9;
        border: 2px solid #28a745;
        padding: 1.25rem;
        margin: 1.5rem 0;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    ">
    {}
    </div>
    """
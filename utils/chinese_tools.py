import re
import jieba # 🌟 中文分词工具

# 🌟 中文停用词列表
STOP_WORDS = set(["的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一个"])

def chinese_text_preprocess(text):
    """🌟 中文文本预处理"""
    # 去除特殊字符
    text = re.sub(r'[^\w\s\u4e00-\u9fa5]', '', text)
    # 分词处理
    words = jieba.cut(text)
    # 去除停用词
    words = [w for w in words if w not in STOP_WORDS]
    return ' '.join(words)
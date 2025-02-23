from logging.handlers import RotatingFileHandler
import logging
import sys

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
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# 配置数据库连接
DATABASE_URI = "mysql+pymysql://user:password@localhost:3306/dbname?charset=utf8mb4"

engine = create_engine(
    DATABASE_URI,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 获取数据库会话
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
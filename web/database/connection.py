"""
데이터베이스 연결 관리
"""

import os
from urllib.parse import quote_plus
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import QueuePool

# 환경변수에서 DB 설정 읽기
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_USER = os.getenv("DB_USER", "dmillion")
DB_PASSWORD = os.getenv("DB_PASSWORD", "dm250120@")
DB_NAME = os.getenv("DB_NAME", "defect_detection_db")

# ✅ 비밀번호 URL 인코딩 (특수문자 처리)
encoded_password = quote_plus(DB_PASSWORD)

# MariaDB 연결 URL
DATABASE_URL = f"mysql+pymysql://{DB_USER}:{encoded_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"

# SQLAlchemy 엔진 생성
engine = create_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20
)

# 세션 팩토리
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base 클래스
Base = declarative_base()


def get_db():
    """
    DB 세션 의존성
    
    Usage:
        @app.get("/items")
        def read_items(db: Session = Depends(get_db)):
            return db.query(Item).all()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """
    데이터베이스 초기화 (테이블 생성)
    주의: 실제 운영에서는 Alembic 사용 권장
    """
    Base.metadata.create_all(bind=engine)
    print("✅ 데이터베이스 테이블 생성 완료")


def test_connection():
    """
    DB 연결 테스트
    """
    try:
        db = SessionLocal()
        db.execute("SELECT 1")
        db.close()
        print("✅ 데이터베이스 연결 성공")
        return True
    except Exception as e:
        print(f"❌ 데이터베이스 연결 실패: {e}")
        return False
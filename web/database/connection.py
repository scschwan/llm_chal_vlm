"""
데이터베이스 연결 관리
"""

import os
from urllib.parse import quote_plus
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import QueuePool
import pymysql

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
    pool_size=5,              # ✅ 연결 풀 크기 축소
    max_overflow=10,          # ✅ 오버플로우 축소
    pool_recycle=3600,        # ✅ 1시간마다 연결 재생성
    pool_timeout=30,          # ✅ 연결 대기 타임아웃
)

# 세션 팩토리
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base 클래스
Base = declarative_base()


def get_db():
    """
    DB 세션 의존성 (SQLAlchemy ORM용)
    
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


def get_db_connection():
    """
    ✅ 직접 연결이 필요한 경우 사용 (pymysql 연결)
    반드시 사용 후 close() 호출 필요
    
    Usage:
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            # ... 쿼리 실행
        finally:
            cursor.close()
            conn.close()
    """
    return pymysql.connect(
        host=DB_HOST,
        port=int(DB_PORT),
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )


def init_db():
    """
    데이터베이스 초기화 (테이블 생성)
    """
    Base.metadata.create_all(bind=engine)
    print("✅ 데이터베이스 테이블 생성 완료")


def test_connection():
    from sqlalchemy import text
    """
    DB 연결 테스트
    """
    try:
        db = SessionLocal()
        #db.execute("SELECT 1 from products;")
        query = text("""
            SELECT 1
        """)
        db.execute(query)
        db.close()
        print("✅ 데이터베이스 연결 성공")
        return True
    except Exception as e:
        print(f"❌ 데이터베이스 연결 실패: {e}")
        return False


def dispose_engine():
    """
    ✅ 연결 풀 정리 (애플리케이션 종료 시 호출)
    """
    global engine
    if engine:
        engine.dispose()
        print("✅ 데이터베이스 연결 풀 정리 완료")


from contextlib import contextmanager

@contextmanager
def get_db_context():
    """
    Context Manager 방식의 DB 세션
    
    Usage:
        with get_db_context() as db:
            # DB 작업
            db.query(...)
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
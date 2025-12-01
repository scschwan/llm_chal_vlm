# 작업자 웹 애플리케이션 아키텍처 정의서 및 시스템 설계서

**프로젝트명**: llm_chal_vlm (작업자 페이지)  
**작성일**: 2025-11-24  
**버전**: 1.0  
**기반 문서**: admin_final_architecture.md (관리자 페이지)

---

## 목차

1. [시스템 개요](#1-시스템-개요)
2. [아키텍처 개요](#2-아키텍처-개요)
3. [기술 스택](#3-기술-스택)
4. [레이어드 아키텍처](#4-레이어드-아키텍처)
5. [API 설계](#5-api-설계)
6. [도메인 모델](#6-도메인-모델)
7. [외부 시스템 연동](#7-외부-시스템-연동)
8. [화면 구성](#8-화면-구성)
9. [처리 흐름](#9-처리-흐름)
10. [배포 구성](#10-배포-구성)

---

## 1. 시스템 개요

### 1.1 목적

llm_chal_vlm의 작업자 페이지는 제조 현장의 **품질 검사 작업자**를 위한 웹 애플리케이션입니다.
불량 이미지를 업로드하고, AI 기반 유사도 검색, 이상 영역 검출, 대응 매뉴얼 자동 생성 기능을 제공합니다.

### 1.2 핵심 기능

| 순서 | 탭 | 기능 | 설명 |
|------|-----|------|------|
| 1 | 이미지 업로드 | 검사 이미지 등록 | 불량 의심 이미지 업로드 |
| 2 | 유사도 검색 | TOP-K 검색 | CLIP 기반 유사 불량 이미지 검색 |
| 3 | 이상 검출 | PatchCore | 정상 이미지 대비 이상 영역 시각화 |
| 4 | 대응 매뉴얼 | LLM/VLM 생성 | AI 기반 조치 방안 자동 생성 |

### 1.3 시스템 관계도

```
┌─────────────────────────────────────────────────────────────────────┐
│                        전체 시스템 구성도                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────────┐         ┌─────────────────┐                  │
│   │ 관리자 브라우저  │         │ 작업자 브라우저   │                  │
│   └────────┬────────┘         └────────┬────────┘                  │
│            │                           │                           │
│            ▼                           ▼                           │
│   ┌─────────────────┐         ┌─────────────────┐                  │
│   │  llm_chal_web   │◄───────►│  llm_chal_vlm   │                  │
│   │  (관리자 페이지) │  HTTP   │  (작업자 페이지) │                  │
│   │  Spring Boot    │         │  FastAPI        │                  │
│   │  Port: 8080     │         │  Port: 8000     │                  │
│   └────────┬────────┘         └────────┬────────┘                  │
│            │                           │                           │
│            │                           ▼                           │
│            │                  ┌─────────────────┐                  │
│            │                  │   LLM Server    │                  │
│            │                  │   Port: 5001    │                  │
│            │                  └─────────────────┘                  │
│            │                           │                           │
│            ▼                           ▼                           │
│   ┌─────────────────────────────────────────────┐                  │
│   │              MariaDB Database               │                  │
│   │              (공용 데이터베이스)             │                  │
│   └─────────────────────────────────────────────┘                  │
│            │                                                       │
│            ▼                                                       │
│   ┌─────────────────────────────────────────────┐                  │
│   │         NCP Object Storage (dm-obs)         │                  │
│   │         (파일 저장소)                        │                  │
│   └─────────────────────────────────────────────┘                  │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### 1.4 서버 간 역할 분담

| 서버 | 포트 | 역할 |
|------|------|------|
| API Server (api_server.py) | 8000 | 웹 UI 제공, CLIP/PatchCore 처리, RAG 검색 |
| LLM Server (llm_server.py) | 5001 | HyperCLOVAX, EXAONE, LLaVA 모델 추론 |

---

## 2. 아키텍처 개요

### 2.1 아키텍처 패턴

**Router-based Modular Architecture (라우터 기반 모듈 아키텍처)** 채택

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Application                      │
├─────────────────────────────────────────────────────────────┤
│              Router Layer (API 라우팅)                       │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐        │
│  │ upload   │ │ search   │ │ anomaly  │ │ manual   │        │
│  │ router   │ │ router   │ │ router   │ │ router   │        │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘        │ 
├─────────────────────────────────────────────────────────────┤
│              Service Layer (비즈니스 로직)                   │
│  ┌──────────────────┐ ┌──────────────────┐                  │
│  │ SimilarityMatcher│ │ AnomalyDetector  │                  │
│  │ (CLIP + FAISS)   │ │ (PatchCore)      │                  │
│  └──────────────────┘ └──────────────────┘                  │
│  ┌──────────────────┐ ┌──────────────────┐                  │
│  │ DefectMapper     │ │ RAG Manager      │                  │
│  │ (불량 정보 매핑)  │ │ (매뉴얼 검색)     │                  │
│  └──────────────────┘ └──────────────────┘                  │
├─────────────────────────────────────────────────────────────┤
│              Data Layer (데이터 접근)                        │
│  ┌──────────────────┐ ┌──────────────────┐                  │
│  │ Database CRUD    │ │ Object Storage   │                  │
│  │ (SQLAlchemy)     │ │ (boto3)          │                  │
│  └──────────────────┘ └──────────────────┘                  │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 디렉토리 구조

```
llm_chal_vlm/
├── web/                           # 웹 애플리케이션
│   ├── api_server.py              # FastAPI 메인 서버
│   ├── routers/                   # API 라우터
│   │   ├── __init__.py
│   │   ├── upload.py              # 이미지 업로드
│   │   ├── search.py              # 유사도 검색 (V1)
│   │   ├── search_v2.py           # 유사도 검색 (V2 - DB 기반)
│   │   ├── anomaly.py             # 이상 검출
│   │   ├── manual.py              # 대응 매뉴얼 생성
│   │   ├── auth.py                # 인증
│   │   └── admin/                 # 관리자 API (별도 서버 연동)
│   ├── pages/                     # HTML 페이지
│   │   ├── upload.html
│   │   ├── search.html
│   │   ├── anomaly.html
│   │   ├── manual.html
│   │   └── login.html
│   ├── static/                    # 정적 파일
│   │   ├── css/
│   │   └── js/
│   ├── database/                  # 데이터베이스
│   │   ├── connection.py          # DB 연결
│   │   ├── models.py              # SQLAlchemy 모델
│   │   └── crud.py                # CRUD 함수
│   └── utils/                     # 유틸리티
│       ├── object_storage.py      # NCP OBS 연동
│       ├── session_helper.py      # 세션 관리
│       └── auth.py                # 인증 유틸
├── modules/                       # AI 모듈
│   ├── similarity_matcher.py      # CLIP 유사도 매칭 (V1)
│   ├── similarity_matcher_v2.py   # CLIP 유사도 매칭 (V2 - DB 기반)
│   ├── anomaly_detector.py        # PatchCore 이상 검출
│   └── vlm/                       # VLM/LLM 관련
│       ├── rag.py                 # RAG 매뉴얼 검색
│       ├── defect_mapper.py       # 불량 정보 매핑
│       ├── llm_inference.py       # LLM 추론
│       └── vlm_inference.py       # VLM 추론
├── llm_server/                    # LLM 전용 서버
│   └── llm_server.py              # HyperCLOVAX, EXAONE, LLaVA
├── data/                          # 데이터
│   ├── patchCore/                 # 제품별 메모리뱅크
│   └── def_split/                 # 불량 이미지
└── manual_store/                  # 매뉴얼 PDF 및 인덱스
    └── unified_index/
```

---

## 3. 기술 스택

### 3.1 Backend

| 구분 | 기술 | 버전 |
|------|------|------|
| Language | Python | 3.9 |
| Framework | FastAPI | 0.104+ |
| ASGI Server | Uvicorn | - |
| ORM | SQLAlchemy | 2.0+ |
| HTTP Client | httpx | - |

### 3.2 AI/ML

| 구분 | 기술 | 용도 |
|------|------|------|
| CLIP | OpenCLIP (ViT-B-32) | 이미지 임베딩 |
| Vector Search | FAISS | 유사도 검색 |
| Anomaly Detection | PatchCore | 이상 영역 검출 |
| RAG | LangChain + FAISS | 매뉴얼 검색 |

### 3.3 LLM/VLM 모델

| 모델 | 크기 | 용도 |
|------|------|------|
| HyperCLOVAX-1.5B | 1.5B | 텍스트 기반 분석 |
| EXAONE-3.5-2.4B | 2.4B | 텍스트 기반 분석 |
| LLaVA-1.5-7B | 7B | 이미지 포함 분석 |

### 3.4 Database

| 구분 | 기술 | 버전 |
|------|------|------|
| RDBMS | MariaDB | 10.5+ |
| Driver | mariadb-connector-python | - |

### 3.5 Cloud Services

| 구분 | 서비스 | 용도 |
|------|--------|------|
| Object Storage | NCP Object Storage | 이미지/매뉴얼 저장 |
| SDK | boto3 (S3 호환) | OBS 연동 |

### 3.6 Frontend

| 구분 | 기술 | 용도 |
|------|------|------|
| Template | HTML + Jinja2 | SSR 렌더링 |
| CSS | Custom CSS | 스타일링 |
| JavaScript | Vanilla JS | 클라이언트 로직 |

---

## 4. 레이어드 아키텍처

### 4.1 Router Layer (API 라우팅)

**역할**: HTTP 요청/응답 처리, API 엔드포인트 정의

| Router | 경로 | 역할 |
|--------|------|------|
| upload | `/upload/**` | 이미지 업로드 |
| search | `/search/**` | 유사도 검색 |
| anomaly | `/anomaly/**` | 이상 검출 |
| manual | `/manual/**` | 대응 매뉴얼 생성 |
| auth | `/auth/**` | 인증 |

### 4.2 Service Layer (비즈니스 로직)

| 서비스 | 클래스 | 역할 |
|--------|--------|------|
| 유사도 매칭 | SimilarityMatcher | CLIP 임베딩 + FAISS 검색 |
| 이상 검출 | AnomalyDetector | PatchCore 기반 이상 영역 검출 |
| 불량 매핑 | DefectMapper | 제품/불량 정보 조회 |
| RAG | RAGManager | 매뉴얼 검색 및 컨텍스트 추출 |

### 4.3 Data Layer (데이터 접근)

| 구성요소 | 파일 | 역할 |
|----------|------|------|
| DB 연결 | connection.py | SQLAlchemy 세션 관리 |
| 모델 | models.py | ORM 엔티티 정의 |
| CRUD | crud.py | 데이터 조작 함수 |
| OBS | object_storage.py | Object Storage 연동 |

---

## 5. API 설계

### 5.1 이미지 업로드 API

**Router**: `web/routers/upload.py`

| 메서드 | 경로 | 기능 |
|--------|------|------|
| POST | `/upload/image` | 이미지 업로드 |
| GET | `/upload/list` | 업로드 파일 목록 |
| DELETE | `/upload/clean` | 업로드 디렉토리 정리 |
| DELETE | `/upload/file/{filename}` | 특정 파일 삭제 |

**Request/Response 예시**:
```python
# POST /upload/image
# Request: multipart/form-data (file)

# Response
{
    "status": "success",
    "filename": "test_image.jpg",
    "file_path": "/home/dmillion/llm_chal_vlm/web/uploads/test_image.jpg",
    "file_size": 1024000,
    "file_size_mb": 0.98,
    "message": "파일 업로드 완료"
}
```

### 5.2 유사도 검색 API

**Router**: `web/routers/search.py`

| 메서드 | 경로 | 기능 |
|--------|------|------|
| POST | `/search/similarity` | 유사 이미지 검색 |
| GET | `/search/index/status` | 인덱스 상태 조회 |

**Request/Response 예시**:
```python
# POST /search/similarity
# Request
{
    "query_image_path": "uploads/test.jpg",
    "top_k": 5
}

# Response
{
    "status": "success",
    "query_image": "/home/dmillion/.../test.jpg",
    "top_k_results": [
        {
            "image_path": "/data/def_split/prod1_hole_001.jpg",
            "similarity_score": 0.95,
            "product": "prod1",
            "defect": "hole",
            "sequence": "001"
        }
    ],
    "total_gallery_size": 500,
    "model_info": "ViT-B-32/openai"
}
```

### 5.3 이상 검출 API

**Router**: `web/routers/anomaly.py`

| 메서드 | 경로 | 기능 |
|--------|------|------|
| POST | `/anomaly/detect` | 이상 검출 수행 |
| POST | `/anomaly/detect-session` | 세션 기반 이상 검출 |
| GET | `/anomaly/image/{result_id}/{filename}` | 결과 이미지 조회 |

**Request/Response 예시**:
```python
# POST /anomaly/detect
# Request
{
    "test_image_path": "uploads/test.jpg",
    "product_name": "prod1",
    "top1_defect_image": "/data/def_split/prod1_hole_001.jpg",
    "defect_name": "hole",
    "search_id": 123,
    "similarity_score": 0.95
}

# Response
{
    "status": "success",
    "product": "prod1",
    "test_image": "/home/dmillion/.../test.jpg",
    "anomaly_score": 0.78,
    "is_anomaly": true,
    "threshold": 0.5,
    "mask_url": "/anomaly/image/test/mask.png",
    "overlay_url": "/anomaly/image/test/overlay.png",
    "comparison_url": "/anomaly/image/test/comparison.png",
    "reference_normal_path": "/data/patchCore/prod1/ok/normal_001.jpg",
    "response_id": 456
}
```

### 5.4 대응 매뉴얼 API

**Router**: `web/routers/manual.py`

| 메서드 | 경로 | 기능 |
|--------|------|------|
| POST | `/manual/generate` | 매뉴얼 생성 |
| POST | `/manual/generate-session` | 세션 기반 매뉴얼 생성 |
| POST | `/manual/feedback` | 작업자 피드백 등록 |

**Request/Response 예시**:
```python
# POST /manual/generate
# Request
{
    "product": "prod1",
    "defect": "hole",
    "anomaly_score": 0.78,
    "is_anomaly": true,
    "model_type": "hyperclovax",  # hyperclovax | exaone | llava
    "image_path": "uploads/test.jpg",  # llava일 때만 필수
    "response_id": 456
}

# Response
{
    "status": "success",
    "product": "prod1",
    "defect_en": "hole",
    "defect_ko": "기공",
    "full_name_ko": "기공 불량",
    "manual_context": {
        "원인": ["주조 온도 불균일", "가스 배출 불량"],
        "조치": ["온도 조절", "금형 점검"]
    },
    "llm_analysis": "## 1️⃣ 불량 현황\n...",
    "anomaly_score": 0.78,
    "is_anomaly": true,
    "model_type": "hyperclovax",
    "processing_time": 3.45,
    "response_id": 456
}
```

**피드백 API**:
```python
# POST /manual/feedback
# Request
{
    "response_id": 456,
    "feedback_user": "홍길동",
    "feedback_rating": 4,
    "feedback_text": "유용한 정보입니다."
}

# Response
{
    "status": "success",
    "message": "피드백이 등록되었습니다",
    "response_id": 456
}
```

---

## 6. 도메인 모델

### 6.1 핵심 엔티티

**위치**: `web/database/models.py`

```python
# 검색 이력
class SearchHistory(Base):
    __tablename__ = 'search_history'
    
    search_id = Column(Integer, primary_key=True)
    session_id = Column(String(100))
    searched_at = Column(DateTime, default=datetime.now)
    product_code = Column(String(50))
    defect_code = Column(String(50))
    query_image_path = Column(String(500))
    top_k = Column(Integer, default=5)
    
    # 관계
    responses = relationship("ResponseHistory", back_populates="search")

# 응답 이력
class ResponseHistory(Base):
    __tablename__ = 'response_history'
    
    response_id = Column(Integer, primary_key=True)
    search_id = Column(Integer, ForeignKey('search_history.search_id'))
    executed_at = Column(DateTime)
    product_code = Column(String(50))
    defect_code = Column(String(50))
    similarity_score = Column(Float)
    anomaly_score = Column(Float)
    test_image_path = Column(String(500))
    model_type = Column(String(50))
    guide_content = Column(Text)
    guide_generated_at = Column(DateTime)
    processing_time = Column(Float)
    feedback_user = Column(String(100))
    feedback_rating = Column(Integer)
    feedback_text = Column(Text)
    feedback_at = Column(DateTime)
    
    # 관계
    search = relationship("SearchHistory", back_populates="responses")

# 이미지
class Image(Base):
    __tablename__ = 'images'
    
    image_id = Column(Integer, primary_key=True)
    product_id = Column(Integer, ForeignKey('products.product_id'))
    defect_type_id = Column(Integer, ForeignKey('defect_types.defect_type_id'))
    image_type = Column(String(20))  # 'normal' or 'defect'
    file_name = Column(String(255))
    file_path = Column(String(500))  # Object Storage 경로
    local_path = Column(String(500))  # 로컬 서버 경로
    uploaded_at = Column(DateTime)
    is_active = Column(Boolean, default=True)
```

### 6.2 ER 다이어그램

```
┌────────────────┐     ┌────────────────┐     ┌────────────────┐
│    products    │     │  defect_types  │     │    manuals     │
├────────────────┤     ├────────────────┤     ├────────────────┤
│ product_id(PK) │◄───►│defect_type_id  │     │ manual_id(PK)  │
│ product_code   │     │ product_id(FK) │     │ product_id(FK) │
│ product_name   │     │ defect_code    │     │ file_path      │
│ is_active      │     │ defect_name    │     │ file_name      │
└────────────────┘     └────────────────┘     └────────────────┘
        │                      │
        │                      │
        ▼                      ▼
┌────────────────────────────────────────────┐
│                   images                   │
├────────────────────────────────────────────┤
│ image_id(PK)                               │
│ product_id(FK), defect_type_id(FK)         │
│ image_type, file_name, file_path           │
└────────────────────────────────────────────┘

┌────────────────┐     ┌────────────────────────────────────────┐
│ search_history │     │           response_history             │
├────────────────┤     ├────────────────────────────────────────┤
│ search_id(PK)  │◄───►│ response_id(PK)                        │
│ session_id     │     │ search_id(FK)                          │
│ product_code   │     │ similarity_score, anomaly_score        │
│ defect_code    │     │ model_type, guide_content              │
│ query_image    │     │ feedback_user, feedback_rating         │
└────────────────┘     └────────────────────────────────────────┘
```

---

## 7. 외부 시스템 연동

### 7.1 LLM Server 연동

**위치**: `web/routers/manual.py`

```python
# LLM 서버 URL
_llm_server_url = "http://localhost:5001"

# HyperCLOVAX 호출
response = await client.post(
    f"{_llm_server_url}/analyze",
    json=payload
)

# EXAONE 호출
response = await client.post(
    f"{_llm_server_url}/analyze_exaone",
    json=payload
)

# LLaVA (VLM) 호출
response = await client.post(
    f"{_llm_server_url}/analyze_vlm",
    json=payload
)
```

### 7.2 Object Storage 연동

**위치**: `web/utils/object_storage.py`

```python
class ObjectStorageManager:
    def __init__(self):
        self.s3 = boto3.client(
            's3',
            endpoint_url='https://kr.object.ncloudstorage.com',
            aws_access_key_id=os.environ.get('NCP_ACCESS_KEY'),
            aws_secret_access_key=os.environ.get('NCP_SECRET_KEY'),
            region_name='kr-standard'
        )
        self.bucket = os.environ.get('NCP_BUCKET', 'dm-obs')
    
    def upload_file(self, local_path, s3_key):
        """파일 업로드"""
        pass
    
    def download_file(self, s3_key, local_path):
        """파일 다운로드"""
        pass
    
    def get_url(self, s3_key):
        """공개 URL 생성"""
        return f"https://kr.object.ncloudstorage.com/{self.bucket}/{s3_key}"
```

### 7.3 Database 연결

**위치**: `web/database/connection.py`

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

---

## 8. 화면 구성

### 8.1 화면 목록

| 화면명 | 파일명 | 경로 | 설명 |
|--------|--------|------|------|
| 로그인 | login.html | /login | 인증 페이지 |
| 이미지 업로드 | upload.html | /page/upload | 검사 이미지 업로드 |
| 유사도 검색 | search.html | /page/search | TOP-K 결과 표시 |
| 이상 검출 | anomaly.html | /page/anomaly | 이상 영역 시각화 |
| 대응 매뉴얼 | manual.html | /page/manual | AI 생성 매뉴얼 |

### 8.2 화면 흐름

```
┌─────────────────────────────────────────────────────────────────┐
│                        작업자 페이지 흐름                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐   │
│  │  업로드   │───►│ 유사도   │───►│  이상    │───►│  대응     │   │
│  │  탭      │    │ 검색 탭   │    │ 검출 탭  │    │ 매뉴얼 탭 │   │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘   │
│       │              │               │               │          │
│       ▼              ▼               ▼               ▼          │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐   │
│  │ 이미지   │     │ TOP-K   │    │ 마스크    │    │ LLM 응답  │   │
│  │ 미리보기 │     │ 결과표시 │    │ 오버레이  │    │ 4개 섹션  │   │
│  │ 인덱스   │     │ TOP-1   │    │ 비교이미지│    │ 피드백    │   │
│  │ 상태표시 │     │ 선택가능 │    │ 점수표시  │    │ 입력      │   │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 8.3 탭 구조 (공통)

```html
<!-- 탭 네비게이션 -->
<div class="tab-container">
    <button class="tab-btn" data-tab="upload">1. 이미지 업로드</button>
    <button class="tab-btn" data-tab="search">2. 유사도 검색</button>
    <button class="tab-btn" data-tab="anomaly">3. 이상 검출</button>
    <button class="tab-btn" data-tab="manual">4. 대응 매뉴얼</button>
</div>

<!-- 탭 콘텐츠 -->
<div class="tab-content" id="upload-tab">...</div>
<div class="tab-content" id="search-tab">...</div>
<div class="tab-content" id="anomaly-tab">...</div>
<div class="tab-content" id="manual-tab">...</div>
```

---

## 9. 처리 흐름

### 9.1 이미지 업로드 흐름

```
1. 사용자 → 업로드 페이지: 이미지 파일 선택
2. upload.js: FormData 생성
3. POST /upload/image: 파일 전송
4. upload.py: 
   - 파일 확장자 검증 (jpg, jpeg, png, webp, bmp)
   - 로컬 디렉토리에 저장
5. 사용자에게 성공 응답
   - 파일명, 경로, 크기 정보 반환
```

### 9.2 유사도 검색 흐름

```
1. 사용자 → 유사도 탭 클릭
2. search.js: 업로드된 이미지 경로 전달
3. POST /search/similarity: 검색 요청
4. search.py:
   - 불량 이미지 인덱스로 자동 전환 (switch_to_defect_index)
   - CLIP 임베딩 생성
   - FAISS TOP-K 검색
   - 결과에서 제품/불량 정보 추출
5. 사용자에게 TOP-K 결과 반환
   - 썸네일, 유사도 점수, 제품/불량명
6. 사용자: TOP-1 클릭으로 변경 가능
```

### 9.3 이상 검출 흐름

```
1. 사용자 → 이상 검출 탭 클릭
2. anomaly.js: 제품명, 불량명, 이미지 경로 전달
3. POST /anomaly/detect: 검출 요청
4. anomaly.py:
   - 정상 이미지 인덱스로 전환 (switch_to_normal_index)
   - PatchCore detect_with_normal_reference 호출
   - 마스크, 오버레이, 비교 이미지 생성
   - DB에 response_history 저장
5. 사용자에게 검출 결과 반환
   - 이상 점수, 판정 결과
   - 마스크/오버레이/비교 이미지 URL
   - response_id
```

### 9.4 대응 매뉴얼 생성 흐름

```
1. 사용자 → 대응 매뉴얼 탭에서 모델 선택
2. manual.js: 제품, 불량, 점수, 모델타입 전달
3. POST /manual/generate: 생성 요청
4. manual.py:
   a. DefectMapper: 불량 정보 조회 (영문명, 한글명)
   b. RAG: search_defect_manual 호출
      - 매뉴얼에서 원인/조치 컨텍스트 검색
   c. LLM Server 호출:
      - HyperCLOVAX: POST /analyze
      - EXAONE: POST /analyze_exaone
      - LLaVA: POST /analyze_vlm (이미지 포함)
   d. DB 업데이트: guide_content, model_type 저장
5. 사용자에게 생성 결과 반환
   - 4개 섹션 (불량현황, 원인분석, 대응방안, 예방조치)
   - 처리 시간

6. 사용자 → 피드백 입력
7. POST /manual/feedback: 피드백 저장
   - 작업자명, 점수(1-5), 코멘트
```

### 9.5 인덱스 자동 전환 로직

```python
# 유사도 검색 시: 불량 이미지 인덱스
async def switch_to_defect_index():
    """
    인덱스 위치: index_cache/defect/
    대상 이미지: data/def_split/ (불량 이미지)
    """
    defect_index_path = INDEX_DIR / "defect"
    if (defect_index_path / "index_data.pt").exists():
        matcher.load_index(str(defect_index_path))
    else:
        matcher.build_index(str(defect_dir))
        matcher.save_index(str(defect_index_path))

# 이상 검출 시: 정상 이미지 인덱스
async def switch_to_normal_index():
    """
    인덱스 위치: index_cache/normal/
    대상 이미지: data/patchCore/{product}/ok/ (정상 이미지)
    """
    normal_index_path = INDEX_DIR / "normal"
    if (normal_index_path / "index_data.pt").exists():
        matcher.load_index(str(normal_index_path))
    else:
        matcher.build_index(str(normal_base_dir))
        matcher.save_index(str(normal_index_path))
```

---

## 10. 배포 구성

### 10.1 인프라 구성

```
┌─────────────────────────────────────────────────────────────────┐
│                    Naver Cloud Platform                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                     VPC (dm-vpc)                         │   │
│  │                   10.200.0.0/16                          │   │
│  │  ┌─────────────────────────────────────────────────────┐ │   │
│  │  │            Load Balancer Subnet                     │ │   │
│  │  │            (dm-lb-sub: 10.200.1.0/24)               │ │   │
│  │  │  ┌────────────┐                                     │ │   │
│  │  │  │    ALB     │  Port 80 → 8000 (작업자)            │ │   │
│  │  │  │            │  Port 80 → 8080 (관리자)            │ │   │
│  │  │  └─────┬──────┘                                     │ │   │
│  │  └────────┼────────────────────────────────────────────┘ │   │
│  │           │                                              │   │
│  │  ┌────────┼────────────────────────────────────────────┐ │   │
│  │  │        ▼        Private Subnet                      │ │   │
│  │  │               (dm-pri-sub: 10.200.3.0/24)           │ │   │
│  │  │  ┌────────────────────────────────────┐             │ │   │
│  │  │  │          GPU Server                │             │ │   │
│  │  │  │  ┌──────────────┐ ┌──────────────┐ │             │ │   │
│  │  │  │  │ API Server   │ │ LLM Server   │ │             │ │   │
│  │  │  │  │ Port: 8000   │ │ Port: 5001   │ │             │ │   │
│  │  │  │  │ (작업자 UI)  │  │ (모델 추론)  │ │             │ │   │
│  │  │  │  └──────────────┘ └──────────────┘ │             │ │   │
│  │  │  └────────────────────────────────────┘             │ │   │
│  │  └─────────────────────────────────────────────────────┘ │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐                       │
│  │  Object Storage │  │    MariaDB      │                       │
│  │    (dm-obs)     │  │                 │                       │
│  └─────────────────┘  └─────────────────┘                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 10.2 서버 시작 스크립트

**API Server (start_api_server.sh)**:
```bash
#!/bin/bash
cd /home/dmillion/llm_chal_vlm/web
source ../venv_patch/bin/activate

# 기존 프로세스 종료
pkill -f 'uvicorn.*api_server' || true

# 서버 시작
nohup uvicorn api_server:app --host 0.0.0.0 --port 8000 \
    > api_server.log 2>&1 &

echo $! > api_server.pid
echo "API Server started on port 8000"
```

**LLM Server (start_llm_server.sh)**:
```bash
#!/bin/bash
cd /home/dmillion/llm_chal_vlm/llm_server
source ../venv310/bin/activate

# 기존 프로세스 종료
pkill -f 'python.*llm_server' || true

# 서버 시작
nohup python llm_server.py > llm_server.log 2>&1 &

echo $! > llm_server.pid
echo "LLM Server started on port 5001"
```

### 10.3 환경 변수

```bash
# Database
export DB_HOST=localhost
export DB_PORT=3306
export DB_NAME=llm_chal_db
export DB_USER=dmillion
export DB_PASSWORD=****

# NCP Object Storage
export NCP_ACCESS_KEY=****
export NCP_SECRET_KEY=****
export NCP_BUCKET=dm-obs

# LLM Server
export LLM_SERVER_URL=http://localhost:5001
```

### 10.4 접속 정보

| 서비스 | URL | 용도 |
|--------|-----|------|
| 작업자 페이지 | http://dm-alb-xxx.kr.lb.naverncp.com:80 | 외부 접속 (ALB) |
| 내부 접속 | http://10.200.3.x:8000 | 서버 내부 |
| LLM 서버 | http://localhost:5001 | 내부 통신 |
| 헬스체크 | /health | ALB 상태 확인 |

---

## 부록: 라우터 코드 요약

### A. upload.py 주요 함수

| 함수 | 경로 | 기능 |
|------|------|------|
| `init_upload_router()` | - | 업로드 디렉토리 초기화 |
| `upload_image()` | POST /upload/image | 이미지 업로드 |
| `list_uploaded_files()` | GET /upload/list | 파일 목록 조회 |
| `clean_uploads()` | DELETE /upload/clean | 디렉토리 정리 |
| `delete_file()` | DELETE /upload/file/{filename} | 파일 삭제 |

### B. search.py 주요 함수

| 함수 | 경로 | 기능 |
|------|------|------|
| `init_search_router()` | - | 매처/인덱스 초기화 |
| `switch_to_defect_index()` | - | 불량 인덱스 전환 |
| `search_similar_images()` | POST /search/similarity | 유사 이미지 검색 |
| `get_search_index_status()` | GET /search/index/status | 인덱스 상태 |

### C. anomaly.py 주요 함수

| 함수 | 경로 | 기능 |
|------|------|------|
| `init_anomaly_router()` | - | 검출기/매처 초기화 |
| `switch_to_normal_index()` | - | 정상 인덱스 전환 |
| `detect_anomaly()` | POST /anomaly/detect | 이상 검출 |
| `detect_anomaly_session()` | POST /anomaly/detect-session | 세션 기반 검출 |
| `serve_anomaly_image()` | GET /anomaly/image/{id}/{file} | 결과 이미지 제공 |

### D. manual.py 주요 함수

| 함수 | 경로 | 기능 |
|------|------|------|
| `init_manual_router()` | - | 매퍼/RAG 초기화 |
| `generate_manual()` | POST /manual/generate | 매뉴얼 생성 |
| `generate_manual_session()` | POST /manual/generate-session | 세션 기반 생성 |
| `submit_feedback()` | POST /manual/feedback | 피드백 등록 |

---

**검토자**: dhkim  
**최종 수정일**: 2025-11-24

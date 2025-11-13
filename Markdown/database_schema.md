# MySQL/MariaDB 데이터베이스 스키마 설계 (수정 ver 2.0)

## 1. 데이터베이스 개요

### 1.1 변경사항 요약
- **사용자 관리**: users 테이블 삭제 → config 파일로 관리 (worker/admin 2개 계정)
- **이미지 전처리**: image_preprocessing 테이블 추가 (제품별 전처리 설정)
- **PK 관리**: 모든 테이블 AUTO_INCREMENT PK 적용
- **제약조건**: FK, Index, Check 제약 미적용 (필요시 수동 추가)
- **DBMS**: MariaDB 추천 (라이센스, 호환성 우수)

### 1.2 테이블 구성 (9개 - users 삭제)
```
1. products                 - 제품
2. manuals                  - 매뉴얼
3. defect_types             - 불량 유형
4. images                   - 이미지 메타데이터
5. image_preprocessing      - 이미지 전처리 설정 (신규)
6. search_history           - 유사도 검색 이력
7. response_history         - 대응 매뉴얼 생성 이력
8. model_params             - 모델 파라미터 설정
9. deployment_logs          - 배포 실행 이력
10. system_config           - 전역 설정 (Key-Value)
```


---

## 2. ERD (수정 버전)

```
┌──────────────┐
│  products    │
└──────────────┘
       │
       ├─────────────────┬──────────────┬──────────────┬──────────────┐
       │                 │              │              │              │
       ▼                 ▼              ▼              ▼              ▼
┌──────────────┐  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│   manuals    │  │defect_types  │ │    images    │ │model_params  │ │image_        │
└──────────────┘  └──────────────┘ └──────────────┘ └──────────────┘ │preprocessing │
                         │                                            └──────────────┘
                         │
                         ▼
                  ┌──────────────┐
                  │search_       │
                  │history       │
                  └──────────────┘
                         │
                         ▼
                  ┌──────────────┐
                  │response_     │
                  │history       │
                  └──────────────┘

┌──────────────┐  ┌──────────────┐
│deployment_   │  │system_config │  (독립 테이블)
│logs          │  └──────────────┘
└──────────────┘
```

---

## 3. 테이블 스키마 상세

### 3.1 products (제품)

```sql
CREATE TABLE products (
    product_id INT AUTO_INCREMENT PRIMARY KEY COMMENT '제품 ID (PK)',
    product_code VARCHAR(50) NOT NULL COMMENT '제품 코드 (예: prod1, prod2)',
    product_name VARCHAR(100) NOT NULL COMMENT '제품명',
    description TEXT COMMENT '제품 설명',
    is_active TINYINT(1) DEFAULT 1 COMMENT '활성 여부 (1: 활성, 0: 비활성)',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '생성일시',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '수정일시'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='제품 마스터';

-- 초기 데이터
INSERT INTO products (product_code, product_name, description) VALUES
('prod1', '주조 제품 A형', '주조 공정 제품'),
('grid', '그리드 제품', '그리드 패턴 제품'),
('carpet', '카펫 제품', '카펫 재질 제품'),
('leather', '가죽 제품', '가죽 재질 제품');
```

---

### 3.2 manuals (매뉴얼)

```sql
CREATE TABLE manuals (
    manual_id INT AUTO_INCREMENT PRIMARY KEY COMMENT '매뉴얼 ID (PK)',
    product_id INT NOT NULL COMMENT '제품 ID (FK)',
    file_name VARCHAR(255) NOT NULL COMMENT '파일명',
    file_path VARCHAR(500) NOT NULL COMMENT '파일 경로 (OBS 또는 로컬)',
    file_size BIGINT COMMENT '파일 크기 (bytes)',
    vector_indexed TINYINT(1) DEFAULT 0 COMMENT 'RAG 벡터 인덱싱 완료 여부 (1: 완료, 0: 미완료)',
    indexed_at DATETIME COMMENT '인덱싱 완료 일시',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '등록일시'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='제품별 대응 매뉴얼';

-- 초기 데이터
INSERT INTO manuals (product_id, file_name, file_path, vector_indexed, indexed_at) VALUES
(1, 'prod1_menual.pdf', '/manual_store/prod1_menual.pdf', 1, NOW()),
(2, 'grid_manual.pdf', '/manual_store/grid_manual.pdf', 1, NOW()),
(3, 'carpet_manual.pdf', '/manual_store/carpet_manual.pdf', 1, NOW()),
(4, 'leather_manual.pdf', '/manual_store/leather_manual.pdf', 1, NOW());
```

---

### 3.3 defect_types (불량 유형)

```sql
CREATE TABLE defect_types (
    defect_type_id INT AUTO_INCREMENT PRIMARY KEY COMMENT '불량 유형 ID (PK)',
    product_id INT NOT NULL COMMENT '제품 ID (FK)',
    defect_code VARCHAR(50) NOT NULL COMMENT '불량 코드 (영문: normal, hole, burr, scratch 등)',
    defect_name_ko VARCHAR(100) NOT NULL COMMENT '불량 명칭 (한글: 정상, 기공, 버, 긁힘 등)',
    defect_name_en VARCHAR(100) COMMENT '불량 명칭 (영문)',
    full_name_ko VARCHAR(200) COMMENT '불량 전체 명칭 (한글)',
    is_active TINYINT(1) DEFAULT 1 COMMENT '활성 여부',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '생성일시'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='제품별 불량 유형';

-- 초기 데이터 (prod1 기준)
INSERT INTO defect_types (product_id, defect_code, defect_name_ko, defect_name_en, full_name_ko) VALUES
(1, 'normal', '정상', 'normal', '정상 제품'),
(1, 'hole', '기공', 'hole', '주조 기공'),
(1, 'burr', '버', 'burr', '버 (날카로운 돌기)'),
(1, 'scratch', '긁힘', 'scratch', '표면 긁힘'),
-- leather 제품 예시
(4, 'normal', '정상', 'normal', '정상 제품'),
(4, 'hole', '기공', 'hole', '기공'),
(4, 'burr', '버', 'burr', '버'),
(4, 'scratch', '긁힘', 'scratch', '긁힘'),
(4, 'fold', '주름', 'fold', '주름'),
(4, 'stain', '얼룩', 'stain', '얼룩'),
(4, 'color', '색상 불량', 'color', '색상 불량');
```

---

### 3.4 images (이미지 메타데이터)

```sql
CREATE TABLE images (
    image_id INT AUTO_INCREMENT PRIMARY KEY COMMENT '이미지 ID (PK)',
    product_id INT NOT NULL COMMENT '제품 ID (FK)',
    image_type VARCHAR(20) NOT NULL COMMENT '이미지 유형 (normal: 정상, defect: 불량, test: 검사용)',
    defect_type_id INT COMMENT '불량 유형 ID (FK, image_type=defect인 경우)',
    file_name VARCHAR(255) NOT NULL COMMENT '파일명',
    file_path VARCHAR(500) NOT NULL COMMENT '파일 경로 (OBS 또는 로컬)',
    file_size BIGINT COMMENT '파일 크기 (bytes)',
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '업로드 일시'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='이미지 메타데이터';

-- 초기 데이터 예시
INSERT INTO images (product_id, image_type, defect_type_id, file_name, file_path) VALUES
(1, 'normal', NULL, 'prod1_ok_0_129.jpeg', '/data/patchCore/prod1/prod1_ok_0_129.jpeg'),
(1, 'defect', 2, 'prod1_hole_001.jpeg', '/data/def_split/prod1_hole_001.jpeg'),
(1, 'defect', 3, 'prod1_burr_001.jpeg', '/data/def_split/prod1_burr_001.jpeg');
```

---

### 3.5 image_preprocessing (이미지 전처리 설정) - **신규 추가**

```sql
CREATE TABLE image_preprocessing (
    preprocessing_id INT AUTO_INCREMENT PRIMARY KEY COMMENT '전처리 설정 ID (PK)',
    product_id INT NOT NULL COMMENT '제품 ID (FK)',
    grayscale CHAR(1) DEFAULT 'N' COMMENT '그레이스케일 변환 (Y/N)',
    histogram CHAR(1) DEFAULT 'N' COMMENT '히스토그램 평활화 (Y/N)',
    contrast CHAR(1) DEFAULT 'N' COMMENT '명암 대비 조정 (Y/N)',
    smoothing CHAR(1) DEFAULT 'N' COMMENT '스무딩 (블러) 처리 (Y/N)',
    normalize CHAR(1) DEFAULT 'N' COMMENT '정규화 (Y/N)',
    is_active TINYINT(1) DEFAULT 1 COMMENT '활성 여부 (1: 활성, 0: 비활성)',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '생성일시',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '수정일시'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='제품별 이미지 전처리 설정';

-- 초기 데이터 (모든 제품 기본값: 전처리 미적용)
INSERT INTO image_preprocessing (product_id, grayscale, histogram, contrast, smoothing, normalize) VALUES
(1, 'N', 'N', 'N', 'N', 'N'),
(2, 'N', 'N', 'N', 'N', 'N'),
(3, 'N', 'N', 'N', 'N', 'N'),
(4, 'N', 'N', 'N', 'N', 'N');
```

**설명**:
- 제품별로 전처리 항목을 Y/N으로 관리
- 작업자 페이지에서 이미지 업로드 시 해당 제품의 전처리 설정을 조회하여 자동 적용
- 관리자 페이지에서 제품별 전처리 설정 변경 가능

---

### 3.6 search_history (유사도 검색 이력)

```sql
CREATE TABLE search_history (
    search_id INT AUTO_INCREMENT PRIMARY KEY COMMENT '검색 ID (PK)',
    searched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '검색 일시',
    uploaded_image_path VARCHAR(500) NOT NULL COMMENT '업로드된 이미지 경로',
    product_code VARCHAR(50) COMMENT '제품 코드',
    defect_code VARCHAR(50) COMMENT '불량 코드',
    top_k_results JSON NOT NULL COMMENT 'TOP-K 검색 결과 JSON',
    processing_time FLOAT COMMENT '처리 시간 (초)'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='유사도 검색 이력';

-- JSON 구조 예시
/*
top_k_results:
[
  {
    "rank": 1,
    "image_path": "/data/def_split/prod1_burr_021.jpeg",
    "image_name": "prod1_burr_021.jpeg",
    "similarity": 0.9884,
    "product": "prod1",
    "defect": "burr",
    "sequence": "021"
  },
  {
    "rank": 2,
    "image_path": "/data/def_split/prod1_burr_013.jpeg",
    "image_name": "prod1_burr_013.jpeg",
    "similarity": 0.9521,
    "product": "prod1",
    "defect": "burr",
    "sequence": "013"
  }
]
*/
```

---

### 3.7 response_history (대응 매뉴얼 생성 이력)

```sql
CREATE TABLE response_history (
    response_id INT AUTO_INCREMENT PRIMARY KEY COMMENT '응답 ID (PK)',
    executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '실행 일시',
    search_id INT COMMENT '연관 검색 ID (FK)',
    
    -- 검출 결과
    product_code VARCHAR(50) NOT NULL COMMENT '제품 코드',
    defect_code VARCHAR(50) NOT NULL COMMENT '불량 코드',
    similarity_score FLOAT COMMENT '유사도 점수',
    anomaly_score FLOAT COMMENT '이상 점수',
    confidence_score FLOAT COMMENT '신뢰도 점수',
    
    -- 이미지 경로
    test_image_path VARCHAR(500) COMMENT '검사 이미지 경로',
    reference_image_path VARCHAR(500) COMMENT '기준 이미지 경로',
    heatmap_path VARCHAR(500) COMMENT '히트맵 이미지 경로',
    overlay_path VARCHAR(500) COMMENT '오버레이 이미지 경로',
    
    -- LLM 생성 가이드
    model_type VARCHAR(50) COMMENT 'LLM 모델 타입 (hyperclovax, exaone, llava)',
    guide_content TEXT COMMENT 'LLM 생성 대응 매뉴얼',
    guide_generated_at DATETIME COMMENT '가이드 생성 일시',
    
    -- 피드백
    feedback_rating INT COMMENT '사용자 평가 (1~5)',
    feedback_text TEXT COMMENT '피드백 내용',
    feedback_at DATETIME COMMENT '피드백 작성 일시',
    
    processing_time FLOAT COMMENT '처리 시간 (초)'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='대응 매뉴얼 생성 및 피드백 이력';
```

---

### 3.8 model_params (모델 파라미터 설정)

```sql
CREATE TABLE model_params (
    param_id INT AUTO_INCREMENT PRIMARY KEY COMMENT '파라미터 ID (PK)',
    product_id INT NOT NULL COMMENT '제품 ID (FK)',
    model_type VARCHAR(50) NOT NULL COMMENT '모델 타입 (clip, patchcore, llm, preprocessing)',
    params JSON NOT NULL COMMENT '모델 파라미터 JSON',
    is_active TINYINT(1) DEFAULT 1 COMMENT '활성 여부',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '생성일시',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '수정일시'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='제품별 모델 파라미터';

-- 초기 데이터
INSERT INTO model_params (product_id, model_type, params, is_active) VALUES
(1, 'clip', '{"model_id": "ViT-B-32/openai", "top_k": 5, "batch_size": 32}', 1),
(1, 'patchcore', '{"image_threshold": 0.85, "pixel_threshold": 0.90}', 1),
(1, 'llm', '{"default_model": "hyperclovax", "temperature": 0.3, "max_tokens": 768}', 1);

-- JSON 구조 예시
/*
CLIP:
{
  "model_id": "ViT-B-32/openai",
  "top_k": 5,
  "batch_size": 32,
  "use_fp16": false
}

PatchCore:
{
  "image_threshold": 0.85,
  "pixel_threshold": 0.90
}

LLM:
{
  "default_model": "hyperclovax",
  "temperature": 0.3,
  "max_tokens": 768
}
*/
```

---

### 3.9 deployment_logs (배포 실행 이력)

```sql
CREATE TABLE deployment_logs (
    deploy_id INT AUTO_INCREMENT PRIMARY KEY COMMENT '배포 ID (PK)',
    deploy_type VARCHAR(50) NOT NULL COMMENT '배포 타입 (clip_rebuild, patchcore_create, rag_index, full_deploy)',
    product_id INT COMMENT '제품 ID (FK)',
    status VARCHAR(20) DEFAULT 'pending' COMMENT '상태 (pending, running, completed, failed)',
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '시작 일시',
    completed_at DATETIME COMMENT '완료 일시',
    result_message TEXT COMMENT '결과 메시지',
    result_data JSON COMMENT '결과 상세 JSON',
    deployed_by VARCHAR(50) COMMENT '배포 실행자 (admin/worker)'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='배포 실행 이력';

-- 초기 데이터
INSERT INTO deployment_logs (deploy_type, product_id, status, completed_at, result_message, result_data, deployed_by) VALUES
('clip_rebuild', 1, 'completed', NOW(), 'CLIP 인덱스 재구축 완료', 
'{"num_images": 2083, "embedding_dim": 512, "processing_time": 60.5}', 'admin'),
('patchcore_create', 1, 'completed', NOW(), 'PatchCore 메모리뱅크 생성 완료',
'{"num_patches": 614, "feature_dim": 3584, "processing_time": 120.3}', 'admin');
```

---

### 3.10 system_config (전역 설정)

```sql
CREATE TABLE system_config (
    config_key VARCHAR(100) PRIMARY KEY COMMENT '설정 키',
    config_value TEXT NOT NULL COMMENT '설정 값 (JSON 또는 문자열)',
    description TEXT COMMENT '설명',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '수정일시',
    updated_by VARCHAR(50) COMMENT '수정자 (admin/worker)'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='시스템 전역 설정';

-- 초기 데이터
INSERT INTO system_config (config_key, config_value, description) VALUES
('default_top_k', '5', '기본 TOP-K 개수'),
('default_clip_model', 'ViT-B-32/openai', '기본 CLIP 모델'),
('default_image_threshold', '0.85', '기본 이미지 임계값'),
('default_pixel_threshold', '0.90', '기본 픽셀 임계값'),
('llm_temperature', '0.3', 'LLM 샘플링 온도'),
('max_upload_size_mb', '50', '최대 업로드 크기 (MB)'),
('session_timeout_minutes', '60', '세션 타임아웃 (분)');
```

---

## 4. 사용자 인증 설정 (config.yaml)

```yaml
# config/auth.yaml
users:
  admin:
    username: admin
    password_hash: $2b$12$xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    user_type: admin
    full_name: 시스템 관리자
  
  worker:
    username: worker
    password_hash: $2b$12$xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    user_type: worker
    full_name: 작업자

# Python에서 사용 예시
# from passlib.hash import bcrypt
# password_hash = bcrypt.hash("your_password")
```

---

## 5. 데이터베이스 생성 및 초기화

### 5.1 MariaDB 설치 및 설정 (Rocky Linux 8.10)

```bash
# MariaDB 설치
sudo dnf install mariadb-server mariadb -y

# 서비스 시작 및 자동 시작 설정
sudo systemctl start mariadb
sudo systemctl enable mariadb

# 초기 보안 설정
sudo mysql_secure_installation
# - root 비밀번호 설정
# - 익명 사용자 제거
# - 원격 root 로그인 비활성화
# - test 데이터베이스 제거

# 버전 확인
mysql --version
# 출력 예: mysql  Ver 15.1 Distrib 10.5.22-MariaDB
```

### 5.2 데이터베이스 생성 스크립트

```sql
-- database_init.sql

-- 데이터베이스 생성
CREATE DATABASE IF NOT EXISTS defect_detection_db
DEFAULT CHARACTER SET utf8mb4
DEFAULT COLLATE utf8mb4_unicode_ci;

USE defect_detection_db;

-- 사용자 생성 및 권한 부여
CREATE USER IF NOT EXISTS 'defect_user'@'localhost' IDENTIFIED BY 'your_secure_password';
GRANT ALL PRIVILEGES ON defect_detection_db.* TO 'defect_user'@'localhost';
FLUSH PRIVILEGES;

-- 테이블 생성 (위의 스키마 순서대로)
SOURCE 01_products.sql;
SOURCE 02_manuals.sql;
SOURCE 03_defect_types.sql;
SOURCE 04_images.sql;
SOURCE 05_image_preprocessing.sql;
SOURCE 06_search_history.sql;
SOURCE 07_response_history.sql;
SOURCE 08_model_params.sql;
SOURCE 09_deployment_logs.sql;
SOURCE 10_system_config.sql;

-- 초기 데이터 삽입
SOURCE init_data.sql;
```

### 5.3 실행

```bash
# MariaDB 접속
mysql -u root -p

# 스크립트 실행
source /path/to/database_init.sql

# 또는 외부에서 직접 실행
mysql -u root -p < database_init.sql
```

---

## 6. 현재 코드와의 호환성 검토

### 6.1 현재 코드 구조 분석

**현재 파일 기반 관리**:
- `defect_mapping.json`: 제품/불량 매핑
- 이미지 파일: 파일명 규칙 (`{product}_{defect}_{seq}.jpg`)
- 인덱스: FAISS 파일 저장

**DB 연동 필요 영역**:
1. **검색 이력 저장**: `/search/similarity` 응답 후
2. **이상 검출 이력**: `/anomaly/detect` 응답 후
3. **매뉴얼 생성 이력**: `/manual/generate` 응답 후
4. **불량 이미지 등록**: `/register_defect` 시 images 테이블 갱신
5. **제품/불량 관리**: 관리자 페이지에서 CRUD

### 6.2 DB 연동 코드 예시 (SQLAlchemy)

```python
# database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# MariaDB 연결
DATABASE_URL = "mysql+pymysql://defect_user:your_secure_password@localhost/defect_detection_db?charset=utf8mb4"

engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

```python
# models.py
from sqlalchemy import Column, Integer, String, Float, Text, TIMESTAMP, JSON
from database import Base

class SearchHistory(Base):
    __tablename__ = "search_history"
    
    search_id = Column(Integer, primary_key=True, autoincrement=True)
    searched_at = Column(TIMESTAMP)
    uploaded_image_path = Column(String(500))
    product_code = Column(String(50))
    defect_code = Column(String(50))
    top_k_results = Column(JSON)
    processing_time = Column(Float)

class ResponseHistory(Base):
    __tablename__ = "response_history"
    
    response_id = Column(Integer, primary_key=True, autoincrement=True)
    executed_at = Column(TIMESTAMP)
    search_id = Column(Integer)
    product_code = Column(String(50))
    defect_code = Column(String(50))
    similarity_score = Column(Float)
    anomaly_score = Column(Float)
    guide_content = Column(Text)
    feedback_rating = Column(Integer)
    feedback_text = Column(Text)
```

```python
# 라우터에서 사용 예시
from fastapi import Depends
from sqlalchemy.orm import Session
from database import get_db
from models import SearchHistory

@router.post("/search/similarity")
async def search_similar_images(request: SearchRequest, db: Session = Depends(get_db)):
    # ... 기존 검색 로직 ...
    
    # DB에 이력 저장
    search_record = SearchHistory(
        uploaded_image_path=str(query_path),
        product_code=results_with_info[0]['product'],
        defect_code=results_with_info[0]['defect'],
        top_k_results=results_with_info,
        processing_time=processing_time
    )
    db.add(search_record)
    db.commit()
    db.refresh(search_record)
    
    return JSONResponse(content={
        "status": "success",
        "search_id": search_record.search_id,  # 추가
        "top_k_results": results_with_info,
        # ...
    })
```

### 6.3 필요한 패키지

```bash
pip install sqlalchemy pymysql cryptography
```

---

## 7. 관리자 화면 Scope 대비 스키마 적합성

### 7.1 관리자 화면 요구사항 (개발 가이드 기준)

**1. 제품/매뉴얼 관리**
- ✅ `products` 테이블: CRUD 지원
- ✅ `defect_types` 테이블: 제품별 불량 유형 관리
- ✅ `manuals` 테이블: PDF 업로드 및 인덱싱 상태 관리

**2. 이미지 데이터 관리**
- ✅ `images` 테이블: 정상/불량 이미지 메타데이터
- ✅ `image_preprocessing` 테이블: 제품별 전처리 설정 (신규)
- ✅ ZIP 업로드 지원 가능 (파일 파싱 후 일괄 INSERT)

**3. 서버 배포 관리**
- ✅ `deployment_logs` 테이블: 배포 이력 추적
- ✅ `model_params` 테이블: CLIP/PatchCore 파라미터 조정

**4. 통합 대시보드**
- ✅ `search_history` + `response_history`: 검사 통계
- ✅ `feedback_rating`: 사용자 만족도
- ✅ View 생성 가능 (집계 쿼리 최적화)

### 7.2 스키마 개선사항 추가 제안

**현재 스키마로 충분하나, 향후 고려사항**:
1. **알림 기능**: `notifications` 테이블 (미조치 알림)
2. **사용자 활동 로그**: `activity_logs` 테이블 (admin/worker 구분)
3. **배치 작업 스케줄**: `scheduled_jobs` 테이블

---

## 8. 마이그레이션 계획

### 8.1 기존 데이터 → DB 이전

```python
# migrate_to_mysql.py
import pymysql
import json
from pathlib import Path

def migrate_defect_mapping():
    """defect_mapping.json → products + defect_types"""
    conn = pymysql.connect(
        host='localhost',
        user='defect_user',
        password='your_secure_password',
        database='defect_detection_db',
        charset='utf8mb4'
    )
    
    cursor = conn.cursor()
    
    # defect_mapping.json 읽기
    mapping_file = Path('/home/dmillion/llm_chal_vlm/web/defect_mapping.json')
    with open(mapping_file) as f:
        mapping_data = json.load(f)
    
    products = mapping_data.get('products', {})
    
    for product_code, product_info in products.items():
        # products 삽입
        cursor.execute("""
            INSERT INTO products (product_code, product_name)
            VALUES (%s, %s)
            ON DUPLICATE KEY UPDATE product_name=VALUES(product_name)
        """, (product_code, product_info['name_ko']))
        
        product_id = cursor.lastrowid or cursor.execute(
            "SELECT product_id FROM products WHERE product_code=%s", (product_code,)
        )
        
        # defect_types 삽입
        defects = product_info.get('defects', {})
        for defect_code, defect_info in defects.items():
            cursor.execute("""
                INSERT INTO defect_types 
                (product_id, defect_code, defect_name_ko, defect_name_en, full_name_ko)
                VALUES (%s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE 
                    defect_name_ko=VALUES(defect_name_ko),
                    full_name_ko=VALUES(full_name_ko)
            """, (
                product_id,
                defect_info['en'],
                defect_info['ko'],
                defect_info['en'],
                defect_info.get('full_name_ko', '')
            ))
    
    conn.commit()
    cursor.close()
    conn.close()
    print("✅ defect_mapping.json 마이그레이션 완료")

if __name__ == "__main__":
    migrate_defect_mapping()
```

---

## 9. 정리 및 권장사항

### 9.1 최종 구성

| 항목 | 설정 |
|------|------|
| DBMS | **MariaDB 10.5+** (라이센스, 호환성 우수) |
| 테이블 수 | **10개** (users 삭제, image_preprocessing 추가) |
| 인증 방식 | **config 파일** (admin/worker 2개 계정) |
| PK 관리 | **AUTO_INCREMENT** (모든 테이블) |
| 제약조건 | **미적용** (필요시 수동 추가) |

### 9.2 다음 단계

1. **MariaDB 설치 및 초기화**
2. **테이블 생성 스크립트 실행**
3. **기존 데이터 마이그레이션**
4. **FastAPI 라우터 DB 연동**
5. **관리자 페이지 개발 시작**

---

**문서 버전**: 2.0  
**최종 수정**: 2025-01-15  
**DBMS**: MariaDB 10.5+  
**테이블 수**: 10개
# PostgreSQL 데이터베이스 스키마 설계

## 1. 데이터베이스 개요

### 1.1 설계 원칙
- **정규화**: 제3정규형(3NF) 준수
- **확장성**: 제품/업체 추가에 유연한 구조
- **이력 관리**: 검사 이력 및 피드백 추적
- **성능**: 적절한 인덱스 및 파티셔닝

### 1.2 주요 엔티티
1. **사용자 관리**: users, roles, permissions
2. **제품 관리**: products, defect_types
3. **이미지 관리**: images, image_metadata
4. **매뉴얼 관리**: manuals, manual_sections
5. **검사 관리**: inspections, inspection_results
6. **배포 관리**: deployments, model_configs
7. **피드백 관리**: feedbacks
8. **대시보드**: 각종 통계 뷰

---

## 2. ERD (Entity Relationship Diagram)

```
┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│    users    │────────<│  user_roles │>────────│    roles    │
└─────────────┘         └─────────────┘         └─────────────┘
       │
       │ created_by
       ├──────────┐
       │          │
       ▼          ▼
┌─────────────┐  ┌─────────────┐
│  products   │  │  manuals    │
└─────────────┘  └─────────────┘
       │                │
       │ product_id     │ manual_id
       ├────────┬───────┼─────────┐
       ▼        ▼       ▼         ▼
┌────────────┐ ┌────────────┐ ┌────────────┐
│defect_types│ │   images   │ │manual_     │
└────────────┘ └────────────┘ │sections    │
       │              │        └────────────┘
       │              │
       │              │ image_id
       │              ▼
       │        ┌────────────┐
       │        │inspections │
       │        └────────────┘
       │              │
       │              │ inspection_id
       │              ├───────────┬────────────┐
       │              ▼           ▼            ▼
       │        ┌────────────┐ ┌────────────┐ ┌────────────┐
       └───────>│inspection_ │ │  feedbacks │ │action_logs │
                │results     │ └────────────┘ └────────────┘
                └────────────┘
                      │
                      │ defect_type_id
                      ▼
                ┌────────────┐
                │defect_types│
                └────────────┘

┌─────────────┐
│deployments  │
└─────────────┘
       │
       │ product_id
       ▼
┌─────────────┐
│model_configs│
└─────────────┘
```

---

## 3. 테이블 스키마 상세

### 3.1 사용자 관리

#### users (사용자)
```sql
CREATE TABLE users (
    user_id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(100),
    phone VARCHAR(20),
    is_active BOOLEAN DEFAULT TRUE,
    last_login_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT users_username_check CHECK (LENGTH(username) >= 3),
    CONSTRAINT users_email_check CHECK (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$')
);

CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_is_active ON users(is_active);

COMMENT ON TABLE users IS '사용자 계정 정보';
COMMENT ON COLUMN users.password_hash IS 'bcrypt 해시된 비밀번호';
```

#### roles (역할)
```sql
CREATE TABLE roles (
    role_id SERIAL PRIMARY KEY,
    role_name VARCHAR(50) UNIQUE NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT roles_name_check CHECK (role_name IN ('admin', 'manager', 'worker', 'viewer'))
);

INSERT INTO roles (role_name, description) VALUES
('admin', '시스템 관리자 - 모든 권한'),
('manager', '관리자 - 제품/데이터 관리'),
('worker', '작업자 - 검사 수행'),
('viewer', '열람자 - 조회만 가능');

COMMENT ON TABLE roles IS '사용자 역할 정의';
```

#### user_roles (사용자-역할 매핑)
```sql
CREATE TABLE user_roles (
    user_id INTEGER REFERENCES users(user_id) ON DELETE CASCADE,
    role_id INTEGER REFERENCES roles(role_id) ON DELETE CASCADE,
    assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    assigned_by INTEGER REFERENCES users(user_id),
    
    PRIMARY KEY (user_id, role_id)
);

CREATE INDEX idx_user_roles_user ON user_roles(user_id);
CREATE INDEX idx_user_roles_role ON user_roles(role_id);

COMMENT ON TABLE user_roles IS '사용자별 역할 할당';
```

---

### 3.2 제품 관리

#### products (제품)
```sql
CREATE TABLE products (
    product_id SERIAL PRIMARY KEY,
    product_code VARCHAR(50) UNIQUE NOT NULL,
    product_name VARCHAR(100) NOT NULL,
    description TEXT,
    category VARCHAR(50),
    is_active BOOLEAN DEFAULT TRUE,
    created_by INTEGER REFERENCES users(user_id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT products_code_check CHECK (LENGTH(product_code) >= 2)
);

CREATE INDEX idx_products_code ON products(product_code);
CREATE INDEX idx_products_active ON products(is_active);

COMMENT ON TABLE products IS '제품 마스터';
COMMENT ON COLUMN products.product_code IS '제품 코드 (예: prod1, prod2)';
```

#### defect_types (불량 유형)
```sql
CREATE TABLE defect_types (
    defect_type_id SERIAL PRIMARY KEY,
    product_id INTEGER REFERENCES products(product_id) ON DELETE CASCADE,
    defect_code VARCHAR(50) NOT NULL,          -- 영문 매핑명 (hole, burr, scratch)
    defect_name_ko VARCHAR(100) NOT NULL,      -- 한글 명칭 (기공, 버, 긁힘)
    defect_name_en VARCHAR(100),               -- 영문 정식 명칭
    full_name_ko VARCHAR(200),                 -- 전체 한글 명칭
    description TEXT,
    severity_level INTEGER DEFAULT 3,          -- 심각도 (1~5)
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(product_id, defect_code),
    CONSTRAINT defect_severity_check CHECK (severity_level BETWEEN 1 AND 5)
);

CREATE INDEX idx_defect_types_product ON defect_types(product_id);
CREATE INDEX idx_defect_types_code ON defect_types(defect_code);
CREATE INDEX idx_defect_types_active ON defect_types(is_active);

COMMENT ON TABLE defect_types IS '제품별 불량 유형 정의';
COMMENT ON COLUMN defect_types.defect_code IS '시스템 내부 코드 (영문)';
COMMENT ON COLUMN defect_types.severity_level IS '1=경미, 5=치명적';
```

---

### 3.3 매뉴얼 관리

#### manuals (매뉴얼)
```sql
CREATE TABLE manuals (
    manual_id SERIAL PRIMARY KEY,
    product_id INTEGER REFERENCES products(product_id) ON DELETE CASCADE,
    manual_title VARCHAR(200) NOT NULL,
    file_name VARCHAR(255) NOT NULL,
    file_path VARCHAR(500) NOT NULL,           -- OBS 경로
    file_size BIGINT,                          -- 파일 크기 (bytes)
    file_type VARCHAR(50),                     -- PDF, DOCX 등
    version VARCHAR(20),
    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    uploaded_by INTEGER REFERENCES users(user_id),
    is_active BOOLEAN DEFAULT TRUE,
    vector_indexed BOOLEAN DEFAULT FALSE,      -- RAG 인덱싱 여부
    indexed_at TIMESTAMP,
    
    UNIQUE(product_id, version)
);

CREATE INDEX idx_manuals_product ON manuals(product_id);
CREATE INDEX idx_manuals_active ON manuals(is_active);
CREATE INDEX idx_manuals_indexed ON manuals(vector_indexed);

COMMENT ON TABLE manuals IS '제품별 대응 매뉴얼';
COMMENT ON COLUMN manuals.vector_indexed IS 'RAG 벡터 DB 인덱싱 완료 여부';
```

#### manual_sections (매뉴얼 섹션)
```sql
CREATE TABLE manual_sections (
    section_id SERIAL PRIMARY KEY,
    manual_id INTEGER REFERENCES manuals(manual_id) ON DELETE CASCADE,
    defect_type_id INTEGER REFERENCES defect_types(defect_type_id),
    section_type VARCHAR(20) NOT NULL,         -- 'cause' or 'action'
    section_title VARCHAR(200),
    section_content TEXT NOT NULL,
    page_number INTEGER,
    chunk_index INTEGER,                       -- RAG 청크 인덱스
    embedding_vector BYTEA,                    -- 임베딩 벡터 (선택)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT section_type_check CHECK (section_type IN ('cause', 'action', 'other'))
);

CREATE INDEX idx_manual_sections_manual ON manual_sections(manual_id);
CREATE INDEX idx_manual_sections_defect ON manual_sections(defect_type_id);
CREATE INDEX idx_manual_sections_type ON manual_sections(section_type);

COMMENT ON TABLE manual_sections IS '매뉴얼 내 섹션 (원인/조치)';
COMMENT ON COLUMN manual_sections.section_type IS 'cause: 발생원인, action: 조치가이드';
```

---

### 3.4 이미지 관리

#### images (이미지)
```sql
CREATE TABLE images (
    image_id SERIAL PRIMARY KEY,
    product_id INTEGER REFERENCES products(product_id),
    image_type VARCHAR(20) NOT NULL,           -- 'normal' or 'defect'
    defect_type_id INTEGER REFERENCES defect_types(defect_type_id),
    file_name VARCHAR(255) NOT NULL,
    file_path VARCHAR(500) NOT NULL,           -- OBS 경로
    file_size BIGINT,
    width INTEGER,
    height INTEGER,
    format VARCHAR(10),                        -- JPEG, PNG 등
    uploaded_by INTEGER REFERENCES users(user_id),
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    
    CONSTRAINT image_type_check CHECK (image_type IN ('normal', 'defect', 'test')),
    CONSTRAINT defect_image_check CHECK (
        (image_type = 'defect' AND defect_type_id IS NOT NULL) OR
        (image_type != 'defect' AND defect_type_id IS NULL)
    )
);

CREATE INDEX idx_images_product ON images(product_id);
CREATE INDEX idx_images_type ON images(image_type);
CREATE INDEX idx_images_defect ON images(defect_type_id);
CREATE INDEX idx_images_active ON images(is_active);

COMMENT ON TABLE images IS '이미지 메타데이터';
COMMENT ON COLUMN images.image_type IS 'normal: 정상, defect: 불량, test: 검사용';
```

#### image_metadata (이미지 추가 메타데이터)
```sql
CREATE TABLE image_metadata (
    metadata_id SERIAL PRIMARY KEY,
    image_id INTEGER REFERENCES images(image_id) ON DELETE CASCADE,
    exif_data JSONB,                           -- EXIF 정보
    preprocessing_applied JSONB,               -- 전처리 적용 내역
    embedding_vector BYTEA,                    -- CLIP 임베딩 (선택)
    quality_score FLOAT,                       -- 이미지 품질 점수
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_image_metadata_image ON image_metadata(image_id);

COMMENT ON TABLE image_metadata IS '이미지 상세 메타데이터';
```

---

### 3.5 검사 관리

#### inspections (검사)
```sql
CREATE TABLE inspections (
    inspection_id SERIAL PRIMARY KEY,
    inspection_code VARCHAR(50) UNIQUE NOT NULL,  -- 검사 코드 (자동 생성)
    product_id INTEGER REFERENCES products(product_id),
    test_image_id INTEGER REFERENCES images(image_id),
    reference_image_id INTEGER REFERENCES images(image_id),  -- 유사도 검색으로 선택된 이미지
    inspector_id INTEGER REFERENCES users(user_id),
    inspection_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) DEFAULT 'pending',      -- pending, completed, cancelled
    processing_time FLOAT,                     -- 처리 시간 (초)
    
    CONSTRAINT inspection_status_check CHECK (status IN ('pending', 'in_progress', 'completed', 'cancelled'))
);

CREATE INDEX idx_inspections_product ON inspections(product_id);
CREATE INDEX idx_inspections_inspector ON inspections(inspector_id);
CREATE INDEX idx_inspections_status ON inspections(status);
CREATE INDEX idx_inspections_date ON inspections(inspection_date);

COMMENT ON TABLE inspections IS '검사 세션';
COMMENT ON COLUMN inspections.inspection_code IS '검사 고유 코드 (예: INS-20251109-0001)';
```

#### inspection_results (검사 결과)
```sql
CREATE TABLE inspection_results (
    result_id SERIAL PRIMARY KEY,
    inspection_id INTEGER REFERENCES inspections(inspection_id) ON DELETE CASCADE,
    defect_detected BOOLEAN NOT NULL,
    defect_type_id INTEGER REFERENCES defect_types(defect_type_id),
    confidence_score FLOAT,                    -- 불량 확신도 (0~1)
    anomaly_score FLOAT,                       -- PatchCore 이상 점수
    similarity_score FLOAT,                    -- CLIP 유사도 점수
    
    -- 검출 결과 이미지 경로들
    heatmap_path VARCHAR(500),
    mask_path VARCHAR(500),
    overlay_path VARCHAR(500),
    comparison_path VARCHAR(500),
    
    -- 검출된 영역 정보 (JSON)
    detected_regions JSONB,
    
    -- LLM 생성 가이드
    guide_generated BOOLEAN DEFAULT FALSE,
    guide_content TEXT,
    guide_quality_score FLOAT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT confidence_check CHECK (confidence_score BETWEEN 0 AND 1),
    CONSTRAINT anomaly_check CHECK (anomaly_score BETWEEN 0 AND 1),
    CONSTRAINT similarity_check CHECK (similarity_score BETWEEN 0 AND 1)
);

CREATE INDEX idx_results_inspection ON inspection_results(inspection_id);
CREATE INDEX idx_results_defect_type ON inspection_results(defect_type_id);
CREATE INDEX idx_results_detected ON inspection_results(defect_detected);

COMMENT ON TABLE inspection_results IS '검사 결과 상세';
COMMENT ON COLUMN inspection_results.detected_regions IS '검출된 ROI 영역 정보 (JSON 배열)';
```

#### action_logs (조치 기록)
```sql
CREATE TABLE action_logs (
    action_id SERIAL PRIMARY KEY,
    inspection_id INTEGER REFERENCES inspections(inspection_id) ON DELETE CASCADE,
    action_taken TEXT NOT NULL,                -- 수행한 조치 내용
    action_result VARCHAR(50),                 -- 'resolved', 'pending', 'failed'
    action_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    action_by INTEGER REFERENCES users(user_id),
    notes TEXT,
    
    CONSTRAINT action_result_check CHECK (action_result IN ('resolved', 'pending', 'failed', 'deferred'))
);

CREATE INDEX idx_action_logs_inspection ON action_logs(inspection_id);
CREATE INDEX idx_action_logs_result ON action_logs(action_result);
CREATE INDEX idx_action_logs_date ON action_logs(action_date);

COMMENT ON TABLE action_logs IS '불량 조치 이력';
```

---

### 3.6 피드백 관리

#### feedbacks (피드백)
```sql
CREATE TABLE feedbacks (
    feedback_id SERIAL PRIMARY KEY,
    inspection_id INTEGER REFERENCES inspections(inspection_id) ON DELETE CASCADE,
    user_id INTEGER REFERENCES users(user_id),
    rating INTEGER NOT NULL,                   -- 1~5점
    feedback_text TEXT,
    feedback_type VARCHAR(50),                 -- 'guide_quality', 'detection_accuracy' 등
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT rating_check CHECK (rating BETWEEN 1 AND 5)
);

CREATE INDEX idx_feedbacks_inspection ON feedbacks(inspection_id);
CREATE INDEX idx_feedbacks_user ON feedbacks(user_id);
CREATE INDEX idx_feedbacks_rating ON feedbacks(rating);
CREATE INDEX idx_feedbacks_type ON feedbacks(feedback_type);

COMMENT ON TABLE feedbacks IS '사용자 피드백';
COMMENT ON COLUMN feedbacks.rating IS '1: 매우 불만족 ~ 5: 매우 만족';
```

---

### 3.7 모델 배포 관리

#### deployments (배포)
```sql
CREATE TABLE deployments (
    deployment_id SERIAL PRIMARY KEY,
    deployment_type VARCHAR(50) NOT NULL,      -- 'clip_embedding', 'patchcore_bank', 'rag_indexing'
    product_id INTEGER REFERENCES products(product_id),
    status VARCHAR(20) DEFAULT 'pending',      -- pending, running, completed, failed
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    deployed_by INTEGER REFERENCES users(user_id),
    
    -- 배포 상세 정보
    config_params JSONB,                       -- 배포 설정 파라미터
    result_summary JSONB,                      -- 배포 결과 요약
    error_message TEXT,
    
    CONSTRAINT deployment_type_check CHECK (deployment_type IN ('clip_embedding', 'patchcore_bank', 'rag_indexing', 'model_update')),
    CONSTRAINT deployment_status_check CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled'))
);

CREATE INDEX idx_deployments_type ON deployments(deployment_type);
CREATE INDEX idx_deployments_product ON deployments(product_id);
CREATE INDEX idx_deployments_status ON deployments(status);
CREATE INDEX idx_deployments_started ON deployments(started_at);

COMMENT ON TABLE deployments IS '모델 배포 이력';
COMMENT ON COLUMN deployments.deployment_type IS 'clip_embedding: CLIP 재구축, patchcore_bank: 메모리뱅크 생성';
```

#### model_configs (모델 설정)
```sql
CREATE TABLE model_configs (
    config_id SERIAL PRIMARY KEY,
    product_id INTEGER REFERENCES products(product_id),
    model_type VARCHAR(50) NOT NULL,           -- 'anomaly_detection', 'object_detection'
    config_name VARCHAR(100),
    
    -- 전처리 설정
    preprocessing JSONB,                       -- grayscale, contrast, histogram 등
    
    -- Anomaly Detection 파라미터
    image_threshold FLOAT DEFAULT 0.85,
    pixel_threshold FLOAT DEFAULT 0.90,
    
    -- Object Detection 파라미터 (선택)
    detection_model VARCHAR(100),              -- yolov8, etc.
    detection_confidence FLOAT DEFAULT 0.5,
    
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT model_type_check CHECK (model_type IN ('anomaly_detection', 'object_detection', 'hybrid'))
);

CREATE INDEX idx_model_configs_product ON model_configs(product_id);
CREATE INDEX idx_model_configs_type ON model_configs(model_type);
CREATE INDEX idx_model_configs_active ON model_configs(is_active);

COMMENT ON TABLE model_configs IS '제품별 모델 설정';
```

---

### 3.8 통계 및 대시보드 뷰

#### dashboard_stats (통합 통계 뷰)
```sql
CREATE OR REPLACE VIEW dashboard_stats AS
SELECT
    p.product_id,
    p.product_name,
    COUNT(DISTINCT CASE WHEN i.image_type = 'normal' THEN i.image_id END) AS normal_image_count,
    COUNT(DISTINCT CASE WHEN i.image_type = 'defect' THEN i.image_id END) AS defect_image_count,
    COUNT(DISTINCT ins.inspection_id) AS total_inspection_count,
    COUNT(DISTINCT CASE WHEN ir.defect_detected = TRUE THEN ins.inspection_id END) AS defect_detected_count,
    COUNT(DISTINCT CASE WHEN al.action_result = 'resolved' THEN ins.inspection_id END) AS resolved_count,
    COUNT(DISTINCT CASE WHEN al.action_result IS NULL AND ir.defect_detected = TRUE THEN ins.inspection_id END) AS pending_action_count,
    AVG(CASE WHEN f.feedback_type = 'guide_quality' THEN f.rating END) AS avg_guide_rating
FROM products p
LEFT JOIN images i ON p.product_id = i.product_id AND i.is_active = TRUE
LEFT JOIN inspections ins ON p.product_id = ins.product_id
LEFT JOIN inspection_results ir ON ins.inspection_id = ir.inspection_id
LEFT JOIN action_logs al ON ins.inspection_id = al.inspection_id
LEFT JOIN feedbacks f ON ins.inspection_id = f.inspection_id
WHERE p.is_active = TRUE
GROUP BY p.product_id, p.product_name;

COMMENT ON VIEW dashboard_stats IS '대시보드 통합 통계';
```

#### recent_inspections (최근 검사 뷰)
```sql
CREATE OR REPLACE VIEW recent_inspections AS
SELECT
    ins.inspection_id,
    ins.inspection_code,
    ins.inspection_date,
    p.product_code,
    p.product_name,
    dt.defect_name_ko,
    ir.defect_detected,
    ir.confidence_score,
    u.username AS inspector,
    CASE
        WHEN al.action_result = 'resolved' THEN '조치완료'
        WHEN ir.defect_detected = TRUE AND al.action_result IS NULL THEN '미조치'
        ELSE '정상'
    END AS action_status
FROM inspections ins
JOIN products p ON ins.product_id = p.product_id
LEFT JOIN inspection_results ir ON ins.inspection_id = ir.inspection_id
LEFT JOIN defect_types dt ON ir.defect_type_id = dt.defect_type_id
LEFT JOIN users u ON ins.inspector_id = u.user_id
LEFT JOIN action_logs al ON ins.inspection_id = al.inspection_id
WHERE ins.status = 'completed'
ORDER BY ins.inspection_date DESC
LIMIT 100;

COMMENT ON VIEW recent_inspections IS '최근 검사 이력 (상위 100건)';
```

---

## 4. 초기 데이터 삽입

```sql
-- 기본 역할 생성 (이미 위에서 수행)

-- 관리자 계정 생성 (비밀번호: admin123 - 실제로는 해시 필요)
INSERT INTO users (username, email, password_hash, full_name) VALUES
('admin', 'admin@example.com', '$2b$12$...', '시스템 관리자'),
('manager1', 'manager@example.com', '$2b$12$...', '제조 관리자'),
('worker1', 'worker@example.com', '$2b$12$...', '작업자1');

-- 역할 할당
INSERT INTO user_roles (user_id, role_id) VALUES
(1, 1),  -- admin
(2, 2),  -- manager
(3, 3);  -- worker

-- 제품 생성
INSERT INTO products (product_code, product_name, description, created_by) VALUES
('prod1', '주조 제품 A형', '주조 공정 제품', 1),
('prod2', '주조 제품 B형', '주조 공정 제품', 1),
('prod3', '주조 제품 C형', '주조 공정 제품', 1);

-- 불량 유형 생성
INSERT INTO defect_types (product_id, defect_code, defect_name_ko, defect_name_en, full_name_ko, severity_level) VALUES
(1, 'hole', '기공', 'hole', '기공 (주조 결함)', 4),
(1, 'burr', '버', 'burr', '버 (날카로운 돌기)', 3),
(1, 'scratch', '긁힘', 'scratch', '긁힘 (표면 손상)', 2);
```

---

## 5. 인덱스 및 성능 최적화

### 5.1 파티셔닝 (대량 데이터 대비)
```sql
-- inspections 테이블 월별 파티셔닝 (선택적)
CREATE TABLE inspections_2025_11 PARTITION OF inspections
FOR VALUES FROM ('2025-11-01') TO ('2025-12-01');

CREATE TABLE inspections_2025_12 PARTITION OF inspections
FOR VALUES FROM ('2025-12-01') TO ('2026-01-01');
```

### 5.2 추가 인덱스
```sql
-- 복합 인덱스
CREATE INDEX idx_inspections_product_date ON inspections(product_id, inspection_date DESC);
CREATE INDEX idx_results_inspection_defect ON inspection_results(inspection_id, defect_type_id);

-- JSONB 인덱스
CREATE INDEX idx_detected_regions_gin ON inspection_results USING GIN (detected_regions);
CREATE INDEX idx_config_params_gin ON model_configs USING GIN (preprocessing);
```

### 5.3 통계 수집
```sql
ANALYZE users;
ANALYZE products;
ANALYZE images;
ANALYZE inspections;
ANALYZE inspection_results;
```

---

## 6. 백업 및 유지보수

### 6.1 백업 스크립트
```bash
#!/bin/bash
# PostgreSQL 백업 스크립트

DB_NAME="defect_detection_db"
BACKUP_DIR="/backup/postgresql"
DATE=$(date +%Y%m%d_%H%M%S)

pg_dump -U postgres $DB_NAME | gzip > $BACKUP_DIR/${DB_NAME}_${DATE}.sql.gz

# 30일 이상 된 백업 삭제
find $BACKUP_DIR -name "*.sql.gz" -mtime +30 -delete
```

### 6.2 정기 유지보수
```sql
-- 주간 VACUUM
VACUUM ANALYZE;

-- 월간 전체 VACUUM
VACUUM FULL;

-- 인덱스 재구축 (필요시)
REINDEX DATABASE defect_detection_db;
```

---

## 7. 마이그레이션 전략

### 7.1 현재 → PostgreSQL 마이그레이션

```python
# 마이그레이션 스크립트 예시
import psycopg2
from pathlib import Path
import json

def migrate_products():
    """제품 데이터 마이그레이션"""
    # 1. 기존 defect_mapping.json 읽기
    with open('/home/dmillion/llm_chal_vlm/web/defect_mapping.json') as f:
        mapping = json.load(f)
    
    # 2. PostgreSQL에 삽입
    conn = psycopg2.connect("dbname=defect_detection_db user=postgres")
    cur = conn.cursor()
    
    for product_code, defects in mapping.items():
        # 제품 삽입
        cur.execute("""
            INSERT INTO products (product_code, product_name, created_by)
            VALUES (%s, %s, 1)
            RETURNING product_id
        """, (product_code, f"제품 {product_code}"))
        
        product_id = cur.fetchone()[0]
        
        # 불량 타입 삽입
        for defect_code, info in defects.items():
            cur.execute("""
                INSERT INTO defect_types 
                (product_id, defect_code, defect_name_ko, defect_name_en, full_name_ko)
                VALUES (%s, %s, %s, %s, %s)
            """, (product_id, info['en'], info['ko'], info['en'], info.get('full_name_ko', '')))
    
    conn.commit()
    cur.close()
    conn.close()

def migrate_images():
    """이미지 메타데이터 마이그레이션"""
    # 파일 시스템 스캔 → DB 삽입
    pass
```

---

## 8. API 연동 예시

### 8.1 제품 조회 API
```python
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

router = APIRouter()

@router.get("/admin/products")
async def get_products(db: Session = Depends(get_db)):
    """제품 목록 조회"""
    products = db.query(Product).filter(Product.is_active == True).all()
    return products

@router.post("/admin/products")
async def create_product(product: ProductCreate, db: Session = Depends(get_db)):
    """제품 생성"""
    new_product = Product(**product.dict())
    db.add(new_product)
    db.commit()
    db.refresh(new_product)
    return new_product
```

### 8.2 검사 생성 API
```python
@router.post("/worker/inspections")
async def create_inspection(
    test_image: UploadFile,
    product_id: int,
    db: Session = Depends(get_db)
):
    """검사 생성 및 실행"""
    # 1. 이미지 저장
    image = save_image_to_obs(test_image, product_id)
    
    # 2. 검사 레코드 생성
    inspection = Inspection(
        product_id=product_id,
        test_image_id=image.image_id,
        inspector_id=current_user.user_id,
        status='in_progress'
    )
    db.add(inspection)
    db.commit()
    
    # 3. AI 모델 실행 (비동기)
    # ... CLIP 검색, PatchCore 검출 등
    
    # 4. 결과 저장
    result = InspectionResult(
        inspection_id=inspection.inspection_id,
        defect_detected=True,
        defect_type_id=detected_defect_id,
        confidence_score=0.95,
        # ...
    )
    db.add(result)
    db.commit()
    
    return inspection
```

---

## 9. 보안 고려사항

### 9.1 Row-Level Security (RLS)
```sql
-- 작업자는 자신의 검사만 조회 가능
ALTER TABLE inspections ENABLE ROW LEVEL SECURITY;

CREATE POLICY inspector_isolation ON inspections
    FOR SELECT
    USING (inspector_id = current_user_id());
```

### 9.2 감사 로그
```sql
CREATE TABLE audit_logs (
    log_id SERIAL PRIMARY KEY,
    table_name VARCHAR(50),
    operation VARCHAR(10),      -- INSERT, UPDATE, DELETE
    user_id INTEGER,
    old_data JSONB,
    new_data JSONB,
    logged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

**작성일**: 2025-11-09  
**버전**: 1.0  
**DBMS**: PostgreSQL 14+

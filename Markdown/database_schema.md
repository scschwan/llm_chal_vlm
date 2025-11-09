# PostgreSQL 데이터베이스 스키마 설계 (단순화 버전)

## 1. 데이터베이스 개요

### 1.1 설계 원칙
- **단순성 우선**: 필수 기능만 포함한 최소 구조
- **확장 가능**: 필요시 테이블 추가 가능한 유연한 설계
- **JSON 활용**: 복잡한 데이터는 JSONB로 유연하게 저장

### 1.2 테이블 구성 (10개)
```
1. users              - 사용자 (관리자/작업자)
2. products           - 제품
3. manuals            - 매뉴얼
4. defect_types       - 불량 유형
5. images             - 이미지 메타데이터
6. search_history     - 유사도 검색 이력
7. response_history   - 대응 매뉴얼 생성 이력
8. model_params       - 모델 파라미터 설정
9. deployment_logs    - 배포 실행 이력
10. system_config     - 전역 설정 (Key-Value)
```

---

## 2. ERD (Entity Relationship Diagram)

```
┌──────────────┐
│    users     │
└──────────────┘
       │ created_by
       ├─────────────────┬──────────────┐
       ▼                 ▼              ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  products    │  │   manuals    │  │    images    │
└──────────────┘  └──────────────┘  └──────────────┘
       │                 │                 │
       │ product_id      │ product_id      │ product_id
       ├─────────────────┼─────────────────┤
       │                 │                 │
       ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│defect_types  │  │model_params  │  │deployment_   │
└──────────────┘  └──────────────┘  │logs          │
       │                             └──────────────┘
       │ defect_type_id
       ▼
┌──────────────┐
│search_       │
│history       │
└──────────────┘
       │
       │ search_id
       ▼
┌──────────────┐
│response_     │
│history       │
└──────────────┘

┌──────────────┐
│system_config │  (독립 테이블)
└──────────────┘
```

---

## 3. 테이블 스키마 상세

### 3.1 users (사용자)

```sql
CREATE TABLE users (
    user_id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    user_type VARCHAR(20) NOT NULL DEFAULT 'worker',
    full_name VARCHAR(100),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login_at TIMESTAMP,
    
    CONSTRAINT users_type_check CHECK (user_type IN ('admin', 'worker'))
);

CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_type ON users(user_type);

COMMENT ON TABLE users IS '사용자 계정';
COMMENT ON COLUMN users.user_type IS 'admin: 관리자, worker: 작업자';

-- 초기 데이터
INSERT INTO users (username, password_hash, user_type, full_name) VALUES
('admin', '$2b$12$...', 'admin', '시스템 관리자'),
('worker1', '$2b$12$...', 'worker', '작업자1');
```

---

### 3.2 products (제품)

```sql
CREATE TABLE products (
    product_id SERIAL PRIMARY KEY,
    product_code VARCHAR(50) UNIQUE NOT NULL,
    product_name VARCHAR(100) NOT NULL,
    description TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_products_code ON products(product_code);
CREATE INDEX idx_products_active ON products(is_active);

COMMENT ON TABLE products IS '제품 마스터';
COMMENT ON COLUMN products.product_code IS '제품 코드 (예: prod1, prod2)';

-- 초기 데이터
INSERT INTO products (product_code, product_name, description) VALUES
('prod1', '주조 제품 A형', '주조 공정 제품'),
('prod2', '주조 제품 B형', '주조 공정 제품'),
('prod3', '주조 제품 C형', '주조 공정 제품');
```

---

### 3.3 manuals (매뉴얼)

```sql
CREATE TABLE manuals (
    manual_id SERIAL PRIMARY KEY,
    product_id INTEGER REFERENCES products(product_id) ON DELETE CASCADE,
    file_name VARCHAR(255) NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    file_size BIGINT,
    vector_indexed BOOLEAN DEFAULT FALSE,
    indexed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(product_id, file_name)
);

CREATE INDEX idx_manuals_product ON manuals(product_id);
CREATE INDEX idx_manuals_indexed ON manuals(vector_indexed);

COMMENT ON TABLE manuals IS '제품별 대응 매뉴얼';
COMMENT ON COLUMN manuals.file_path IS 'OBS 또는 로컬 파일 경로';
COMMENT ON COLUMN manuals.vector_indexed IS 'RAG 벡터 DB 인덱싱 완료 여부';

-- 초기 데이터 예시
INSERT INTO manuals (product_id, file_name, file_path, vector_indexed) VALUES
(1, 'prod1_menual.pdf', '/manual_store/prod1_menual.pdf', TRUE);
```

---

### 3.4 defect_types (불량 유형)

```sql
CREATE TABLE defect_types (
    defect_type_id SERIAL PRIMARY KEY,
    product_id INTEGER REFERENCES products(product_id) ON DELETE CASCADE,
    defect_code VARCHAR(50) NOT NULL,
    defect_name_ko VARCHAR(100) NOT NULL,
    defect_name_en VARCHAR(100),
    full_name_ko VARCHAR(200),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(product_id, defect_code)
);

CREATE INDEX idx_defect_types_product ON defect_types(product_id);
CREATE INDEX idx_defect_types_code ON defect_types(defect_code);

COMMENT ON TABLE defect_types IS '제품별 불량 유형 (정상 포함)';
COMMENT ON COLUMN defect_types.defect_code IS '영문 코드 (normal, hole, burr, scratch)';
COMMENT ON COLUMN defect_types.defect_name_ko IS '한글 명칭 (정상, 기공, 버, 긁힘)';

-- 초기 데이터
INSERT INTO defect_types (product_id, defect_code, defect_name_ko, defect_name_en, full_name_ko) VALUES
(1, 'normal', '정상', 'normal', '정상 제품'),
(1, 'hole', '기공', 'hole', '기공 (주조 결함)'),
(1, 'burr', '버', 'burr', '버 (날카로운 돌기)'),
(1, 'scratch', '긁힘', 'scratch', '긁힘 (표면 손상)');
```

---

### 3.5 images (이미지 메타데이터)

```sql
CREATE TABLE images (
    image_id SERIAL PRIMARY KEY,
    product_id INTEGER REFERENCES products(product_id) ON DELETE CASCADE,
    image_type VARCHAR(20) NOT NULL,
    defect_type_id INTEGER REFERENCES defect_types(defect_type_id),
    file_name VARCHAR(255) NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    file_size BIGINT,
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT image_type_check CHECK (image_type IN ('normal', 'defect', 'test'))
);

CREATE INDEX idx_images_product ON images(product_id);
CREATE INDEX idx_images_type ON images(image_type);
CREATE INDEX idx_images_defect ON images(defect_type_id);

COMMENT ON TABLE images IS '등록된 이미지 메타데이터';
COMMENT ON COLUMN images.image_type IS 'normal: 정상, defect: 불량, test: 검사용';
COMMENT ON COLUMN images.file_path IS 'OBS 또는 로컬 파일 경로';

-- 초기 데이터 예시
INSERT INTO images (product_id, image_type, defect_type_id, file_name, file_path) VALUES
(1, 'normal', NULL, 'prod1_ok_0_129.jpeg', '/data/patchCore/prod1/ok/prod1_ok_0_129.jpeg'),
(1, 'defect', 2, 'prod1_hole_001.jpeg', '/data/def_split/prod1_hole_001.jpeg'),
(1, 'defect', 3, 'prod1_burr_001.jpeg', '/data/def_split/prod1_burr_001.jpeg');
```

---

### 3.6 search_history (유사도 검색 이력)

```sql
CREATE TABLE search_history (
    search_id SERIAL PRIMARY KEY,
    searched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    uploaded_image_path VARCHAR(500) NOT NULL,
    product_code VARCHAR(50),
    defect_code VARCHAR(50),
    top_k_results JSONB NOT NULL,
    processing_time FLOAT,
    
    CONSTRAINT search_topk_check CHECK (jsonb_typeof(top_k_results) = 'array')
);

CREATE INDEX idx_search_history_date ON search_history(searched_at DESC);
CREATE INDEX idx_search_history_product ON search_history(product_code);
CREATE INDEX idx_search_history_defect ON search_history(defect_code);
CREATE INDEX idx_search_topk_gin ON search_history USING GIN (top_k_results);

COMMENT ON TABLE search_history IS '유사도 검색 실행 이력';
COMMENT ON COLUMN search_history.top_k_results IS 'TOP-K 결과 JSON 배열: [{"rank": 1, "image_path": "...", "similarity": 0.98}, ...]';

-- JSON 구조 예시
/*
top_k_results 형식:
[
  {
    "rank": 1,
    "image_path": "/data/def_split/prod1_burr_021.jpeg",
    "image_name": "prod1_burr_021.jpeg",
    "similarity": 0.9884
  },
  {
    "rank": 2,
    "image_path": "/data/def_split/prod1_burr_013.jpeg",
    "image_name": "prod1_burr_013.jpeg",
    "similarity": 0.9521
  }
]
*/
```

---

### 3.7 response_history (대응 매뉴얼 생성 이력)

```sql
CREATE TABLE response_history (
    response_id SERIAL PRIMARY KEY,
    executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    search_id INTEGER REFERENCES search_history(search_id) ON DELETE SET NULL,
    
    -- 검출 결과
    product_code VARCHAR(50) NOT NULL,
    defect_code VARCHAR(50) NOT NULL,
    similarity_score FLOAT,
    anomaly_score FLOAT,
    confidence_score FLOAT,
    
    -- 이미지 경로
    test_image_path VARCHAR(500),
    reference_image_path VARCHAR(500),
    heatmap_path VARCHAR(500),
    overlay_path VARCHAR(500),
    
    -- LLM 생성 가이드
    guide_content TEXT,
    guide_generated_at TIMESTAMP,
    
    -- 피드백
    feedback_rating INTEGER,
    feedback_text TEXT,
    feedback_at TIMESTAMP,
    
    processing_time FLOAT,
    
    CONSTRAINT feedback_rating_check CHECK (feedback_rating BETWEEN 1 AND 5)
);

CREATE INDEX idx_response_history_date ON response_history(executed_at DESC);
CREATE INDEX idx_response_history_search ON response_history(search_id);
CREATE INDEX idx_response_history_product ON response_history(product_code);
CREATE INDEX idx_response_history_defect ON response_history(defect_code);
CREATE INDEX idx_response_history_rating ON response_history(feedback_rating);

COMMENT ON TABLE response_history IS '대응 매뉴얼 생성 및 피드백 이력';
COMMENT ON COLUMN response_history.search_id IS '연관된 유사도 검색 ID';
COMMENT ON COLUMN response_history.guide_content IS 'LLM이 생성한 대응 매뉴얼 전문';
COMMENT ON COLUMN response_history.feedback_rating IS '사용자 평가 (1~5점)';
```

---

### 3.8 model_params (모델 파라미터 설정)

```sql
CREATE TABLE model_params (
    param_id SERIAL PRIMARY KEY,
    product_id INTEGER REFERENCES products(product_id) ON DELETE CASCADE,
    model_type VARCHAR(50) NOT NULL,
    params JSONB NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT model_type_check CHECK (model_type IN ('clip', 'patchcore', 'llm', 'preprocessing'))
);

CREATE INDEX idx_model_params_product ON model_params(product_id);
CREATE INDEX idx_model_params_type ON model_params(model_type);
CREATE INDEX idx_model_params_active ON model_params(is_active);
CREATE INDEX idx_model_params_gin ON model_params USING GIN (params);

COMMENT ON TABLE model_params IS '제품별 모델 파라미터 설정';
COMMENT ON COLUMN model_params.model_type IS 'clip, patchcore, llm, preprocessing';
COMMENT ON COLUMN model_params.params IS '모델별 파라미터 JSON';

-- 초기 데이터 예시
INSERT INTO model_params (product_id, model_type, params, is_active) VALUES
(1, 'clip', '{"model_id": "ViT-B-32/openai", "top_k": 5}', TRUE),
(1, 'patchcore', '{"image_threshold": 0.85, "pixel_threshold": 0.90}', TRUE),
(1, 'llm', '{"model_name": "mistralai/Mistral-7B-Instruct-v0.2", "temperature": 0.7, "max_tokens": 512}', TRUE),
(1, 'preprocessing', '{"grayscale": false, "contrast": false, "histogram": false}', TRUE);

-- JSON 구조 예시
/*
CLIP 파라미터:
{
  "model_id": "ViT-B-32/openai",
  "top_k": 5,
  "use_fp16": false
}

PatchCore 파라미터:
{
  "image_threshold": 0.85,
  "pixel_threshold": 0.90,
  "backbone": "wide_resnet50_2"
}

LLM 파라미터:
{
  "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
  "temperature": 0.7,
  "max_tokens": 512,
  "use_4bit": true
}

전처리 파라미터:
{
  "grayscale": false,
  "contrast": true,
  "histogram": false
}
*/
```

---

### 3.9 deployment_logs (배포 실행 이력)

```sql
CREATE TABLE deployment_logs (
    deploy_id SERIAL PRIMARY KEY,
    deploy_type VARCHAR(50) NOT NULL,
    product_id INTEGER REFERENCES products(product_id),
    status VARCHAR(20) DEFAULT 'pending',
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    result_message TEXT,
    result_data JSONB,
    deployed_by INTEGER REFERENCES users(user_id),
    
    CONSTRAINT deploy_type_check CHECK (deploy_type IN ('clip_rebuild', 'patchcore_create', 'rag_index', 'full_deploy')),
    CONSTRAINT deploy_status_check CHECK (status IN ('pending', 'running', 'completed', 'failed'))
);

CREATE INDEX idx_deployment_logs_type ON deployment_logs(deploy_type);
CREATE INDEX idx_deployment_logs_product ON deployment_logs(product_id);
CREATE INDEX idx_deployment_logs_status ON deployment_logs(status);
CREATE INDEX idx_deployment_logs_started ON deployment_logs(started_at DESC);

COMMENT ON TABLE deployment_logs IS '서버 배포 실행 이력';
COMMENT ON COLUMN deployment_logs.deploy_type IS 'clip_rebuild: CLIP 재구축, patchcore_create: 메모리뱅크 생성, rag_index: RAG 인덱싱';
COMMENT ON COLUMN deployment_logs.result_data IS '배포 결과 상세 정보 JSON';

-- 초기 데이터 예시
INSERT INTO deployment_logs (deploy_type, product_id, status, completed_at, result_message, result_data, deployed_by) VALUES
('clip_rebuild', 1, 'completed', CURRENT_TIMESTAMP, 'CLIP 인덱스 재구축 완료', 
 '{"num_images": 50, "embedding_dim": 512, "processing_time": 15.3}', 1),
('patchcore_create', 1, 'completed', CURRENT_TIMESTAMP, 'PatchCore 메모리뱅크 생성 완료',
 '{"num_patches": 614, "feature_dim": 3584, "processing_time": 25.7}', 1);

-- JSON 구조 예시
/*
CLIP 재구축 결과:
{
  "num_images": 50,
  "embedding_dim": 512,
  "processing_time": 15.3,
  "index_file": "/web/index_cache/index_data.pt"
}

PatchCore 생성 결과:
{
  "num_patches": 614,
  "feature_dim": 3584,
  "memory_bank_file": "/data/patchCore/prod1/bank.pt",
  "processing_time": 25.7
}

RAG 인덱싱 결과:
{
  "num_chunks": 120,
  "embedding_dim": 768,
  "vector_db_path": "/manual_store",
  "processing_time": 8.5
}
*/
```

---

### 3.10 system_config (전역 설정)

```sql
CREATE TABLE system_config (
    config_key VARCHAR(100) PRIMARY KEY,
    config_value JSONB NOT NULL,
    description TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_by INTEGER REFERENCES users(user_id)
);

CREATE INDEX idx_system_config_updated ON system_config(updated_at DESC);

COMMENT ON TABLE system_config IS '시스템 전역 설정 (Key-Value)';

-- 초기 데이터
INSERT INTO system_config (config_key, config_value, description) VALUES
('default_top_k', '5', '기본 TOP-K 개수'),
('default_clip_model', '"ViT-B-32/openai"', '기본 CLIP 모델'),
('default_image_threshold', '0.85', '기본 이미지 임계값'),
('default_pixel_threshold', '0.90', '기본 픽셀 임계값'),
('llm_temperature', '0.7', 'LLM 샘플링 온도'),
('max_upload_size_mb', '50', '최대 업로드 크기 (MB)'),
('session_timeout_minutes', '60', '세션 타임아웃 (분)');

-- JSON 구조 예시
/*
단순 값:
config_value: 5
config_value: "ViT-B-32/openai"
config_value: 0.85

복잡한 설정:
config_value: {
  "enabled": true,
  "retry_count": 3,
  "timeout": 30
}
*/
```

---

## 4. 데이터베이스 생성 스크립트

### 4.1 전체 스키마 생성

```sql
-- 데이터베이스 생성
CREATE DATABASE defect_detection_db
    WITH 
    ENCODING = 'UTF8'
    LC_COLLATE = 'ko_KR.UTF-8'
    LC_CTYPE = 'ko_KR.UTF-8'
    TEMPLATE = template0;

\c defect_detection_db

-- 확장 기능 활성화
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- 테이블 생성 (순서 중요 - 외래키 제약 고려)
\i 01_users.sql
\i 02_products.sql
\i 03_manuals.sql
\i 04_defect_types.sql
\i 05_images.sql
\i 06_search_history.sql
\i 07_response_history.sql
\i 08_model_params.sql
\i 09_deployment_logs.sql
\i 10_system_config.sql

-- 초기 데이터 삽입
\i init_data.sql
```

---

## 5. 유용한 뷰 (View)

### 5.1 대시보드 통계 뷰

```sql
CREATE OR REPLACE VIEW v_dashboard_stats AS
SELECT
    p.product_code,
    p.product_name,
    COUNT(DISTINCT CASE WHEN i.image_type = 'normal' THEN i.image_id END) AS normal_count,
    COUNT(DISTINCT CASE WHEN i.image_type = 'defect' THEN i.image_id END) AS defect_count,
    COUNT(DISTINCT dt.defect_type_id) - 1 AS defect_type_count,  -- 정상 제외
    COUNT(DISTINCT sh.search_id) AS total_search_count,
    COUNT(DISTINCT rh.response_id) AS total_response_count,
    COUNT(DISTINCT CASE WHEN rh.feedback_rating IS NOT NULL THEN rh.response_id END) AS feedback_count,
    ROUND(AVG(rh.feedback_rating)::numeric, 2) AS avg_rating
FROM products p
LEFT JOIN images i ON p.product_id = i.product_id
LEFT JOIN defect_types dt ON p.product_id = dt.product_id
LEFT JOIN search_history sh ON p.product_code = sh.product_code
LEFT JOIN response_history rh ON p.product_code = rh.product_code
WHERE p.is_active = TRUE
GROUP BY p.product_id, p.product_code, p.product_name;

COMMENT ON VIEW v_dashboard_stats IS '대시보드 제품별 통계';
```

### 5.2 최근 검사 이력 뷰

```sql
CREATE OR REPLACE VIEW v_recent_inspections AS
SELECT
    rh.response_id,
    rh.executed_at,
    rh.product_code,
    p.product_name,
    rh.defect_code,
    dt.defect_name_ko,
    rh.similarity_score,
    rh.anomaly_score,
    CASE 
        WHEN rh.feedback_rating IS NOT NULL THEN '조치완료'
        ELSE '미조치'
    END AS status,
    rh.feedback_rating
FROM response_history rh
LEFT JOIN products p ON rh.product_code = p.product_code
LEFT JOIN defect_types dt ON rh.product_code = dt.product_id::text 
    AND rh.defect_code = dt.defect_code
ORDER BY rh.executed_at DESC
LIMIT 50;

COMMENT ON VIEW v_recent_inspections IS '최근 검사 이력 (상위 50건)';
```

### 5.3 배포 상태 뷰

```sql
CREATE OR REPLACE VIEW v_deployment_status AS
SELECT
    dl.deploy_type,
    p.product_code,
    p.product_name,
    dl.status,
    dl.started_at,
    dl.completed_at,
    EXTRACT(EPOCH FROM (COALESCE(dl.completed_at, NOW()) - dl.started_at)) AS duration_seconds,
    dl.result_message
FROM deployment_logs dl
LEFT JOIN products p ON dl.product_id = p.product_id
WHERE dl.deploy_id IN (
    SELECT MAX(deploy_id)
    FROM deployment_logs
    GROUP BY deploy_type, product_id
)
ORDER BY dl.started_at DESC;

COMMENT ON VIEW v_deployment_status IS '제품별 최신 배포 상태';
```

---

## 6. 인덱스 및 성능 최적화

### 6.1 추가 인덱스

```sql
-- 복합 인덱스
CREATE INDEX idx_search_product_date ON search_history(product_code, searched_at DESC);
CREATE INDEX idx_response_product_date ON response_history(product_code, executed_at DESC);
CREATE INDEX idx_response_feedback ON response_history(feedback_rating) WHERE feedback_rating IS NOT NULL;

-- 부분 인덱스
CREATE INDEX idx_images_normal ON images(product_id) WHERE image_type = 'normal';
CREATE INDEX idx_images_defect ON images(product_id, defect_type_id) WHERE image_type = 'defect';
CREATE INDEX idx_deployment_running ON deployment_logs(deploy_type, product_id) WHERE status = 'running';
```

### 6.2 통계 수집

```sql
-- 정기 통계 수집 (cron으로 실행)
ANALYZE users;
ANALYZE products;
ANALYZE images;
ANALYZE search_history;
ANALYZE response_history;
ANALYZE deployment_logs;
```

---

## 7. 백업 및 복원

### 7.1 백업 스크립트

```bash
#!/bin/bash
# backup_db.sh

DB_NAME="defect_detection_db"
BACKUP_DIR="/backup/postgresql"
DATE=$(date +%Y%m%d_%H%M%S)

# 전체 백업
pg_dump -U postgres -F c $DB_NAME > $BACKUP_DIR/${DB_NAME}_${DATE}.dump

# 압축 백업
pg_dump -U postgres $DB_NAME | gzip > $BACKUP_DIR/${DB_NAME}_${DATE}.sql.gz

# 30일 이상 된 백업 삭제
find $BACKUP_DIR -name "*.dump" -mtime +30 -delete
find $BACKUP_DIR -name "*.sql.gz" -mtime +30 -delete

echo "Backup completed: ${DB_NAME}_${DATE}"
```

### 7.2 복원

```bash
# .dump 파일 복원
pg_restore -U postgres -d defect_detection_db /backup/postgresql/defect_detection_db_20251109.dump

# .sql.gz 파일 복원
gunzip -c /backup/postgresql/defect_detection_db_20251109.sql.gz | psql -U postgres defect_detection_db
```

---

## 8. 마이그레이션 스크립트 예시

### 8.1 기존 데이터 → PostgreSQL

```python
# migrate_to_postgres.py
import psycopg2
import json
from pathlib import Path

def migrate_products():
    """제품 및 불량 타입 마이그레이션"""
    conn = psycopg2.connect("dbname=defect_detection_db user=postgres")
    cur = conn.cursor()
    
    # defect_mapping.json 읽기
    mapping_file = Path('/home/dmillion/llm_chal_vlm/web/defect_mapping.json')
    with open(mapping_file) as f:
        mapping = json.load(f)
    
    for product_code, defects in mapping.items():
        # 제품 삽입
        cur.execute("""
            INSERT INTO products (product_code, product_name)
            VALUES (%s, %s)
            ON CONFLICT (product_code) DO NOTHING
            RETURNING product_id
        """, (product_code, f"제품 {product_code}"))
        
        result = cur.fetchone()
        if result:
            product_id = result[0]
            
            # 정상 타입 먼저 삽입
            cur.execute("""
                INSERT INTO defect_types (product_id, defect_code, defect_name_ko, defect_name_en)
                VALUES (%s, 'normal', '정상', 'normal')
                ON CONFLICT DO NOTHING
            """, (product_id,))
            
            # 불량 타입 삽입
            for defect_info in defects.values():
                cur.execute("""
                    INSERT INTO defect_types 
                    (product_id, defect_code, defect_name_ko, defect_name_en, full_name_ko)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT DO NOTHING
                """, (
                    product_id,
                    defect_info['en'],
                    defect_info['ko'],
                    defect_info['en'],
                    defect_info.get('full_name_ko', '')
                ))
    
    conn.commit()
    cur.close()
    conn.close()
    print("✅ 제품 및 불량 타입 마이그레이션 완료")

def migrate_images():
    """이미지 메타데이터 마이그레이션"""
    conn = psycopg2.connect("dbname=defect_detection_db user=postgres")
    cur = conn.cursor()
    
    # 정상 이미지 스캔
    normal_dir = Path('/home/dmillion/llm_chal_vlm/data/patchCore/prod1/ok')
    for img_file in normal_dir.glob('*.jpeg'):
        product_code = img_file.stem.split('_')[0]  # prod1_ok_0_129 → prod1
        
        cur.execute("""
            INSERT INTO images (product_id, image_type, file_name, file_path)
            SELECT product_id, 'normal', %s, %s
            FROM products WHERE product_code = %s
            ON CONFLICT DO NOTHING
        """, (img_file.name, str(img_file), product_code))
    
    # 불량 이미지 스캔
    defect_dir = Path('/home/dmillion/llm_chal_vlm/data/def_split')
    for img_file in defect_dir.glob('*.jpeg'):
        parts = img_file.stem.split('_')  # prod1_hole_001
        if len(parts) >= 3:
            product_code = parts[0]
            defect_code = parts[1]
            
            cur.execute("""
                INSERT INTO images (product_id, image_type, defect_type_id, file_name, file_path)
                SELECT p.product_id, 'defect', dt.defect_type_id, %s, %s
                FROM products p
                JOIN defect_types dt ON p.product_id = dt.product_id
                WHERE p.product_code = %s AND dt.defect_code = %s
                ON CONFLICT DO NOTHING
            """, (img_file.name, str(img_file), product_code, defect_code))
    
    conn.commit()
    cur.close()
    conn.close()
    print("✅ 이미지 마이그레이션 완료")

if __name__ == "__main__":
    print("마이그레이션 시작...")
    migrate_products()
    migrate_images()
    print("✅ 전체 마이그레이션 완료")
```

---

## 9. API 연동 예시 (FastAPI)

### 9.1 데이터베이스 연결

```python
# database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = "postgresql://postgres:password@localhost/defect_detection_db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

### 9.2 모델 정의

```python
# models.py
from sqlalchemy import Column, Integer, String, Float, Boolean, TIMESTAMP, ForeignKey, Text
from sqlalchemy.dialects.postgresql import JSONB
from database import Base

class Product(Base):
    __tablename__ = "products"
    
    product_id = Column(Integer, primary_key=True)
    product_code = Column(String(50), unique=True, nullable=False)
    product_name = Column(String(100), nullable=False)
    description = Column(Text)
    is_active = Column(Boolean, default=True)
    created_at = Column(TIMESTAMP)
    updated_at = Column(TIMESTAMP)

class SearchHistory(Base):
    __tablename__ = "search_history"
    
    search_id = Column(Integer, primary_key=True)
    searched_at = Column(TIMESTAMP)
    uploaded_image_path = Column(String(500))
    product_code = Column(String(50))
    defect_code = Column(String(50))
    top_k_results = Column(JSONB)
    processing_time = Column(Float)

class ResponseHistory(Base):
    __tablename__ = "response_history"
    
    response_id = Column(Integer, primary_key=True)
    executed_at = Column(TIMESTAMP)
    search_id = Column(Integer, ForeignKey("search_history.search_id"))
    product_code = Column(String(50))
    defect_code = Column(String(50))
    similarity_score = Column(Float)
    anomaly_score = Column(Float)
    guide_content = Column(Text)
    feedback_rating = Column(Integer)
    feedback_text = Column(Text)
```

### 9.3 API 엔드포인트

```python
# api.py
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from database import get_db
from models import SearchHistory, ResponseHistory

router = APIRouter()

@router.post("/worker/search")
async def create_search(
    image_path: str,
    top_k_results: dict,
    db: Session = Depends(get_db)
):
    """유사도 검색 이력 저장"""
    search = SearchHistory(
        uploaded_image_path=image_path,
        product_code=top_k_results.get('product_code'),
        defect_code=top_k_results.get('defect_code'),
        top_k_results=top_k_results['results']
    )
    db.add(search)
    db.commit()
    db.refresh(search)
    return {"search_id": search.search_id}

@router.post("/worker/response")
async def create_response(
    search_id: int,
    guide_content: str,
    db: Session = Depends(get_db)
):
    """대응 매뉴얼 생성 이력 저장"""
    response = ResponseHistory(
        search_id=search_id,
        guide_content=guide_content
    )
    db.add(response)
    db.commit()
    return {"response_id": response.response_id}

@router.get("/admin/stats")
async def get_dashboard_stats(db: Session = Depends(get_db)):
    """대시보드 통계 조회"""
    stats = db.execute("SELECT * FROM v_dashboard_stats").fetchall()
    return stats
```

---

## 10. 유지보수

### 10.1 정기 작업

```sql
-- 일일 통계 수집
ANALYZE;

-- 주간 VACUUM
VACUUM ANALYZE;

-- 월간 재인덱스 (필요시)
REINDEX DATABASE defect_detection_db;
```

### 10.2 모니터링 쿼리

```sql
-- 테이블 크기 확인
SELECT
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- 느린 쿼리 확인
SELECT 
    query,
    calls,
    total_time,
    mean_time
FROM pg_stat_statements
ORDER BY mean_time DESC
LIMIT 10;
```

---

**작성일**: 2025-11-09  
**버전**: 2.0 (단순화)  
**DBMS**: PostgreSQL 14+  
**테이블 수**: 10개
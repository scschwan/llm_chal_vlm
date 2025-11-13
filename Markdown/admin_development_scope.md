# 관리자 페이지 개발 Scope 및 아키텍처

**작성일**: 2025-01-15  
**버전**: 1.0  
**대상**: 제조 불량 검출 시스템 관리자 페이지  

---

## 목차

1. [개발 개요](#1-개발-개요)
2. [시스템 아키텍처](#2-시스템-아키텍처)
3. [기능 상세 설계](#3-기능-상세-설계)
4. [화면 설계](#4-화면-설계)
5. [API 설계](#5-api-설계)
6. [데이터베이스 연동](#6-데이터베이스-연동)
7. [개발 우선순위](#7-개발-우선순위)
8. [기술 스택](#8-기술-스택)

---

## 1. 개발 개요

### 1.1 프로젝트 배경

**현재 상태**:
- ✅ 작업자 페이지 개발 완료 (4개 탭)
  - 이미지 업로드
  - 유사도 검색 (TOP-K)
  - 이상 검출 (PatchCore)
  - 대응 매뉴얼 생성 (RAG + LLM)
- ✅ 파일 기반 데이터 관리 (JSON, 로컬 파일)
- ✅ MariaDB 구축 완료 (dmillion 계정)

**개발 필요사항**:
- ❌ 관리자 페이지 미개발
- ❌ DB 연동 미완료
- ❌ 제품/매뉴얼/이미지 관리 기능 없음
- ❌ 통계/모니터링 대시보드 없음

### 1.2 개발 목표

1. **제품 라이프사이클 관리**: 제품 등록 → 이미지 수집 → 모델 배포 → 운영 모니터링
2. **데이터 중앙화**: 파일 기반 → DB 기반으로 전환
3. **운영 자동화**: 배포, 인덱싱, 전처리 설정 관리
4. **실시간 모니터링**: 검사 이력, 불량 통계, 피드백 분석

### 1.3 사용자 역할

| 역할 | 계정 | 권한 |
|------|------|------|
| **관리자 (Admin)** | `admin` | 모든 기능 접근, 제품/모델 관리, 배포 실행 |
| **작업자 (Worker)** | `worker` | 검사 실행, 피드백 입력 (관리 기능 제한) |

**인증 방식**: 
- `config/auth.yaml` 파일 기반
- Session 또는 JWT 토큰 관리
- 로그인 화면 별도 구현

---

## 2. 시스템 아키텍처

### 2.1 전체 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (HTML/JS)                       │
├──────────────────────┬──────────────────────────────────────────┤
│   작업자 페이지      │           관리자 페이지                  │
│   (worker/)          │           (admin/)                       │
│  ┌────────────────┐  │  ┌────────────────┐  ┌────────────────┐ │
│  │ 1. 업로드      │  │  │ 1. 대시보드    │  │ 5. 배포 관리   │ │
│  │ 2. 유사도 검색 │  │  │ 2. 제품 관리   │  │ 6. 이력 조회   │ │
│  │ 3. 이상 검출   │  │  │ 3. 매뉴얼 관리 │  │ 7. 통계 분석   │ │
│  │ 4. 매뉴얼 생성 │  │  │ 4. 이미지 관리 │  │                │ │
│  └────────────────┘  │  └────────────────┘  └────────────────┘ │
└──────────────────────┴──────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Backend (FastAPI + Python)                    │
├──────────────────────┬──────────────────────────────────────────┤
│   Worker API         │           Admin API                      │
│   (기존 4개)         │           (신규 개발)                    │
│  ┌────────────────┐  │  ┌────────────────┐  ┌────────────────┐ │
│  │ /upload/*      │  │  │ /admin/dash    │  │ /admin/deploy  │ │
│  │ /search/*      │  │  │ /admin/product │  │ /admin/history │ │
│  │ /anomaly/*     │  │  │ /admin/manual  │  │ /admin/stats   │ │
│  │ /manual/*      │  │  │ /admin/image   │  │                │ │
│  └────────────────┘  │  └────────────────┘  └────────────────┘ │
└──────────────────────┴──────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────┐
│                      Business Logic Layer                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Product      │  │ Image        │  │ Deployment   │          │
│  │ Manager      │  │ Manager      │  │ Manager      │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Manual       │  │ History      │  │ Statistics   │          │
│  │ Manager      │  │ Manager      │  │ Manager      │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────┐
│                          Data Layer                              │
├──────────────────────┬──────────────────────────────────────────┤
│   MariaDB            │   File Storage        │   AI Models      │
│  (10 Tables)         │   (OBS/Local)         │   (CLIP/Patch)   │
│  ┌────────────────┐  │  ┌────────────────┐   │  ┌────────────┐ │
│  │ products       │  │  │ Images         │   │  │ CLIP Index │ │
│  │ defect_types   │  │  │ Manuals (PDF)  │   │  │ PatchCore  │ │
│  │ images         │  │  │ Results        │   │  │ RAG Vector │ │
│  │ manuals        │  │  │                │   │  │            │ │
│  │ preprocessing  │  │  │                │   │  │            │ │
│  │ history...     │  │  │                │   │  │            │ │
│  └────────────────┘  │  └────────────────┘   │  └────────────┘ │
└──────────────────────┴──────────────────────────────────────────┘
```

### 2.2 디렉토리 구조

```
llm_chal_vlm/
├── web/
│   ├── api_server.py                    # 메인 서버 (기존)
│   ├── config/
│   │   └── auth.yaml                    # 사용자 인증 정보
│   ├── database/
│   │   ├── __init__.py
│   │   ├── connection.py                # MariaDB 연결
│   │   ├── models.py                    # SQLAlchemy 모델
│   │   └── crud.py                      # CRUD 함수
│   ├── routers/
│   │   ├── upload.py                    # ✅ 기존
│   │   ├── search.py                    # ✅ 기존
│   │   ├── anomaly.py                   # ✅ 기존
│   │   ├── manual.py                    # ✅ 기존
│   │   ├── auth.py                      # ❌ 신규: 로그인/세션
│   │   └── admin/                       # ❌ 신규: 관리자 API
│   │       ├── __init__.py
│   │       ├── dashboard.py             # 대시보드
│   │       ├── product.py               # 제품 관리
│   │       ├── manual.py                # 매뉴얼 관리
│   │       ├── image.py                 # 이미지 관리
│   │       ├── preprocessing.py         # 전처리 설정
│   │       ├── deployment.py            # 배포 관리
│   │       ├── history.py               # 이력 조회
│   │       └── statistics.py            # 통계 분석
│   ├── pages/
│   │   ├── login.html                   # ❌ 신규: 로그인
│   │   ├── upload.html                  # ✅ 기존
│   │   ├── search.html                  # ✅ 기존
│   │   ├── anomaly.html                 # ✅ 기존
│   │   ├── manual.html                  # ✅ 기존
│   │   └── admin/                       # ❌ 신규: 관리자 화면
│   │       ├── dashboard.html           # 대시보드
│   │       ├── product.html             # 제품 관리
│   │       ├── manual.html              # 매뉴얼 관리
│   │       ├── image.html               # 이미지 관리
│   │       ├── preprocessing.html       # 전처리 설정
│   │       ├── deployment.html          # 배포 관리
│   │       ├── history.html             # 이력 조회
│   │       └── statistics.html          # 통계 분석
│   └── static/
│       ├── js/
│       │   ├── common.js                # ✅ 기존
│       │   ├── auth.js                  # ❌ 신규: 인증
│       │   └── admin/                   # ❌ 신규: 관리자 JS
│       │       ├── dashboard.js
│       │       ├── product.js
│       │       └── ...
│       └── css/
│           ├── common.css               # ✅ 기존
│           └── admin.css                # ❌ 신규: 관리자 스타일
├── modules/
│   ├── similarity_matcher.py            # ✅ 기존
│   ├── anomaly_detector.py              # ✅ 기존
│   └── vlm/                             # ✅ 기존
└── data/                                # ✅ 기존
```

---

## 3. 기능 상세 설계

### 3.1 인증 및 권한 관리

#### 3.1.1 로그인 기능

**화면**: `/login.html`

**기능**:
- 사용자명/비밀번호 입력
- `config/auth.yaml` 파일 기반 인증
- 로그인 성공 시 세션 생성
- 역할(admin/worker)에 따라 리다이렉트

**API**:
```
POST /auth/login
Request:
{
  "username": "admin",
  "password": "password"
}

Response:
{
  "status": "success",
  "user_type": "admin",
  "token": "jwt_token_or_session_id",
  "redirect_url": "/admin/dashboard.html"
}
```

#### 3.1.2 세션 관리

**방식**: 
- Session 기반 (FastAPI Session Middleware)
- 또는 JWT 토큰 (선택)

**미들웨어**:
```python
# 모든 /admin/* 경로는 admin 권한 필요
# 모든 /worker/* 경로는 worker 이상 권한 필요
```

#### 3.1.3 로그아웃

**API**:
```
POST /auth/logout
```

---

### 3.2 대시보드 (Dashboard)

#### 화면 구성

```
┌─────────────────────────────────────────────────────────────┐
│ 관리자 대시보드                    [로그아웃]                │
├─────────────────────────────────────────────────────────────┤
│ [대시보드] [제품관리] [매뉴얼] [이미지] [배포] [이력] [통계] │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│ │등록 제품 │ │총 검사수 │ │불량 검출 │ │평균 평점 │       │
│ │   4개    │ │ 1,234건  │ │  156건   │ │  4.2/5   │       │
│ └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
│                                                              │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ 제품별 데이터 현황                                    │   │
│ │ ┌────────────────────────────────────────────────┐  │   │
│ │ │제품   │정상이미지│불량이미지│매뉴얼│배포상태  │  │   │
│ │ │prod1  │   30     │   45     │  ✓   │ 완료     │  │   │
│ │ │grid   │   25     │   38     │  ✓   │ 완료     │  │   │
│ │ │carpet │   28     │   42     │  ✓   │ 완료     │  │   │
│ │ │leather│   35     │   60     │  ✓   │ 완료     │  │   │
│ │ └────────────────────────────────────────────────┘  │   │
│ └──────────────────────────────────────────────────────┘   │
│                                                              │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ 최근 검사 이력 (10건)                                 │   │
│ │ ┌────────────────────────────────────────────────┐  │   │
│ │ │일시        │제품  │불량   │점수  │조치상태    │  │   │
│ │ │01-15 14:35│prod1 │hole   │0.95  │● 조치완료 │  │   │
│ │ │01-15 14:20│grid  │normal │0.12  │-          │  │   │
│ │ │01-15 14:10│carpet│scratch│0.88  │○ 미조치   │  │   │
│ │ └────────────────────────────────────────────────┘  │   │
│ └──────────────────────────────────────────────────────┘   │
│                                                              │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ 배포 상태                                             │   │
│ │ • CLIP 인덱스: ✓ 최신 (01-15 10:00)                  │   │
│ │ • PatchCore: ✓ 최신 (01-15 10:30)                    │   │
│ │ • RAG 벡터DB: ✓ 최신 (01-15 09:00)                   │   │
│ └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

#### 기능

1. **요약 통계**:
   - 등록 제품 수
   - 총 검사 수
   - 불량 검출 수
   - 평균 사용자 평점

2. **제품별 현황**:
   - 제품명, 정상/불량 이미지 수
   - 매뉴얼 등록 여부
   - 배포 상태

3. **최근 검사 이력**:
   - 최근 10건 표시
   - 조치 상태 표시
   - 클릭 시 상세 보기

4. **배포 상태**:
   - CLIP, PatchCore, RAG 최신 배포 시각

#### API

```
GET /admin/dashboard/summary
Response:
{
  "total_products": 4,
  "total_inspections": 1234,
  "total_defects": 156,
  "avg_rating": 4.2
}

GET /admin/dashboard/products
Response:
{
  "products": [
    {
      "product_code": "prod1",
      "normal_images": 30,
      "defect_images": 45,
      "has_manual": true,
      "deploy_status": "completed"
    }
  ]
}

GET /admin/dashboard/recent_inspections?limit=10
Response:
{
  "inspections": [...]
}

GET /admin/dashboard/deployment_status
Response:
{
  "clip": {"status": "completed", "updated_at": "2025-01-15 10:00"},
  "patchcore": {"status": "completed", "updated_at": "2025-01-15 10:30"},
  "rag": {"status": "completed", "updated_at": "2025-01-15 09:00"}
}
```

---

### 3.3 제품 관리 (Product Management)

#### 화면 구성

```
┌─────────────────────────────────────────────────────────────┐
│ 제품 관리                                                    │
├─────────────────────────────────────────────────────────────┤
│ [+ 신규 제품 등록]                                           │
│                                                              │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ 제품: prod1 - 주조 제품 A형          [수정] [삭제]   │   │
│ │ 설명: 주조 공정 제품                                  │   │
│ │ 등록일: 2025-01-10 / 활성: ● ON                      │   │
│ │                                                        │   │
│ │ 불량 유형 (4개):                                      │   │
│ │ ┌────────────────────────────────────────────────┐  │   │
│ │ │ ID │영문    │한글  │전체명칭      │[수정][삭제]│  │   │
│ │ │ 1  │normal  │정상  │정상 제품     │            │  │   │
│ │ │ 2  │hole    │기공  │주조 기공     │ [수정][삭제]│  │   │
│ │ │ 3  │burr    │버    │날카로운 돌기 │ [수정][삭제]│  │   │
│ │ │ 4  │scratch │긁힘  │표면 긁힘     │ [수정][삭제]│  │   │
│ │ └────────────────────────────────────────────────┘  │   │
│ │ [+ 불량 유형 추가]                                    │   │
│ └──────────────────────────────────────────────────────┘   │
│                                                              │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ 제품: grid - 그리드 제품             [수정] [삭제]   │   │
│ │ ...                                                   │   │
│ └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

#### 기능

1. **제품 CRUD**:
   - 제품 등록: 제품코드, 제품명, 설명
   - 제품 수정
   - 제품 삭제 (soft delete)
   - 활성/비활성 토글

2. **불량 유형 관리**:
   - 제품별 불량 유형 추가
   - 불량 유형 수정 (영문, 한글, 전체명칭)
   - 불량 유형 삭제
   - 정상(normal) 유형은 자동 생성

3. **유효성 검사**:
   - 제품코드 중복 체크
   - 불량코드 중복 체크 (제품 내)

#### API

```
# 제품 목록
GET /admin/product/list
Response:
{
  "products": [
    {
      "product_id": 1,
      "product_code": "prod1",
      "product_name": "주조 제품 A형",
      "description": "...",
      "is_active": true,
      "defect_count": 4,
      "created_at": "2025-01-10"
    }
  ]
}

# 제품 등록
POST /admin/product
Request:
{
  "product_code": "prod5",
  "product_name": "신규 제품",
  "description": "설명"
}

# 제품 수정
PUT /admin/product/{product_id}
Request:
{
  "product_name": "수정된 제품명",
  "description": "수정된 설명",
  "is_active": true
}

# 제품 삭제
DELETE /admin/product/{product_id}

# 불량 유형 목록
GET /admin/product/{product_id}/defects

# 불량 유형 추가
POST /admin/product/{product_id}/defect
Request:
{
  "defect_code": "crack",
  "defect_name_ko": "균열",
  "defect_name_en": "crack",
  "full_name_ko": "표면 균열"
}

# 불량 유형 수정
PUT /admin/defect/{defect_type_id}

# 불량 유형 삭제
DELETE /admin/defect/{defect_type_id}
```

---

### 3.4 매뉴얼 관리 (Manual Management)

#### 화면 구성

```
┌─────────────────────────────────────────────────────────────┐
│ 매뉴얼 관리                                                  │
├─────────────────────────────────────────────────────────────┤
│ 제품 선택: [prod1 ▼]                                        │
│                                                              │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ 등록된 매뉴얼                                         │   │
│ │ ┌────────────────────────────────────────────────┐  │   │
│ │ │파일명            │크기  │인덱싱│일시    │     │  │   │
│ │ │prod1_menual.pdf  │2.3MB │✓완료 │01-10   │[삭제]│  │   │
│ │ └────────────────────────────────────────────────┘  │   │
│ │                                                        │   │
│ │ [+ 매뉴얼 업로드]                                      │   │
│ │                                                        │   │
│ │ ┌────────────────────────────────────┐              │   │
│ │ │ 파일 선택: [파일 선택]              │              │   │
│ │ │ 지원 형식: PDF                      │              │   │
│ │ │ 최대 크기: 50MB                     │              │   │
│ │ │                                     │              │   │
│ │ │ [업로드 및 자동 인덱싱]             │              │   │
│ │ └────────────────────────────────────┘              │   │
│ └──────────────────────────────────────────────────────┘   │
│                                                              │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ RAG 벡터 DB 상태                                      │   │
│ │ • 인덱스 경로: /manual_store/unified_index            │   │
│ │ • 총 문서 수: 4개                                     │   │
│ │ • 총 청크 수: 120개                                   │   │
│ │ • 마지막 갱신: 2025-01-15 09:00                       │   │
│ │                                                        │   │
│ │ [전체 재인덱싱]                                        │   │
│ └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

#### 기능

1. **매뉴얼 업로드**:
   - PDF 파일 업로드
   - 자동 RAG 인덱싱
   - `manuals` 테이블에 메타데이터 저장

2. **매뉴얼 삭제**:
   - 파일 삭제
   - DB 레코드 삭제
   - RAG 인덱스 갱신

3. **인덱스 관리**:
   - 전체 재인덱싱 버튼
   - 인덱싱 진행 상태 표시
   - 인덱스 상태 조회

#### API

```
# 제품별 매뉴얼 목록
GET /admin/manual/list/{product_id}
Response:
{
  "manuals": [
    {
      "manual_id": 1,
      "file_name": "prod1_menual.pdf",
      "file_size": 2411724,
      "vector_indexed": true,
      "indexed_at": "2025-01-10 12:00"
    }
  ]
}

# 매뉴얼 업로드
POST /admin/manual/upload
Request: multipart/form-data
- product_id: 1
- file: (binary)

Response:
{
  "status": "success",
  "manual_id": 5,
  "indexing_started": true
}

# 매뉴얼 삭제
DELETE /admin/manual/{manual_id}

# RAG 인덱스 상태
GET /admin/manual/rag/status
Response:
{
  "index_path": "/manual_store/unified_index",
  "total_documents": 4,
  "total_chunks": 120,
  "last_updated": "2025-01-15 09:00"
}

# RAG 전체 재인덱싱
POST /admin/manual/rag/rebuild
Response:
{
  "status": "started",
  "deploy_id": 123
}
```

---

### 3.5 이미지 데이터 관리 (Image Management)

#### 화면 구성

```
┌─────────────────────────────────────────────────────────────┐
│ 이미지 데이터 관리                                           │
├─────────────────────────────────────────────────────────────┤
│ 제품 선택: [prod1 ▼]                                        │
│                                                              │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ 정상 이미지 (30장)                                    │   │
│ │ [파일 선택] [ZIP 업로드]  [업로드]                    │   │
│ │                                                        │   │
│ │ 미리보기:                                              │   │
│ │ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ... [더보기]               │   │
│ │ │img│ │img│ │img│ │img│                             │   │
│ │ └───┘ └───┘ └───┘ └───┘                             │   │
│ └──────────────────────────────────────────────────────┘   │
│                                                              │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ 불량 이미지                                            │   │
│ │                                                        │   │
│ │ • hole (15장)    [파일 선택] [ZIP] [업로드] [삭제]   │   │
│ │   ┌───┐ ┌───┐ ┌───┐ ...                             │   │
│ │                                                        │   │
│ │ • burr (20장)    [파일 선택] [ZIP] [업로드] [삭제]   │   │
│ │   ┌───┐ ┌───┐ ┌───┐ ...                             │   │
│ │                                                        │   │
│ │ • scratch (10장) [파일 선택] [ZIP] [업로드] [삭제]   │   │
│ │   ┌───┐ ┌───┐ ┌───┐ ...                             │   │
│ └──────────────────────────────────────────────────────┘   │
│                                                              │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ 이미지 전처리 설정                                     │   │
│ │ ☐ Grayscale (그레이스케일 변환)                       │   │
│ │ ☐ Histogram (히스토그램 평활화)                       │   │
│ │ ☑ Contrast (명암 대비 조정)                           │   │
│ │ ☐ Smoothing (스무딩 처리)                             │   │
│ │ ☐ Normalize (정규화)                                  │   │
│ │                                                        │   │
│ │ [설정 저장]                                            │   │
│ └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

#### 기능

1. **정상 이미지 관리**:
   - 단일 파일 업로드
   - ZIP 파일 일괄 업로드
   - 썸네일 미리보기
   - 이미지 삭제

2. **불량 이미지 관리**:
   - 불량 유형별 관리
   - 파일명 자동 생성: `{product}_{defect}_{seq}.jpg`
   - ZIP 파일 업로드 시 자동 분류
   - 이미지 삭제

3. **전처리 설정**:
   - 5가지 전처리 on/off 설정
   - 제품별로 독립 설정
   - `image_preprocessing` 테이블 저장

4. **DB 연동**:
   - `images` 테이블에 메타데이터 저장
   - 파일은 로컬 또는 OBS에 저장

#### API

```
# 제품별 이미지 통계
GET /admin/image/stats/{product_id}
Response:
{
  "normal_count": 30,
  "defect_counts": {
    "hole": 15,
    "burr": 20,
    "scratch": 10
  }
}

# 이미지 목록
GET /admin/image/list/{product_id}?image_type=normal&defect_code=hole
Response:
{
  "images": [
    {
      "image_id": 1,
      "file_name": "prod1_ok_0_129.jpeg",
      "file_path": "/data/patchCore/prod1/prod1_ok_0_129.jpeg",
      "file_size": 123456,
      "uploaded_at": "2025-01-10"
    }
  ]
}

# 이미지 업로드 (단일)
POST /admin/image/upload
Request: multipart/form-data
- product_id: 1
- image_type: normal | defect
- defect_type_id: (defect인 경우 필수)
- file: (binary)

# 이미지 업로드 (ZIP)
POST /admin/image/upload_zip
Request: multipart/form-data
- product_id: 1
- image_type: normal | defect
- defect_type_id: (defect인 경우)
- file: (binary ZIP)

Response:
{
  "status": "success",
  "uploaded_count": 25,
  "failed_count": 0
}

# 이미지 삭제
DELETE /admin/image/{image_id}

# 전처리 설정 조회
GET /admin/preprocessing/{product_id}
Response:
{
  "preprocessing_id": 1,
  "grayscale": "N",
  "histogram": "N",
  "contrast": "Y",
  "smoothing": "N",
  "normalize": "N"
}

# 전처리 설정 저장
PUT /admin/preprocessing/{product_id}
Request:
{
  "grayscale": "N",
  "histogram": "Y",
  "contrast": "Y",
  "smoothing": "N",
  "normalize": "N"
}
```

---

### 3.6 배포 관리 (Deployment Management)

#### 화면 구성

```
┌─────────────────────────────────────────────────────────────┐
│ 서버 배포 관리                                               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ CLIP 인덱스 재구축                                    │   │
│ │ 상태: ✓ 완료 (2025-01-15 10:00)                      │   │
│ │ 인덱스 크기: 2,083개 이미지 / 512차원                │   │
│ │                                                        │   │
│ │ 제품 선택: [전체 ▼]                                   │   │
│ │ [재구축 시작]                                          │   │
│ │                                                        │   │
│ │ 진행률: ████████████████████ 100%                     │   │
│ └──────────────────────────────────────────────────────┘   │
│                                                              │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ PatchCore 메모리뱅크 생성                             │   │
│ │ 상태: ○ 대기중                                        │   │
│ │                                                        │   │
│ │ • prod1:   ✓ 완료 (614 패치)  [재생성]               │   │
│ │ • grid:    ✓ 완료 (520 패치)  [재생성]               │   │
│ │ • carpet:  ✓ 완료 (580 패치)  [재생성]               │   │
│ │ • leather: ✓ 완료 (720 패치)  [재생성]               │   │
│ │                                                        │   │
│ │ 제품 선택: [prod1 ▼]                                  │   │
│ │ [배포 시작]                                            │   │
│ └──────────────────────────────────────────────────────┘   │
│                                                              │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ RAG 벡터 DB 재인덱싱                                  │   │
│ │ 상태: ✓ 완료 (2025-01-15 09:00)                      │   │
│ │ 총 문서: 4개 / 총 청크: 120개                        │   │
│ │                                                        │   │
│ │ [재인덱싱 시작]                                        │   │
│ └──────────────────────────────────────────────────────┘   │
│                                                              │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ 배포 이력                                              │   │
│ │ ┌────────────────────────────────────────────────┐  │   │
│ │ │일시        │타입       │제품  │상태  │소요시간 │  │   │
│ │ │01-15 10:00│CLIP재구축 │전체  │완료  │60.5초   │  │   │
│ │ │01-15 09:00│RAG인덱싱  │전체  │완료  │8.5초    │  │   │
│ │ │01-14 16:30│PatchCore  │prod1 │완료  │120.3초  │  │   │
│ │ └────────────────────────────────────────────────┘  │   │
│ └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

#### 기능

1. **CLIP 인덱스 재구축**:
   - 전체 또는 제품별 선택
   - 백그라운드 작업 실행
   - 진행률 표시 (WebSocket 또는 폴링)

2. **PatchCore 메모리뱅크 생성**:
   - 제품별 개별 생성
   - 정상 이미지 기반

3. **RAG 벡터 DB 재인덱싱**:
   - 전체 매뉴얼 재인덱싱
   - 통합 벡터 DB 생성

4. **배포 이력 조회**:
   - 모든 배포 작업 이력
   - 상태, 소요시간 표시

5. **비동기 처리**:
   - Celery 또는 BackgroundTasks 사용
   - `deployment_logs` 테이블에 기록

#### API

```
# CLIP 재구축 시작
POST /admin/deploy/clip/rebuild
Request:
{
  "product_id": 1  # null이면 전체
}
Response:
{
  "status": "started",
  "deploy_id": 123
}

# PatchCore 생성 시작
POST /admin/deploy/patchcore/create
Request:
{
  "product_id": 1
}
Response:
{
  "status": "started",
  "deploy_id": 124
}

# RAG 재인덱싱 시작
POST /admin/deploy/rag/rebuild
Response:
{
  "status": "started",
  "deploy_id": 125
}

# 배포 상태 조회
GET /admin/deploy/status/{deploy_id}
Response:
{
  "deploy_id": 123,
  "deploy_type": "clip_rebuild",
  "status": "running",
  "progress": 45,
  "started_at": "2025-01-15 10:00",
  "estimated_time": "15초 남음"
}

# 배포 이력 조회
GET /admin/deploy/history?limit=20
Response:
{
  "deployments": [...]
}
```

---

### 3.7 이력 조회 (History)

#### 화면 구성

```
┌─────────────────────────────────────────────────────────────┐
│ 검사 이력 조회                                               │
├─────────────────────────────────────────────────────────────┤
│ 필터:                                                        │
│ 제품: [전체 ▼] 불량: [전체 ▼] 기간: [최근 7일 ▼]           │
│ 조치상태: [전체 ▼]  [검색]                                  │
│                                                              │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ 검사 이력 (총 156건)                                  │   │
│ │ ┌────────────────────────────────────────────────┐  │   │
│ │ │No│일시     │제품 │불량  │유사도│이상도│조치│평점│  │   │
│ │ │15│01-15 14│prod1│hole  │0.95  │0.88  │완료│5  │  │   │
│ │ │14│01-15 13│grid │normal│0.12  │0.05  │-   │-  │  │   │
│ │ │13│01-15 12│carpet│scra │0.91  │0.82  │미조│-  │  │   │
│ │ └────────────────────────────────────────────────┘  │   │
│ │ [1] [2] [3] ... [16] (페이지네이션)                  │   │
│ └──────────────────────────────────────────────────────┘   │
│                                                              │
│ [선택 항목 클릭 시 상세 보기]                                │
│                                                              │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ 검사 ID: 15                                           │   │
│ │ 일시: 2025-01-15 14:35:22                            │   │
│ │ 제품: prod1 (주조 제품 A형)                           │   │
│ │ 불량 유형: hole (주조 기공)                           │   │
│ │                                                        │   │
│ │ 점수:                                                  │   │
│ │ • 유사도: 0.9523                                      │   │
│ │ • 이상 점수: 0.8756                                   │   │
│ │                                                        │   │
│ │ 이미지:                                                │   │
│ │ [검사이미지] [기준이미지] [오버레이] [히트맵]         │   │
│ │                                                        │   │
│ │ LLM 가이드:                                            │   │
│ │ ┌────────────────────────────────────────────────┐  │   │
│ │ │【현상】제품 표면에 직경 약 2mm 크기의...       │  │   │
│ │ │【원인】주조 시 금속 내부의 가스가...           │  │   │
│ │ │【조치】주조 온도를 10~15°C 낮춰...            │  │   │
│ │ └────────────────────────────────────────────────┘  │   │
│ │                                                        │   │
│ │ 피드백:                                                │   │
│ │ • 평점: ★★★★★ (5점)                                │   │
│ │ • 내용: 정확한 원인 분석 및 조치 방법 제시            │   │
│ │ • 작성일: 2025-01-15 15:00                            │   │
│ │                                                        │   │
│ │ [닫기]                                                 │   │
│ └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

#### 기능

1. **검색 필터**:
   - 제품, 불량 유형, 기간, 조치 상태

2. **목록 표시**:
   - 페이지네이션 (10건씩)
   - 정렬 (최신순)

3. **상세 보기**:
   - 모든 검사 정보
   - 이미지 보기
   - LLM 가이드 내용
   - 피드백 내용

4. **데이터 출력**:
   - Excel 내보내기
   - CSV 내보내기

#### API

```
# 이력 검색
GET /admin/history/search?product_code=prod1&defect_code=hole&start_date=2025-01-01&end_date=2025-01-15&status=completed&page=1&limit=10
Response:
{
  "total": 156,
  "page": 1,
  "limit": 10,
  "items": [
    {
      "response_id": 15,
      "executed_at": "2025-01-15 14:35:22",
      "product_code": "prod1",
      "defect_code": "hole",
      "similarity_score": 0.9523,
      "anomaly_score": 0.8756,
      "feedback_rating": 5,
      "has_feedback": true
    }
  ]
}

# 이력 상세
GET /admin/history/{response_id}
Response:
{
  "response_id": 15,
  "executed_at": "2025-01-15 14:35:22",
  "product_code": "prod1",
  "product_name": "주조 제품 A형",
  "defect_code": "hole",
  "defect_name_ko": "주조 기공",
  "similarity_score": 0.9523,
  "anomaly_score": 0.8756,
  "test_image_path": "/uploads/test_15.jpg",
  "reference_image_path": "/data/patchCore/prod1/normal_001.jpg",
  "heatmap_path": "/anomaly_outputs/test_15/overlay.png",
  "guide_content": "【현상】...",
  "model_type": "hyperclovax",
  "feedback_rating": 5,
  "feedback_text": "정확한 원인 분석...",
  "feedback_at": "2025-01-15 15:00"
}

# 데이터 내보내기
GET /admin/history/export?format=xlsx&filters=...
Response: (파일 다운로드)
```

---

### 3.8 통계 분석 (Statistics)

#### 화면 구성

```
┌─────────────────────────────────────────────────────────────┐
│ 통계 분석                                                    │
├─────────────────────────────────────────────────────────────┤
│ 기간: [최근 30일 ▼]  [조회]                                │
│                                                              │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ 검사 추이 (일별)                                      │   │
│ │      검사수                                            │   │
│ │  60 ┤        ●                                         │   │
│ │  50 ┤      ●   ●                                       │   │
│ │  40 ┤    ●       ●                                     │   │
│ │  30 ┤  ●           ●                                   │   │
│ │  20 ┤●               ●                                 │   │
│ │  10 ┤                 ●                                │   │
│ │   0 └────────────────────────────                     │   │
│ │     01/01  01/08  01/15  01/22  01/30                 │   │
│ └──────────────────────────────────────────────────────┘   │
│                                                              │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ 제품별 불량 검출 통계                                  │   │
│ │    제품   │총검사│불량검출│검출률│TOP 불량           │   │
│ │   prod1   │  320 │   45   │14.1% │hole (20)         │   │
│ │   grid    │  280 │   38   │13.6% │burr (18)         │   │
│ │   carpet  │  310 │   42   │13.5% │scratch (20)      │   │
│ │   leather │  324 │   31   │ 9.6% │fold (12)         │   │
│ └──────────────────────────────────────────────────────┘   │
│                                                              │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ 불량 유형별 분포 (전체)                                │   │
│ │   hole:    ███████████████████ 32% (50건)            │   │
│ │   burr:    ████████████████ 28% (44건)               │   │
│ │   scratch: ████████████ 24% (37건)                   │   │
│ │   fold:    ████ 10% (16건)                           │   │
│ │   stain:   ██ 4% (6건)                               │   │
│ │   color:   █ 2% (3건)                                │   │
│ └──────────────────────────────────────────────────────┘   │
│                                                              │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ 사용자 만족도                                          │   │
│ │ 평균 평점: 4.2 / 5.0                                  │   │
│ │ 피드백 등록률: 68.2% (106/156)                        │   │
│ │                                                        │   │
│ │ 평점 분포:                                             │   │
│ │   5점: ████████████████████████ 45건 (42%)           │   │
│ │   4점: ██████████████████ 35건 (33%)                 │   │
│ │   3점: ████████ 18건 (17%)                           │   │
│ │   2점: ████ 6건 (6%)                                 │   │
│ │   1점: ██ 2건 (2%)                                   │   │
│ └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

#### 기능

1. **검사 추이 분석**:
   - 일별/주별/월별 검사 수 그래프
   - Chart.js 또는 Plotly 사용

2. **제품별 통계**:
   - 총 검사 수
   - 불량 검출 수/비율
   - TOP 불량 유형

3. **불량 유형별 분포**:
   - 전체 불량 중 유형별 비율
   - 바 차트

4. **사용자 만족도**:
   - 평균 평점
   - 피드백 등록률
   - 평점별 분포

5. **데이터 출력**:
   - 리포트 PDF 생성
   - 차트 이미지 다운로드

#### API

```
# 검사 추이
GET /admin/stats/inspection_trend?start_date=2025-01-01&end_date=2025-01-30&interval=day
Response:
{
  "data": [
    {"date": "2025-01-01", "count": 25},
    {"date": "2025-01-02", "count": 32},
    ...
  ]
}

# 제품별 통계
GET /admin/stats/by_product?start_date=2025-01-01&end_date=2025-01-30
Response:
{
  "products": [
    {
      "product_code": "prod1",
      "total_inspections": 320,
      "defect_count": 45,
      "defect_rate": 0.141,
      "top_defect": {"code": "hole", "count": 20}
    }
  ]
}

# 불량 유형별 분포
GET /admin/stats/defect_distribution?start_date=2025-01-01&end_date=2025-01-30
Response:
{
  "defects": [
    {"code": "hole", "count": 50, "percentage": 32},
    {"code": "burr", "count": 44, "percentage": 28},
    ...
  ]
}

# 사용자 만족도
GET /admin/stats/user_satisfaction?start_date=2025-01-01&end_date=2025-01-30
Response:
{
  "avg_rating": 4.2,
  "total_responses": 156,
  "feedback_count": 106,
  "feedback_rate": 0.682,
  "rating_distribution": {
    "5": 45,
    "4": 35,
    "3": 18,
    "2": 6,
    "1": 2
  }
}
```

---

## 4. 화면 설계

### 4.1 공통 레이아웃

**헤더**:
```
┌─────────────────────────────────────────────────────────────┐
│ 제조 불량 검출 시스템 - 관리자        사용자: admin  [로그아웃]│
└─────────────────────────────────────────────────────────────┘
```

**네비게이션 (탭)**:
```
┌─────────────────────────────────────────────────────────────┐
│ [대시보드] [제품관리] [매뉴얼] [이미지] [배포] [이력] [통계] │
└─────────────────────────────────────────────────────────────┘
```

**푸터**:
```
┌─────────────────────────────────────────────────────────────┐
│ © 2025 디밀리언. All rights reserved.                        │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 반응형 디자인

- **데스크톱**: 1920x1080 기준
- **태블릿**: 1024x768 지원
- **모바일**: 비지원 (관리 기능 특성상)

### 4.3 UI 컴포넌트

- **버튼**: Primary, Secondary, Danger
- **테이블**: 정렬, 페이지네이션
- **모달**: 확인, 경고, 입력
- **프로그레스바**: 배포 진행 상태
- **차트**: Line, Bar, Pie (Chart.js)

---

## 5. API 설계

### 5.1 인증 API

```
POST   /auth/login          # 로그인
POST   /auth/logout         # 로그아웃
GET    /auth/session        # 세션 확인
```

### 5.2 관리자 API

```
# 대시보드
GET    /admin/dashboard/summary
GET    /admin/dashboard/products
GET    /admin/dashboard/recent_inspections
GET    /admin/dashboard/deployment_status

# 제품 관리
GET    /admin/product/list
POST   /admin/product
GET    /admin/product/{product_id}
PUT    /admin/product/{product_id}
DELETE /admin/product/{product_id}
GET    /admin/product/{product_id}/defects
POST   /admin/product/{product_id}/defect
PUT    /admin/defect/{defect_type_id}
DELETE /admin/defect/{defect_type_id}

# 매뉴얼 관리
GET    /admin/manual/list/{product_id}
POST   /admin/manual/upload
DELETE /admin/manual/{manual_id}
GET    /admin/manual/rag/status
POST   /admin/manual/rag/rebuild

# 이미지 관리
GET    /admin/image/stats/{product_id}
GET    /admin/image/list/{product_id}
POST   /admin/image/upload
POST   /admin/image/upload_zip
DELETE /admin/image/{image_id}
GET    /admin/preprocessing/{product_id}
PUT    /admin/preprocessing/{product_id}

# 배포 관리
POST   /admin/deploy/clip/rebuild
POST   /admin/deploy/patchcore/create
POST   /admin/deploy/rag/rebuild
GET    /admin/deploy/status/{deploy_id}
GET    /admin/deploy/history

# 이력 조회
GET    /admin/history/search
GET    /admin/history/{response_id}
GET    /admin/history/export

# 통계 분석
GET    /admin/stats/inspection_trend
GET    /admin/stats/by_product
GET    /admin/stats/defect_distribution
GET    /admin/stats/user_satisfaction
```

---

## 6. 데이터베이스 연동

### 6.1 연결 설정

```python
# database/connection.py
import pymysql
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "mysql+pymysql://dmillion:password@localhost:3306/defect_detection_db?charset=utf8mb4"

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=False
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

### 6.2 모델 정의

```python
# database/models.py
from sqlalchemy import Column, Integer, String, Text, Float, TIMESTAMP, JSON
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Product(Base):
    __tablename__ = "products"
    
    product_id = Column(Integer, primary_key=True, autoincrement=True)
    product_code = Column(String(50), nullable=False)
    product_name = Column(String(100), nullable=False)
    description = Column(Text)
    is_active = Column(Integer, default=1)
    created_at = Column(TIMESTAMP)
    updated_at = Column(TIMESTAMP)

class DefectType(Base):
    __tablename__ = "defect_types"
    
    defect_type_id = Column(Integer, primary_key=True, autoincrement=True)
    product_id = Column(Integer, nullable=False)
    defect_code = Column(String(50), nullable=False)
    defect_name_ko = Column(String(100), nullable=False)
    defect_name_en = Column(String(100))
    full_name_ko = Column(String(200))
    is_active = Column(Integer, default=1)
    created_at = Column(TIMESTAMP)

# ... 나머지 테이블들
```

### 6.3 CRUD 함수

```python
# database/crud.py
from sqlalchemy.orm import Session
from . import models

def get_products(db: Session, is_active: bool = True):
    query = db.query(models.Product)
    if is_active:
        query = query.filter(models.Product.is_active == 1)
    return query.all()

def create_product(db: Session, product_data: dict):
    product = models.Product(**product_data)
    db.add(product)
    db.commit()
    db.refresh(product)
    return product

def update_product(db: Session, product_id: int, product_data: dict):
    product = db.query(models.Product).filter(models.Product.product_id == product_id).first()
    if product:
        for key, value in product_data.items():
            setattr(product, key, value)
        db.commit()
        db.refresh(product)
    return product

def delete_product(db: Session, product_id: int):
    product = db.query(models.Product).filter(models.Product.product_id == product_id).first()
    if product:
        product.is_active = 0  # Soft delete
        db.commit()
    return product

# ... 나머지 CRUD 함수들
```

---

## 7. 개발 우선순위

### Phase 1: 기본 구조 (1주)
1. ✅ 로그인/인증 시스템
2. ✅ 대시보드 (요약 통계만)
3. ✅ 제품 관리 (CRUD)
4. ✅ DB 연동 기본 구조

### Phase 2: 데이터 관리 (2주)
1. ✅ 매뉴얼 관리
2. ✅ 이미지 관리 (ZIP 업로드 포함)
3. ✅ 전처리 설정

### Phase 3: 배포 및 모니터링 (2주)
1. ✅ 배포 관리 (비동기 처리)
2. ✅ 이력 조회
3. ✅ 통계 분석

### Phase 4: 연동 및 최적화 (1주)
1. ✅ 작업자 페이지 DB 연동
2. ✅ 성능 최적화
3. ✅ 테스트 및 버그 수정

---

## 8. 기술 스택

### 8.1 Backend

| 기술 | 버전 | 용도 |
|------|------|------|
| **Python** | 3.9 | 백엔드 언어 |
| **FastAPI** | 0.100+ | API 프레임워크 |
| **SQLAlchemy** | 2.0+ | ORM |
| **PyMySQL** | 1.1+ | MariaDB 드라이버 |
| **Pydantic** | 2.0+ | 데이터 검증 |
| **python-multipart** | 0.0.6+ | 파일 업로드 |
| **bcrypt** | 4.0+ | 비밀번호 해싱 |

### 8.2 Frontend

| 기술 | 버전 | 용도 |
|------|------|------|
| **HTML5** | - | 마크업 |
| **JavaScript (ES6)** | - | 클라이언트 로직 |
| **CSS3** | - | 스타일링 |
| **Chart.js** | 4.0+ | 차트 |
| **Fetch API** | - | AJAX 통신 |

### 8.3 Database

| 기술 | 버전 | 용도 |
|------|------|------|
| **MariaDB** | 10.5+ | 관계형 DB |

### 8.4 기타

| 기술 | 버전 | 용도 |
|------|------|------|
| **uvicorn** | 0.23+ | ASGI 서버 |
| **python-jose** | 3.3+ | JWT (선택) |

---

## 9. 다음 단계

1. **인증 시스템 구현** (auth.py, login.html)
2. **데이터베이스 모델 구축** (models.py, crud.py)
3. **대시보드 개발** (dashboard.html, dashboard.py)
4. **제품 관리 개발** (product.html, product.py)
5. **순차적으로 나머지 기능 개발**

---

**문서 버전**: 1.0  
**작성일**: 2025-01-15  
**다음 리뷰**: Phase 1 완료 후

# 유사이미지 검색 솔루션 프로젝트 현황 및 향후 계획

**최종 업데이트:** 2025-01-13  
**프로젝트:** 제조 불량 검출 AI 시스템  
**환경:** Naver Cloud Platform, Rocky Linux 8.10, Python 3.9, Tesla T4 GPU

---

## 📋 목차

1. [프로젝트 개요](#프로젝트-개요)
2. [현재 구현 완료 기능](#현재-구현-완료-기능)
3. [시스템 아키텍처](#시스템-아키텍처)
4. [주요 기술 스택](#주요-기술-스택)
5. [향후 개발 계획](#향후-개발-계획)
6. [시급 기능 개선 목록](#시급-기능-개선-목록)
7. [장기 개발 로드맵](#장기-개발-로드맵)

---

## 프로젝트 개요

### 목표
제조 현장에서 발생하는 불량을 AI 기술로 자동 검출하고, 대응 매뉴얼을 자동 생성하여 작업자의 신속한 조치를 지원하는 통합 시스템 구축

### 주요 기능
1. **CLIP 기반 유사도 검색**: 입력 불량 이미지와 유사한 기존 불량 사례 검색
2. **PatchCore Anomaly Detection**: 정상 이미지 대비 이상 영역 자동 검출
3. **LLM/VLM 대응 매뉴얼 생성**: HyperCLOVAX, EXAONE 3.5, LLaVA 모델을 활용한 자동 대응 방안 생성

---

## 🎯 완성된 전체 워크플로우
```
1. 이미지 업로드
   ↓
2. 유사도 매칭 (불량 이미지 인덱스 기반)
   ↓
3. 이상 검출 (정상 이미지 인덱스로 자동 전환)
   ↓
4. 대응 매뉴얼 생성
   - 3개 AI 모델 선택 (HyperCLOVAX, EXAONE, LLaVA)
   - RAG 기반 매뉴얼 검색
   - 4개 섹션 표준 출력
   - 작업자 조치 내역 입력
```

---

## 📂 주요 파일 구조

### 백엔드 (Python)
```
llm_chal_vlm/
├── modules/
│   ├── similarity_matcher.py      ✅ Phase 1 최적화 완료
│   ├── anomaly_detector.py
│   └── vlm/
│       ├── rag.py                 ✅ 통합 RAG 시스템
│       └── defect_mapper.py       ✅ 불량 정보 매핑
├── web/
│   ├── api_server.py              ✅ 메인 API 서버
│   ├── defect_mapping.json        ⚠️ 중요: 관리자 페이지 연동 필요
│   ├── routers/
│   │   ├── upload.py              ✅ 업로드 라우터
│   │   ├── search.py              ✅ 유사도 검색 라우터
│   │   ├── anomaly.py             ✅ 이상 검출 라우터
│   │   └── manual.py              ✅ 매뉴얼 생성 라우터
│   ├── pages/
│   │   ├── upload.html            ✅ 업로드 페이지
│   │   ├── search.html            ✅ 유사도 매칭 페이지
│   │   ├── anomaly.html           ✅ 이상 검출 페이지
│   │   └── manual.html            ✅ 대응 매뉴얼 페이지
│   └── static/
│       ├── js/
│       │   ├── common.js          ✅ 공통 유틸리티
│       │   ├── upload.js          ✅ 세션 관리 개선
│       │   ├── search.js
│       │   ├── anomaly.js
│       │   └── manual.js
│       └── css/
├── llm_server/
│   └── llm_server.py              ✅ Keepalive 타이머 추가
├── manual_store/                  ✅ 통합 매뉴얼 디렉토리
│   ├── prod1_menual.pdf
│   ├── grid_manual.pdf
│   ├── carpet_manual.pdf
│   ├── leather_manual.pdf
│   └── unified_index/             (자동 생성)
└── data/
    ├── def_split/                 (불량 이미지)
    ├── ok_split/                  (정상 이미지)
    └── patchCore/                 (제품별 메모리 뱅크)

## 현재 구현 완료 기능

### 1. 유사도 검색 (CLIP 기반)
- ✅ **모델**: ViT-B-32/openai
- ✅ **인덱스 관리**: FAISS 인덱스 자동 구축/저장/로드
- ✅ **TOP-K 검색**: 상위 K개 유사 이미지 검색 (K=1~20 설정 가능)
- ✅ **이미지 스왑**: 썸네일 클릭으로 TOP-1 변경 가능
- ✅ **불량 이미지 등록**: 신규 불량 이미지 자동 명명 규칙으로 등록
  - 형식: `{제품명}_{불량명}_{시퀀스번호}.jpg`
  - 등록 시 자동 인덱스 재구축

### 2. 이상 영역 검출 (PatchCore)
- ✅ **메모리 뱅크**: 제품별(prod1, prod2, prod3) 사전 학습된 메모리 뱅크
- ✅ **정상 참조 이미지**: 유사도 검색 TOP-1 또는 수동 지정 가능
- ✅ **이상 점수 계산**: Pixel-level + Image-level anomaly score
- ✅ **시각화**: 
  - 정상 기준 이미지
  - 이상 영역 마스크
  - 오버레이 이미지 (빨간색 표시)
  - 좌우 비교 이미지

### 3. LLM/VLM 대응 매뉴얼 생성
- ✅ **3개 모델 지원**:
  - **HyperCLOVAX-1.5B**: 텍스트 기반 분석
  - **EXAONE 3.5-2.4B**: Chat template 기반 분석
  - **LLaVA-1.5-7B**: 이미지 포함 VLM 분석
- ✅ **RAG 매뉴얼 검색**: PDF 매뉴얼 기반 LangChain + FAISS 벡터 DB
- ✅ **4개 섹션 표준 출력**:
  1. **불량 현황**: 판정 결과 요약
  2. **원인 분석**: 매뉴얼 원인 인용
  3. **대응 방안**: 즉시 조치사항
  4. **예방 조치**: 재발 방지 방안
- ✅ **프롬프트 정제**: 일관된 구조화된 프롬프트 생성
- ✅ **응답 슬라이싱**: 정규표현식 기반 4개 섹션 자동 추출

### 4. 웹 인터페이스
- ✅ **FastAPI 기반 API 서버** (포트 5000)
  - 유사도 검색 API
  - 이상 검출 API
  - 매뉴얼 생성 API (LLM/EXAONE/VLM)
- ✅ **LLM 전용 서버** (포트 5001)
  - 3개 모델 사전 로드
  - 독립적인 API 엔드포인트
- ✅ **탭 기반 UI**:
  - 유사도 검색 탭
  - 이상 영역 검출 탭
  - 대응 매뉴얼 생성 탭
- ✅ **인덱스 관리 UI**: 상태 확인 및 재구축 버튼

### 5. 인프라 (Naver Cloud Platform)
- ✅ **GPU 서버**: Tesla T4 GPU, Rocky Linux 8.10
- ✅ **네트워크**: VPC (10.200.0.0/16), ALB 로드밸런서
- ✅ **포트 구성**:
  - ALB: 80 → Backend 5000 (API 서버)
  - NLB: 2022 → SSH
  - LLM 서버: 5001 (내부 통신)

---

## 시스템 아키텍처
```
┌─────────────────────────────────────────────────────────────┐
│                        사용자 (웹 브라우저)                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ↓ HTTP (포트 80)
┌─────────────────────────────────────────────────────────────┐
│                      ALB (로드밸런서)                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ↓ 포트 5000
┌─────────────────────────────────────────────────────────────┐
│                   API 서버 (api_server.py)                    │
│  - 유사도 검색 (CLIP + FAISS)                                │
│  - 이상 검출 (PatchCore)                                      │
│  - RAG 매뉴얼 검색 (LangChain)                                │
│  - DefectMapper (불량 정보 매핑)                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ↓ 내부 HTTP (포트 5001)
┌─────────────────────────────────────────────────────────────┐
│                 LLM 서버 (llm_server.py)                      │
│  - HyperCLOVAX-1.5B (사전 로드)                               │
│  - EXAONE 3.5-2.4B (사전 로드)                                │
│  - LLaVA-1.5-7B (사전 로드)                                   │
│                                                               │
│  API 엔드포인트:                                              │
│  - /analyze (HyperCLOVAX)                                     │
│  - /analyze_exaone (EXAONE)                                   │
│  - /analyze_vlm (LLaVA)                                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 주요 기술 스택

### AI/ML 프레임워크
- **PyTorch**: 모델 추론
- **Transformers (v4.43+)**: HuggingFace 모델 로드
- **OpenCLIP**: CLIP 모델
- **PatchCore**: Anomaly Detection
- **LangChain**: RAG 파이프라인
- **FAISS**: 벡터 인덱스 (유사도 검색 + 매뉴얼 검색)

### 모델
| 모델 | 용도 | 크기 | Dtype |
|------|------|------|-------|
| ViT-B-32/openai | 유사도 검색 | 151M | FP16 |
| HyperCLOVAX-1.5B | LLM 텍스트 분석 | 1.5B | FP16 |
| EXAONE-3.5-2.4B | LLM 텍스트 분석 | 2.4B | BF16 |
| LLaVA-1.5-7B | VLM 이미지 분석 | 7B | FP16 |

### 웹 서버
- **FastAPI**: REST API 서버
- **Uvicorn**: ASGI 서버
- **HTML/CSS/JS**: 프론트엔드

### 데이터
- **이미지 데이터**: JPG/PNG (제품별/불량별 분류)
- **매뉴얼**: PDF (LangChain으로 임베딩)
- **메타데이터**: JSON (defect_mapping.json)

---

## 향후 개발 계획

### Phase 1: UI/UX 개선 (우선순위: 높음)

#### 1.1 이미지 업로드 화면 분리
- [ ] **독립 화면 구성**: 업로드 영역 크게 확장
- [ ] **인덱스 관리 유지**: 
  - 인덱스 상태 확인
  - 인덱스 재구축 버튼
- [ ] **파일 정보 표시**: 업로드된 이미지 메타데이터 (크기, 해상도)

#### 1.2 유사도 매칭 결과 화면 분리
- [ ] **독립 화면 구성**: TOP-K 결과를 전체 화면으로 표시
- [ ] **불량 등록 기능 유지**: 신규 불량 이미지 등록 모달
- [ ] **다음 단계 버튼**: "이상 영역 검출로 이동" 버튼 추가
- [ ] **LLM/VLM 버튼 제거**: 매뉴얼 생성은 이상 검출 이후에만 가능

#### 1.3 이상 영역 검출 화면 분리 및 자동화
- [ ] **독립 화면 구성**: 기존 탭 → 별도 페이지
- [ ] **자동 검출**: 화면 진입 시 자동으로 PatchCore 실행
- [ ] **제품/불량 정보 전달**: 
  - 유사도 매칭에서 추출한 TOP-1의 `product_name`, `defect_name` 전달
  - 방법: Global 변수 또는 URL 파라미터
- [ ] **시각화 간소화**:
  - ~~정상 기준 이미지~~ (삭제)
  - ~~마스크 이미지~~ (삭제)
  - ✅ 비교 결과 이미지만 표시 (정상 vs 이상 영역)
- [ ] **매뉴얼 생성 버튼 추가**: HyperCLOVAX, EXAONE, VLM 선택

#### 1.4 대응 매뉴얼 생성 화면 개선
- [ ] **3개 섹션 구성**:
  1. **이미지 비교 섹션**:
     - TOP-1 정상 이미지
     - Segmentation 적용된 이미지 (오버레이)
  2. **AI 생성 답변 섹션**:
     - 선택한 모델의 4개 섹션 답변
     - 처리 시간 표시
  3. **작업자 입력 섹션**:
     - 작업자명 (텍스트 입력)
     - 조치내역 (텍스트 영역)
     - 피드백 점수 (1~5점 라디오 버튼)
- [ ] **조치내역 등록 버튼**: DB 저장 트리거

### Phase 2: 데이터베이스 구축 (우선순위: 높음)

#### 2.1 조치내역 저장 스키마 설계
```sql
CREATE TABLE defect_analysis_history (
    id SERIAL PRIMARY KEY,
    search_id VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW(),
    product_name VARCHAR(50),
    defect_name VARCHAR(50),
    input_image_path VARCHAR(255),
    top1_image_path VARCHAR(255),
    model_used VARCHAR(20),  -- 'hyperclovax', 'exaone', 'llava'
    llm_response TEXT,
    processing_time FLOAT,
    has_feedback BOOLEAN DEFAULT FALSE,
    worker_name VARCHAR(100),
    action_taken TEXT,
    feedback_score INT CHECK (feedback_score BETWEEN 1 AND 5),
    anomaly_score FLOAT,
    is_anomaly BOOLEAN
);
```

#### 2.2 API 구현
- [ ] **POST /history/save**: 조치내역 저장
- [ ] **GET /history/list**: 이력 목록 조회 (페이징)
- [ ] **GET /history/{id}**: 특정 이력 상세 조회
- [ ] **GET /history/stats**: 통계 데이터 (모델별, 불량별)

#### 2.3 대시보드 연동
- [ ] **이력 테이블**: 저장된 데이터 표 형태로 출력
- [ ] **필터링**: 제품명, 불량명, 날짜, 모델, 피드백 점수
- [ ] **통계 차트**: 
  - 모델별 사용 빈도
  - 불량 유형별 발생 건수
  - 피드백 점수 분포

---

## 시급 기능 개선 목록

### ⚠️ 1. 인덱스 자동 전환 (최우선)

**현재 문제:**
- 유사도 매칭과 이상 검출에서 동일한 인덱스를 사용
- 불량 이미지와 정상 이미지가 혼재된 인덱스로 인해 정확도 저하

**해결 방안:**
```python
# api_server.py에 추가

# 유사도 검색 화면 진입 시
@app.on_event("startup")
async def ensure_defect_index():
    """불량 이미지 인덱스로 자동 전환"""
    defect_dir = project_root / "data" / "def_split"
    matcher.build_index(str(defect_dir))
    matcher.save_index(str(INDEX_DIR / "defect"))

# 이상 검출 화면 진입 시
@app.get("/switch_to_normal_index")
async def switch_to_normal_index():
    """정상 이미지 인덱스로 자동 전환"""
    normal_dir = project_root / "data" / "ok_split"
    matcher.build_index(str(normal_dir))
    matcher.save_index(str(INDEX_DIR / "normal"))
    return {"status": "success", "index_type": "normal"}
```

**구현 계획:**
- [ ] `def_split` 디렉토리: 불량 이미지 전용
- [ ] `ok_split` 디렉토리: 정상 이미지 전용
- [ ] 화면별 자동 인덱스 전환 로직
- [ ] 인덱스 타입 상태 표시 (UI)

### 2. 로봇 이동 거리 정보 (8번 항목)
- [ ] **데이터 수집**: 로봇 제어 시스템과 연동하여 이동 거리 획득
- [ ] **DB 저장**: `robot_distance` 컬럼 추가
- [ ] **UI 표시**: 대응 매뉴얼 화면에 로봇 이동 거리 정보 표시

### 3. SSIM 산출 근거 (7번 항목)
- [ ] **코드 분석**: `modules/ssim_utils.py` 파일 확인
- [ ] **증빙 자료 작성**: SSIM 계산 로직 문서화
- [ ] **시각화**: SSIM 히트맵 생성 및 저장

---

## 장기 개발 로드맵

### Phase 3: 관리자 기능 (우선순위: 중)

#### 3.1 이미지 전처리 설정
- [ ] **관리자 페이지**: 제품별 전처리 옵션 설정
  - 밝기/대비 조정
  - 노이즈 제거
  - 크기 정규화
- [ ] **자동 적용**: 설정 시 모든 이미지에 전처리 자동 적용
  - 임베딩 이미지 (갤러리)
  - 입력 이미지 (쿼리)
- [ ] **전처리 이력 관리**: 적용된 전처리 파라미터 DB 저장

#### 3.2 입력 이미지 이력 관리
- [ ] **자동 저장**: 업로드된 모든 입력 이미지 서버 저장
- [ ] **메타데이터**: 업로드 시간, 파일명, 크기, 해상도
- [ ] **이력 조회**: 과거 입력 이미지 검색 및 재분석
- [ ] **저장 경로**: `uploads/{날짜}/{파일명}`

### Phase 4: 성능 최적화 (우선순위: 중)

#### 4.1 모델 최적화
- [ ] **양자화**: 4-bit/8-bit 양자화로 메모리 사용량 감소
- [ ] **TensorRT**: GPU 추론 속도 향상
- [ ] **배치 처리**: 여러 이미지 동시 처리

#### 4.2 인덱스 최적화
- [ ] **증분 업데이트**: 전체 재구축 대신 추가 이미지만 임베딩
- [ ] **압축**: FAISS PQ (Product Quantization)
- [ ] **샤딩**: 제품별 인덱스 분리

### Phase 5: 확장 기능 (우선순위: 낮음)

#### 5.1 다중 카메라 뷰
- [ ] 여러 각도 이미지 동시 분석
- [ ] 3D 재구성

#### 5.2 실시간 모니터링
- [ ] 라인 카메라 연동
- [ ] 실시간 불량 알림
- [ ] 대시보드 실시간 업데이트

#### 5.3 자동 학습 파이프라인
- [ ] Active Learning
- [ ] 피드백 기반 모델 재학습
- [ ] A/B 테스트

---

## 파일 구조
```
llm_chal_vlm/
├── llm_server/
│   └── llm_server.py           # LLM/VLM 서버 (3개 모델 사전 로드)
├── web/
│   ├── api_server.py           # FastAPI 메인 서버
│   ├── matching.html           # 웹 UI
│   ├── static/
│   │   ├── css/matching.css
│   │   └── js/matching.js
│   └── defect_mapping.json     # 불량 정보 매핑
├── modules/
│   ├── similarity_matcher.py   # CLIP 유사도 검색
│   ├── anomaly_detector.py     # PatchCore 이상 검출
│   └── vlm/
│       ├── rag.py              # RAG 매뉴얼 검색
│       └── defect_mapper.py    # 불량 정보 매핑
├── data/
│   ├── def_split/              # 불량 이미지 (유사도 검색용)
│   ├── ok_split/               # 정상 이미지 (이상 검출용)
│   └── patchCore/              # PatchCore 메모리 뱅크
│       ├── prod1/
│       ├── prod2/
│       └── prod3/
├── manual_store/
│   ├── prod1_menual.pdf        # PDF 매뉴얼
│   └── faiss_index/            # 매뉴얼 벡터 DB
└── markdown/
    └── project_status.md       # 이 파일
```

---

## 개발 참고사항

### 환경변수
```bash
# llm_server.py
PORT=5001
VLM_MODEL=llava-hf/llava-1.5-7b-hf

# api_server.py
LLM_SERVER_URL=http://localhost:5001
```

### 서버 실행
```bash
# LLM 서버 (포트 5001)
cd llm_server
python llm_server.py

# API 서버 (포트 5000)
cd web
python api_server.py
```

### 인덱스 구축
```bash
# 불량 이미지 인덱스
curl -X POST http://localhost:5000/build_index \
  -H "Content-Type: application/json" \
  -d '{"gallery_dir": "../data/def_split", "save_index": true}'

# 정상 이미지 인덱스
curl -X POST http://localhost:5000/build_index \
  -H "Content-Type: application/json" \
  -d '{"gallery_dir": "../data/ok_split", "save_index": true}'
```

---

## 문의 및 지원

**개발자**: dhkim@dmillions.co.kr  
**프로젝트 저장소**: https://github.com/scschwan/llm_chal_vlm  
---

**버전 이력:**
- v2.0 (2025-01-13): HyperCLOVAX, EXAONE 3.5, LLaVA 3개 모델 통합, 4개 섹션 표준 출력
- v1.5 (2025-01-10): VLM 프롬프트 정제, 응답 슬라이싱 개선
- v1.0 (2025-01-05): 기본 기능 구현 완료
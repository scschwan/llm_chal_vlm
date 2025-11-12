# 유사이미지 검색 솔루션 - 세션 작업 기록
**작성일**: 2025-01-15  
**프로젝트**: 제조 불량 검출 AI 시스템 (Phase 1 최적화 + 통합 RAG 구축)

---

## 📋 이번 세션 작업 요약

### 1. Phase 1 성능 최적화 완료 ✅
- **배치 처리**: 이미지 32개씩 묶어서 처리
- **다중 프로세스**: DataLoader 4 workers 병렬 로딩
- **이미지 리사이즈**: CLIP 입력 크기(224x224)로 최적화

**성능 개선:**
- 기존: 2,083개 이미지 → 5-10분
- 개선: 2,083개 이미지 → 30-60초 (약 10배 향상)

### 2. 통합 RAG 시스템 구축 완료 ✅
- **메타데이터 기반 필터링**: 제품명별 매뉴얼 분리 검색
- **통합 벡터 DB**: 모든 PDF를 한번에 임베딩 후 제품별 필터링
- **자동 인덱스 관리**: 서버 시작 시 자동 구축 및 캐싱

**지원 제품:**
- prod1 (3개 불량)
- grid (3개 불량)
- carpet (4개 불량)
- leather (6개 불량)

### 3. DefectMapper 시스템 구축 ✅
- 제품별 불량 정보 매핑 관리
- `defect_mapping.json` 파일 기반
- 동적 제품/불량 추가 기능

### 4. 세션 관리 개선 ✅
- 탭 이동 시 세션 유지
- "다시 업로드" 버튼만 전체 초기화
- 서버 재시작 시 업로드 디렉토리 자동 정리

### 5. LLM 서버 안정성 개선 ✅
- Keepalive 타이머 추가 (60초마다 로그 출력)
- 모든 주요 이벤트에 타임스탬프 추가
- SSH 세션 끊김 방지

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
```

---

## ⚠️ 다음 세션에서 해결해야 할 이슈

### 1. RAG 검색 결과 필터링 개선 (최우선) 🔥

**현재 문제:**
```
[RAG] prod1 제품 매뉴얼 검색: 3개 결과
[MANUAL] RAG 검색 완료: 원인 0개, 조치 0개  ← 필터링 실패!
```

**원인 분석:**
1. PDF 검색은 정상 작동 (3개 결과 반환)
2. 검색된 텍스트에서 "원인"/"조치" 키워드 필터링 실패
3. 매뉴얼 텍스트 구조와 필터링 로직 불일치 가능성

**개선 방향:**
- [ ] PDF 매뉴얼 내부 구조 분석 필요
- [ ] 필터링 키워드 확장 (현재: ["원인", "발생", "이유", "때문"])
- [ ] 텍스트 전처리 개선 (공백, 줄바꿈 처리)
- [ ] 섹션 헤더 기반 분류 로직 추가
- [ ] 필터링 실패 시 전체 텍스트 반환 옵션

**디버깅 정보 수집:**
```python
# modules/vlm/rag.py의 search_by_product 함수에 추가
print(f"[DEBUG] 검색된 텍스트 샘플:")
for i, doc in enumerate(primary_docs[:3]):
    print(f"  [{i}] {doc.page_content[:200]}...")
```

### 2. defect_mapping.json 관리자 페이지 연동

**현재 상태:**
- ✅ 수동 JSON 파일 편집으로 관리
- ❌ 웹 UI 없음

**향후 구현 필요:**
```python
# 관리자 페이지 API 엔드포인트 (예시)
@app.get("/admin/mapping")
async def get_mapping_admin():
    """매핑 관리 페이지"""
    # 전체 매핑 정보 반환

@app.post("/admin/mapping/product")
async def add_product(product_id, product_name_ko):
    """제품 추가"""
    mapper.add_product(product_id, product_name_ko)
    
    # ⚠️ RAG 인덱스도 함께 재구축해야 함!
    rag.rebuild_index()

@app.post("/admin/mapping/defect")
async def add_defect(product, defect_id, defect_ko, full_name_ko, keywords):
    """불량 추가"""
    mapper.add_defect(product, defect_id, defect_ko, full_name_ko, keywords)

@app.post("/admin/rag/rebuild")
async def rebuild_rag():
    """RAG 인덱스 재구축"""
    # 새 매뉴얼 PDF 추가 시 호출
    rag.rebuild_index()
```

**중요: defect_mapping.json 갱신 시 필수 작업**
1. `defect_mapping.json` 파일 수정
2. API 서버 재시작 또는 `/mapping/reload` 호출
3. 새 매뉴얼 PDF 추가 시: `/rag/rebuild` 호출

---

## 📝 defect_mapping.json 파일

**파일 위치:** `/home/dmillion/llm_chal_vlm/web/defect_mapping.json`

**현재 구조:**
```json
{
  "products": {
    "제품ID": {
      "name_ko": "제품명(한글)",
      "defects": {
        "불량ID": {
          "en": "영문명",
          "ko": "한글명",
          "full_name_ko": "전체명",
          "keywords": ["검색키워드1", "검색키워드2"]
        }
      }
    }
  }
}
```

**⚠️ 관리 시 주의사항:**
1. 제품 추가/삭제 시 RAG 인덱스 재구축 필요
2. 불량 추가 시 keywords 배열에 다양한 유사어 포함
3. 파일명 규칙: `{제품ID}_manual.pdf` (예: `leather_manual.pdf`)
4. JSON 형식 오류 주의 (쉼표, 따옴표)

**현재 등록된 제품/불량:**
| 제품 | 불량 개수 | 불량 유형 |
|------|----------|----------|
| prod1 | 3 | hole, burr, scratch |
| grid | 3 | hole, burr, scratch |
| carpet | 4 | hole, burr, scratch, stain |
| leather | 6 | hole, burr, scratch, fold, stain, color |

---

## 🔧 환경 설정

### Python 패키지 (주요)
```bash
# Phase 1 최적화 관련
torch>=2.0.0
torchvision
open_clip_torch
faiss-gpu  # 또는 faiss-cpu

# RAG 관련
langchain
langchain-community
sentence-transformers
pypdf

# 웹 서버
fastapi
uvicorn

# 유틸리티
pillow
numpy
```

### 서버 환경
- OS: Rocky Linux 8.10
- Python: 3.9
- GPU: Tesla T4
- CUDA: 11.8

### 실행 명령
```bash
# API 서버
cd /home/dmillion/llm_chal_vlm/web
python api_server.py

# LLM 서버
cd /home/dmillion/llm_chal_vlm/llm_server
python llm_server.py
```

---

## 📊 성능 지표

### 인덱스 구축 속도
| 항목 | 기존 | Phase 1 | 개선율 |
|------|------|---------|--------|
| 불량 이미지 (2,083개) | 5-10분 | 30-60초 | **10배** |
| 정상 이미지 (전체) | - | 약 1-2분 | - |

### RAG 검색 속도
| 항목 | 속도 |
|------|------|
| 인덱스 구축 (최초 1회) | 1-2분 |
| 인덱스 로드 (서버 시작) | 1-2초 |
| 제품별 검색 | 0.1-0.5초 |

---

## 🐛 알려진 이슈

### 1. RAG 필터링 실패 (긴급)
- **증상**: 검색 결과는 있으나 원인/조치 분류 실패
- **영향**: 매뉴얼 생성 시 빈 컨텍스트
- **우선순위**: 최우선

### 2. 경고 메시지 (낮음)
```
RuntimeWarning: networkx backend defined more than once
LangChainDeprecationWarning: pydantic v1 compatibility
```
- **해결**: `api_server.py` 상단에 경고 필터 추가됨
```python
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
```

### 3. 세션 관리 (해결됨 ✅)
- ~~탭 이동 시 세션 초기화 문제~~
- 해결: 세션 유지 로직 개선

---

## 🎯 향후 개발 로드맵

### Phase 2: 관리자 기능 (2-3주)
- [ ] 관리자 페이지 구축
  - [ ] 제품/불량 관리 UI
  - [ ] 매뉴얼 업로드 UI
  - [ ] RAG 인덱스 관리
- [ ] 사용자 권한 관리
- [ ] 이력 조회 기능

### Phase 3: 데이터베이스 연동 (3-4주)
- [ ] PostgreSQL 연동
- [ ] 조치내역 저장 (10개 테이블)
- [ ] 통계 대시보드
- [ ] 검색 이력 관리

### Phase 4: 고급 기능 (4주+)
- [ ] 증분 인덱스 업데이트
- [ ] 모델 양자화 (FP16/INT8)
- [ ] 배치 이미지 처리
- [ ] 실시간 모니터링

---

## 📞 참고 정보

### API 엔드포인트
```
# 기본
GET  /                          # 업로드 페이지
GET  /health2                   # 헬스체크 (ALB용)

# 업로드
POST /upload/image              # 이미지 업로드
GET  /upload/status             # 업로드 상태

# 유사도 검색
POST /search/similarity         # 유사 이미지 검색
GET  /search/index/status       # 검색 인덱스 상태

# 이상 검출
POST /anomaly/detect            # 이상 검출 수행
GET  /anomaly/image/{id}/{file} # 결과 이미지 서빙

# 매뉴얼 생성
POST /manual/generate           # 매뉴얼 생성

# RAG 관리
GET  /rag/status                # RAG 상태 조회
POST /rag/rebuild               # 인덱스 재구축

# 매핑 관리
GET  /mapping/status            # 매핑 상태 조회
POST /mapping/reload            # 매핑 재로드
```

### 주요 디렉토리
```
/home/dmillion/llm_chal_vlm/
├── data/                       # 이미지 데이터
├── manual_store/               # 매뉴얼 PDF
├── web/                        # 웹 서버
├── llm_server/                 # LLM 서버
└── modules/                    # 공통 모듈
```

---

## ✅ 체크리스트 (다음 세션 시작 전)

- [ ] RAG 필터링 로직 개선
- [ ] PDF 매뉴얼 내부 구조 분석
- [ ] 디버깅 로그 추가하여 검색 결과 확인
- [ ] 필터링 키워드 확장 테스트
- [ ] defect_mapping.json이 leather/fold 포함하는지 확인
- [ ] 서버 재시작 후 매핑 로드 로그 확인

---
**버전**: 2.0  
**다음 세션 목표**: RAG 검색 결과 필터링 개선 및 안정화
# 다음 세션 시작 가이드

## 현재까지 완료된 내용

### ✅ 완료 (2025-11-12 기준)
1. **CLIP 기반 유사도 검색**: TOP-K 검색, 인덱스 관리 완료
2. **PatchCore 이상 검출**: 자동 정상 기준 이미지 선정, 결과 시각화 완료
3. **웹 UI**: matching.html (검색 + 이상 검출 + 불량 등록) 완료
4. **불량 이미지 관리**: 자동 파일명 생성, SEQ 번호 관리 완료
5. **RAG 파이프라인**: PDF 매뉴얼 벡터 DB 구축, 불량별 원인/조치 검색 완료
6. **LLM 통합**: LLM 서버 연동, 매뉴얼 기반 대응 방안 생성 완료

### ⚠️ 최근 해결한 이슈
- RAG 벡터 DB 빈 상태 문제 → 재구축으로 해결
- 원인/조치 구분 안 되는 문제 → 정규식 파싱 개선
- LLM 출력 반복 생성 문제 → Stop token 및 후처리 추가

### 🔄 진행 중
- LLM 응답 품질 개선 (반복 억제, 출력 길이 제어)
- 프롬프트 최적화 (매뉴얼 인용 강제, 간결성 향상)

---

## 빠른 시작

### 1. 환경 확인
```bash
ssh -p 2022 root@dm-nlb-112319415-f8e0a97d0b99.kr.lb.naverncp.com
cd /home/dmillion/llm_chal_vlm
```

### 2. 서버 상태 확인
```bash
# API 서버 (포트 5000)
ps aux | grep api_server

# LLM 서버 (포트 5001)
ps aux | grep llm_server

# 실행 중이 아니면 시작
cd web
python api_server.py &

cd ../llm_server
python llm_server.py &
```

### 3. 웹 접속
```
http://dm-alb-112319279-991b4e0889c4.kr.lb.naverncp.com
```

---

## 시스템 아키텍처

### 전체 워크플로우
```
[불량 이미지 업로드]
    ↓
[CLIP 유사도 검색] (TOP-K)
    ↓
[PatchCore 이상 검출] (자동 정상 기준 선정)
    ↓
[RAG 매뉴얼 검색] (원인/조치 분리)
    ↓
[LLM 대응 방안 생성] (4개 섹션)
    ↓
[결과 출력]
```

### 주요 컴포넌트

#### 1. API 서버 (api_server.py, 포트 5000)
- **엔드포인트**:
  - `/search/upload`: 이미지 업로드 및 유사도 검색
  - `/detect_anomaly`: PatchCore 이상 검출
  - `/generate_manual_advanced`: 통합 분석 (검색 + 이상 검출 + 매뉴얼)
  - `/manual/generate/llm`: LLM 대응 방안 생성
- **담당**: 전체 파이프라인 오케스트레이션

#### 2. LLM 서버 (llm_server.py, 포트 5001)
- **모델**: 
  - LLM: `naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B`
  - VLM: `llava-hf/llava-1.5-7b-hf` (선택적)
- **엔드포인트**:
  - `/analyze`: 텍스트 기반 분석 (매뉴얼 + 메타데이터)
  - `/analyze_vlm`: 이미지 + 텍스트 분석
- **최근 개선**:
  - Stop token 추가로 반복 생성 방지
  - Repetition penalty 1.3 적용
  - 출력 후처리 (`_clean_llm_output()`)

#### 3. RAG 매니저 (modules/vlm/rag_manager.py)
- **기능**: PDF 매뉴얼 벡터 검색
- **벡터 DB**: FAISS (jhgan/ko-sbert-nli 임베딩)
- **파싱 로직**: 
  - 불량별 섹션 분리 (정규식 기반)
  - 원인/조치 별도 추출
  - 불릿 포인트(•) 자동 정리
- **경로**:
  - PDF: `/home/dmillion/llm_chal_vlm/manual_store/prod1_menual.pdf`
  - 벡터 DB: `/home/dmillion/llm_chal_vlm/manual_store/`

---

## 주요 설정

### GPU 환경
- **서버**: NCP GPU 서버 (Tesla T4 x2, 총 32GB)
- **OS**: Rocky Linux 8.10
- **Python**: 3.9
- **CUDA**: 사용 가능

### 모델 메모리 사용
- CLIP (ViT-B-32): ~1GB
- PatchCore: ~2GB (제품당 메모리 뱅크)
- LLM (HyperCLOVA 1.5B): ~4GB (FP16)
- 임베딩 (ko-sbert): ~500MB
- **총 사용량**: ~10GB (충분한 여유)

### 데이터 구조
```
/home/dmillion/llm_chal_vlm/
├── data/
│   ├── def_split/              # 불량 이미지 (검색 인덱스)
│   │   ├── prod1_hole_001.jpg
│   │   ├── prod1_burr_001.jpg
│   │   └── ...
│   └── patchCore/              # PatchCore 메모리 뱅크
│       ├── prod1/
│       ├── prod2/
│       └── prod3/
├── manual_store/               # RAG 벡터 DB
│   ├── prod1_menual.pdf
│   ├── index.faiss
│   └── index.pkl
└── web/
    ├── uploads/                # 업로드된 이미지
    └── static/                 # 정적 파일
```

---

## 핵심 코드 위치

### RAG 관련
```python
# modules/vlm/rag_manager.py
class RAGManager:
    def search_defect_manual(self, product, defect_en, keywords):
        """
        불량별 원인/조치 분리 검색
        - 정규식으로 섹션 파싱
        - 불릿 포인트 자동 정리
        """
```

### LLM 프롬프트
```python
# llm_server/llm_server.py
def _build_prompt(req: AnalysisRequest):
    """
    매뉴얼 기반 간결한 프롬프트 생성
    - 원인/조치 명확히 구분
    - 4개 섹션 강제 (불량 현황/원인/대응/예방)
    """
```

### 통합 파이프라인
```python
# web/api_server.py
@app.post("/generate_manual_advanced")
async def generate_manual_advanced(request: dict):
    """
    1. 유사도 검색 → product/defect 추출
    2. PatchCore 이상 검출
    3. RAG 매뉴얼 검색
    4. LLM 대응 방안 생성
    """
```

---

## 테스트 방법

### 1. RAG 검색 테스트
```bash
cd /home/dmillion/llm_chal_vlm
python3 << 'EOF'
from modules.vlm import RAGManager
from pathlib import Path

rag = RAGManager(
    pdf_path=Path("manual_store/prod1_menual.pdf"),
    vector_store_path=Path("manual_store"),
    verbose=True
)

results = rag.search_defect_manual("prod1", "hole", ["hole", "기공"])
print(f"원인: {len(results['원인'])}개")
print(f"조치: {len(results['조치'])}개")

# 원인/조치가 다른지 확인
assert results['원인'] != results['조치'], "원인과 조치가 구분되어야 함"
EOF
```

### 2. LLM 서버 테스트
```bash
curl -X POST http://localhost:5001/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "product": "prod1",
    "defect_en": "hole",
    "defect_ko": "기공",
    "full_name_ko": "기공",
    "anomaly_score": 0.0,
    "is_anomaly": false,
    "manual_context": {
      "원인": ["주조 과정 중 금속 용탕 내 공기나 가스가 완전히 배출되지 않고 응고 중에 남아"],
      "조치": ["탈기(De-gassing) 장치 점검 및 아르곤 탈기 공정 강화"]
    },
    "max_new_tokens": 400,
    "temperature": 0.2
  }'
```

### 3. 통합 테스트 (웹 UI)
1. 브라우저에서 `http://dm-alb-112319279-991b4e0889c4.kr.lb.naverncp.com` 접속
2. 불량 이미지 업로드 (예: prod1_hole_xxx.jpg)
3. "고급 분석" 버튼 클릭
4. 결과 확인:
   - 유사 이미지 (TOP-5)
   - 이상 검출 결과 (히트맵)
   - LLM 대응 방안 (4개 섹션)

---

## 알려진 이슈 및 해결 방법

### 1. LLM 출력 반복 생성
**증상**: 4개 섹션 작성 후 다시 반복하거나 "assistant" 문구 출현

**해결**:
```python
# llm_server.py에서
gen_kwargs = dict(
    repetition_penalty=1.3,  # 반복 억제
)

# 후처리 추가
text = text.split("assistant")[0].strip()
text = text.split("[회사")[0].strip()
```

### 2. RAG 검색 결과 없음
**증상**: `원인: 0개, 조치: 0개`

**해결**:
```bash
# 벡터 DB 재구축
cd /home/dmillion/llm_chal_vlm
rm -rf manual_store/*.faiss manual_store/*.pkl
python3 modules/vlm/rag_manager.py
```

### 3. PatchCore 메모리 부족
**증상**: CUDA out of memory

**해결**:
```python
# api_server.py에서
detector = create_detector(
    device="cuda:0",  # 특정 GPU 지정
)

# 또는 FP16 사용
detector.model.half()
```

---

## 다음 작업 (우선순위)

### 높음
1. ✅ LLM 출력 품질 안정화 (완료)
2. ⏳ 웹 UI 개선 (LLM 결과 탭 추가)
3. ⏳ 다중 제품 매뉴얼 지원 (prod2, prod3)

### 중간
1. 데이터베이스 연동 (PostgreSQL)
2. 검색/분석 히스토리 저장
3. 사용자 권한 관리

### 낮음
1. VLM 모드 활성화 (이미지 직접 분석)
2. 배치 처리 (여러 이미지 동시 분석)
3. 성능 모니터링 대시보드

---

## 체크리스트

### RAG 파이프라인
- [x] PDF 벡터 DB 구축
- [x] 불량별 섹션 파싱 (정규식)
- [x] 원인/조치 분리 검색
- [x] LLM 프롬프트 통합
- [ ] 다중 PDF 지원 (prod2, prod3)

### LLM 통합
- [x] LLM 서버 구축 (포트 5001)
- [x] API 엔드포인트 (/analyze)
- [x] 프롬프트 템플릿 개선
- [x] 출력 반복 방지 (stop token)
- [x] 후처리 로직 (cleaning)
- [ ] VLM 모드 테스트

### 웹 UI
- [x] 유사도 검색 UI
- [x] 이상 검출 UI
- [x] 불량 등록 UI
- [ ] LLM 결과 탭 추가
- [ ] 히스토리 뷰어

---

## 다음 세션 시작 멘트
```
"이전 세션에서 CLIP 유사도 검색, PatchCore 이상 검출, RAG 파이프라인, LLM 통합까지 완료했습니다.

현재 상태:
- API 서버: http://dm-alb-112319279-991b4e0889c4.kr.lb.naverncp.com (포트 5000)
- LLM 서버: localhost:5001
- 불량 PDF: /home/dmillion/llm_chal_vlm/manual_store/prod1_menual.pdf
- 벡터 DB: /home/dmillion/llm_chal_vlm/manual_store/
- 환경: NCP GPU 서버 (Tesla T4 x2, Rocky Linux 8.10)

최근 해결한 이슈:
1. RAG 검색 결과 없음 → 벡터 DB 재구축
2. 원인/조치 구분 안 됨 → 정규식 파싱 개선
3. LLM 출력 반복 생성 → Stop token + 후처리

다음 작업:
- 웹 UI에 LLM 결과 탭 추가
- 다중 제품 매뉴얼 지원 (prod2, prod3)
- DB 연동 (PostgreSQL)

session_handover.md와 project_status.md를 확인했으니 바로 시작해주세요."
```

---

**작성일**: 2025-11-12  
**다음 세션**: 웹 UI 개선 및 다중 제품 지원
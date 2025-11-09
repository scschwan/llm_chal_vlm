# 다음 세션 시작 가이드

## 현재까지 완료된 내용

### ✅ 완료
1. **CLIP 기반 유사도 검색**: TOP-K 검색, 인덱스 관리
2. **PatchCore 이상 검출**: 자동 기준 이미지 선정, 결과 시각화
3. **웹 UI**: matching.html (검색 + 이상 검출 + 불량 등록)
4. **불량 이미지 관리**: 자동 파일명 생성, SEQ 번호 관리

### ⏳ 다음 작업: VLM/LLM 대응 매뉴얼 생성

## 빠른 시작

### 1. 환경 확인
```bash
ssh -p 2022 root@dm-nlb-112319415-f8e0a97d0b99.kr.lb.naverncp.com
cd /home/dmillion/llm_chal_vlm
```

### 2. 서버 상태 확인
```bash
# API 서버 실행 여부
ps aux | grep api_server

# 실행 중이 아니면 시작
cd web
python api_server.py
```

### 3. 웹 접속
```
http://dm-alb-112319279-991b4e0889c4.kr.lb.naverncp.com
```

## VLM/LLM 모듈 개발 계획

### 목표
불량 이미지 분석 결과 → PDF 매뉴얼 검색 → LLM 대응 방안 생성

### 필요 파일
- `modules/rag_manager.py`: RAG 파이프라인
- `modules/vlm_inference.py`: VLM 추론
- `defect_menual.pdf`: 불량 대응 매뉴얼 (이미 존재)

### 개발 단계

#### Step 1: RAG 파이프라인 구축
```python
# modules/rag_manager.py 생성 필요

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

class DefectManualRAG:
    def __init__(self, pdf_path, vector_store_path):
        # PDF 로드 및 벡터 DB 구축
        pass
    
    def search_manual(self, query, top_k=3):
        # 관련 매뉴얼 검색
        pass
```

#### Step 2: VLM 통합
```python
# modules/vlm_inference.py 개선 필요
# 기존 modules/vlm_local.py 참고

class VLMInference:
    def __init__(self, model_name="llava-v1.6-mistral-7b-hf"):
        # LLaVA 또는 EXAONE 로드
        pass
    
    def generate_response(self, images, manual_context, prompt):
        # 이미지 + 매뉴얼 컨텍스트 → 대응 방안 생성
        pass
```

#### Step 3: API 엔드포인트 추가
```python
# api_server.py에 추가

@app.post("/generate_manual")
async def generate_manual(request: ManualGenerationRequest):
    """
    불량 분석 결과를 받아 대응 매뉴얼 생성
    
    Input:
    - test_image_path
    - anomaly_result (이상 검출 결과)
    - search_results (유사 이미지)
    
    Output:
    - manual_sections (검색된 매뉴얼 섹션)
    - generated_response (LLM 생성 답변)
    """
    pass
```

#### Step 4: UI 통합
- matching.html에 3번째 탭 추가 또는
- manual_mapping.html 활용

## 주요 설정

### 모델 선택
1. **LLaVA 1.6**: 검증된 멀티모달 모델
2. **EXAONE-3.5-VL**: 한국어 특화 (추천)

### 한국어 임베딩
- `jhgan/ko-sbert-nli`: 한국어 문장 임베딩

### GPU 메모리
- Tesla T4 x2 (총 32GB)
- CLIP + PatchCore + VLM 동시 로드 가능

## 참고 코드

### 기존 VLM 코드
```python
# modules/vlm_local.py 참고
# 현재는 플레이스홀더이므로 실제 구현 필요
```

### 기존 프롬프트
```python
# modules/prompts.py 참고
# build_ok_def_pair_prompt() 활용 가능
```

## 테스트 시나리오

### 1. 엔드투엔드 테스트
1. 불량 이미지 업로드
2. 유사도 검색 (TOP-5)
3. 이상 검출
4. **매뉴얼 생성 (신규)**
5. 결과 확인

### 2. 샘플 입력
```python
{
    "test_image": "prod1_hole_021.jpg",
    "anomaly_score": 0.85,
    "detected_defect": "hole",
    "roi_locations": [(x, y, w, h), ...]
}
```

### 3. 예상 출력
```json
{
    "status": "success",
    "manual_sections": [
        {
            "title": "구멍 불량 대응 방안",
            "content": "...",
            "page": 15
        }
    ],
    "llm_response": "검출된 구멍 불량은 다음과 같이 대응하십시오:\n1. ...\n2. ...",
    "confidence": 0.92
}
```

## 잠재적 이슈

### 1. GPU 메모리 부족
- **해결**: 모델 양자화 (4-bit, 8-bit)
- **대안**: VLM 추론 시 CLIP 언로드

### 2. PDF 처리 느림
- **해결**: 벡터 DB 사전 구축 및 캐싱

### 3. 한국어 품질
- **해결**: EXAONE 모델 사용 또는 프롬프트 개선

## 체크리스트

- [ ] `modules/rag_manager.py` 생성
- [ ] `modules/vlm_inference.py` 구현
- [ ] PDF 벡터 DB 구축 스크립트
- [ ] API 엔드포인트 `/generate_manual` 추가
- [ ] UI 탭 추가 또는 기존 페이지 수정
- [ ] 통합 테스트
- [ ] 성능 최적화 (필요 시)

## 다음 세션 시작 멘트
```
"이전 세션에서 CLIP 유사도 검색과 PatchCore 이상 검출까지 완료했습니다.
이제 VLM/LLM 대응 매뉴얼 생성 모듈을 구현하려고 합니다.

현재 상태:
- API 서버: http://dm-alb-112319279-991b4e0889c4.kr.lb.naverncp.com (포트 5000)
- 불량 PDF: /home/dmillion/llm_chal_vlm/defect_menual.pdf
- 환경: NCP GPU 서버 (Tesla T4 x2, Rocky Linux 8.10)

구현할 기능:
1. PDF 매뉴얼 RAG 파이프라인 (LangChain + FAISS)
2. VLM 모델 통합 (LLaVA 또는 EXAONE)
3. API 엔드포인트 /generate_manual
4. UI 통합

project_status.md와 session_handover.md를 확인했으니 바로 시작해주세요."
```

---

**작성일**: 2025-11-09
**다음 세션**: VLM/LLM 모듈 구현
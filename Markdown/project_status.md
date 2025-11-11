# 유사이미지 검색 솔루션 개발 현황

## 프로젝트 개요
- **목표**: CLIP 기반 유사이미지 검색 → Anomaly Detection → LLM 대응 매뉴얼 생성 파이프라인 구축
- **환경**: Naver Cloud Platform, Rocky Linux 8.10, Python 3.9, Tesla T4 x2 GPU
- **개발 방식**: Python 모듈 형태로 개발하여 외부 웹서버(Spring AI 등)에서 호출 가능하도록 구성

---

## 개발 구조
```
웹 개발자: Spring AI 기반 WAS 개발 (별도)
     ↓ (API 호출)
Python 모듈: 
  1. TOP-K 유사도 매칭 ✅
  2. 특징점 추출 (PatchCore Anomaly Detection) ✅
  3. VLM/LLM 대응 매뉴얼 생성 ✅
```

---

## 임시 테스트 환경
- **web 폴더**: 임시로 만든 HTML + FastAPI 기반 테스트 페이지
- **용도**: 웹 개발자의 WAS가 완성되기 전까지 기능 검증용
- **접속**: http://dm-alb-112319279-991b4e0889c4.kr.lb.naverncp.com

---

## 개발 모듈 현황

### 1. TOP-K 유사도 매칭 모듈 ✅ 완료
- **기반**: `modules/similarity_matcher.py`
- **모델**: CLIP ViT-B-32 (OpenAI)
- **기능**: 
  - CLIP 기반 이미지 임베딩
  - FAISS 인덱스 활용 고속 검색
  - TOP-K 유사 이미지 반환 (기본 K=5)
- **출력**: JSON 형태로 유사 이미지 경로 및 유사도 스코어
- **인덱스 경로**: `/home/dmillion/llm_chal_vlm/index_cache/`
- **갤러리**: `/home/dmillion/llm_chal_vlm/data/def_split/`

### 2. 특징점 추출 모듈 ✅ 완료
- **기반**: `modules/anomaly_detector.py` + PatchCore
- **방법**: PatchCore Anomaly Detection
- **기능**:
  - 입력 이미지와 자동 선정된 정상 기준 이미지 비교
  - 이상 영역 검출 (히트맵, 마스크, 오버레이)
  - 제품별 메모리 뱅크 관리 (prod1/prod2/prod3)
- **출력**: 
  - JSON: 이상 점수, 판정 결과
  - 이미지: 히트맵, 마스크, 오버레이
- **메모리 뱅크**: `/home/dmillion/llm_chal_vlm/data/patchCore/`

### 3. VLM/LLM 대응 매뉴얼 생성 모듈 ✅ 완료
- **방법**: RAG (Retrieval-Augmented Generation)
- **컴포넌트**:
  1. **RAG Manager** (`modules/vlm/rag_manager.py`)
     - PDF 매뉴얼 벡터 DB 구축 (FAISS)
     - 임베딩: jhgan/ko-sbert-nli
     - 불량별 원인/조치 분리 검색
  
  2. **LLM Server** (`llm_server/llm_server.py`, 포트 5001)
     - 모델: naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B
     - 매뉴얼 기반 4개 섹션 생성:
       - 불량 현황 요약
       - 원인 분석
       - 대응 방안
       - 예방 조치
  
  3. **Defect Mapper** (`modules/vlm/defect_mapper.py`)
     - 제품별/불량별 매핑 정보 관리
     - 검색 키워드 자동 생성

- **출력**: JSON 형태로 구조화된 대응 매뉴얼 텍스트
- **매뉴얼**: `/home/dmillion/llm_chal_vlm/manual_store/prod1_menual.pdf`
- **벡터 DB**: `/home/dmillion/llm_chal_vlm/manual_store/`

---

## 통합 API 엔드포인트

### API 서버 (포트 5000)
```python
# 1. 유사도 검색
POST /search/upload
Body: {file: 이미지파일, top_k: 5}
Response: {top_k_results: [...], total_gallery_size: N}

# 2. 이상 검출
POST /detect_anomaly
Body: {test_image_path: "...", reference_image_path: "..."}
Response: {image_score: 0.XX, is_anomaly: bool, mask_url: "..."}

# 3. 통합 분석 (검색 + 이상검출 + RAG + LLM)
POST /generate_manual_advanced
Body: {image_path: "..."}
Response: {
  similarity: {...},
  anomaly: {...},
  manual: {원인: [...], 조치: [...]},
  vlm_analysis: "..."
}

# 4. LLM 대응 방안 생성 (단독)
POST /manual/generate/llm
Body: {
  image_path: "...",
  product_name: "prod1",
  defect_name: "hole",
  anomaly_score: 0.0,
  is_anomaly: false
}
Response: {
  status: "success",
  llm_analysis: "...",
  manual: {...}
}
```

### LLM 서버 (포트 5001)
```python
# 텍스트 기반 분석
POST /analyze
Body: {
  product: "prod1",
  defect_en: "hole",
  defect_ko: "기공",
  anomaly_score: 0.0,
  is_anomaly: false,
  manual_context: {원인: [...], 조치: [...]}
}
Response: {
  status: "success",
  analysis: "...",
  model: "hyperclovax"
}

# VLM 이미지 분석 (선택적)
POST /analyze_vlm
Body: {
  image_path: "...",
  prompt: "..."
}
```

---

## 기존 코드베이스 개선 사항

### 1. `modules/similarity_matcher.py` ✅
- 인덱스 저장/로드 기능 추가 완료
- 자동 인덱스 재구축 기능 추가

### 2. `modules/anomaly_detector.py` ✅
- 정상 기준 이미지 자동 선정 기능 추가
- 유사도 매처 통합 (`detect_with_normal_reference()`)

### 3. `modules/vlm/rag_manager.py` ✅
- PDF 파싱 로직 구현 (정규식 기반)
- 원인/조치 분리 검색 기능
- 벡터 DB 캐싱

### 4. `llm_server/llm_server.py` ✅ 신규 생성
- LLM/VLM 통합 서버
- 프롬프트 템플릿 개선
- 출력 반복 방지 (stop token + 후처리)

---

## 개발 우선순위

### ✅ 완료
1. TOP-K 유사도 매칭 모듈
2. PatchCore 이상 검출 모듈
3. RAG 파이프라인 구축
4. LLM 서버 구축 및 통합
5. 통합 API 엔드포인트

### ⏳ 진행 중
1. LLM 출력 품질 개선 (반복 억제, 간결성)
2. 웹 UI 개선 (LLM 결과 탭 추가)

### 🔜 예정
1. 다중 제품 매뉴얼 지원 (prod2, prod3)
2. 데이터베이스 연동 (PostgreSQL)
3. 검색/분석 히스토리 저장
4. 사용자 권한 관리
5. VLM 모드 활성화 (이미지 직접 분석)

---

## NCP 인프라 정보
- **GPU 서버 접속**: `ssh -p 2022 root@dm-nlb-112319415-f8e0a97d0b99.kr.lb.naverncp.com`
- **ALB 주소**: `http://dm-alb-112319279-991b4e0889c4.kr.lb.naverncp.com:80`
- **서비스 포트**: 
  - API 서버: 5000 (ALB를 통해 80으로 접근)
  - LLM 서버: 5001 (내부 통신)
- **헬스체크**: `/health` (API 서버)
- **Object Storage**: `dm-obs` (VPC 전용, 사설 도메인 사용)

---

## GPU 메모리 사용 현황
- CLIP (ViT-B-32): ~1GB
- PatchCore: ~2GB (제품당 메모리 뱅크)
- LLM (HyperCLOVA 1.5B, FP16): ~4GB
- 임베딩 (ko-sbert): ~500MB
- **총 사용량**: ~10GB / 32GB (충분한 여유)

---

## 최근 해결한 주요 이슈

### 1. RAG 벡터 DB 빈 상태 문제 (2025-11-12)
**증상**: `search_defect_manual()` 호출 시 원인/조치 모두 0개 반환

**원인**: 
- 벡터 DB 폴더는 존재하지만 `index.faiss` 파일 없음
- 빈 벡터 DB를 로드하여 검색 결과 없음

**해결**: 
```bash
rm -rf manual_store/*.faiss manual_store/*.pkl
python3 modules/vlm/rag_manager.py  # 재구축
```

### 2. 원인/조치 구분 안 되는 문제 (2025-11-12)
**증상**: `manual_context`에서 원인과 조치가 동일한 내용

**원인**: 
- `search_defect_manual()` 함수의 파싱 로직 미비
- 전체 PDF 청크를 원인/조치 양쪽에 모두 반환

**해결**:
```python
# 정규식으로 "발생 원인"과 "조치 가이드" 섹션 분리
cause_match = re.search(r'발생\s*원인(.*?)조치\s*가이드', section, re.DOTALL)
action_match = re.search(r'조치\s*가이드(.*?)(?:요약|$)', section, re.DOTALL)
```

### 3. LLM 출력 반복 생성 문제 (2025-11-12)
**증상**: 
- 4개 섹션 작성 후 "assistant" 문구와 함께 다시 반복
- 또는 "[회사 로고]" 후 재작성

**원인**: 
- Stop token 미설정
- LLM이 자기 출력을 보고 계속 생성

**해결**:
```python
# 1. Repetition penalty 추가
gen_kwargs = dict(repetition_penalty=1.3)

# 2. 생성 텍스트 후처리
text = text.split("assistant")[0].strip()
text = text.split("[회사")[0].strip()

# 3. 예방 조치 섹션 이후 5줄 지나면 자르기
```

---

## 다음 단계

### 즉시 착수
1. 웹 UI에 LLM 결과 표시 탭 추가
2. prod2, prod3 매뉴얼 PDF 추가 및 벡터 DB 구축
3. 제품별 동적 매뉴얼 선택 로직

### 단기 (1-2주)
1. PostgreSQL 데이터베이스 스키마 구현
2. 검색/분석 히스토리 저장 기능
3. 사용자 인증 및 권한 관리

### 중기 (1개월)
1. VLM 모드 완전 활성화 (이미지 직접 분석)
2. 배치 처리 기능 (여러 이미지 동시 분석)
3. 성능 모니터링 대시보드

### 장기 (2개월 이상)
1. 자동 재학습 파이프라인 (Active Learning)
2. 불량 예측 모델 추가
3. 모바일 앱 연동

---

**최종 업데이트**: 2025-11-12  
**작성자**: AI Assistant  
**다음 리뷰**: 2025-11-19
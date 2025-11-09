# 유사이미지 검색 솔루션 개발 현황 (2025-11-09 업데이트)

## 프로젝트 개요
- **목표**: CLIP 기반 유사이미지 검색 → PatchCore Anomaly Detection → LLM 대응 매뉴얼 생성 파이프라인 구축
- **환경**: Naver Cloud Platform, Rocky Linux 8.10, Python 3.9
- **배포 방식**: FastAPI REST API 서버 (외부 Spring AI WAS에서 호출 가능)

## 인프라 구성 (NCP)
- **GPU 서버**: Tesla T4 x2
- **VPC**: dm-vpc (10.200.0.0/16)
- **서브넷**: 
  - Public: dm-pub-sub (10.200.0.0/24)
  - LB: dm-lb-sub (10.200.1.0/24)
  - NAT: dm-nat-sub (10.200.2.0/24)
  - Private: dm-pri-sub (10.200.3.0/24)
- **로드밸런서**: 
  - ALB: http://dm-alb-112319279-991b4e0889c4.kr.lb.naverncp.com:80
  - NLB: dm-nlb-112319415-f8e0a97d0b99.kr.lb.naverncp.com:2022
- **접속**: `ssh -p 2022 root@dm-nlb-112319415-f8e0a97d0b99.kr.lb.naverncp.com`
- **서비스 포트**: 
  - 내부: 5000 (API 서버)
  - 외부: ALB 80 → 8080/5000

## 개발 완료 모듈

### ✅ 모듈 1: TOP-K 유사도 매칭
- **파일**: `modules/similarity_matcher.py`
- **기능**: 
  - CLIP (ViT-B-32/openai) 기반 이미지 임베딩
  - FAISS 인덱스 활용 고속 검색
  - TOP-K 유사 이미지 반환
  - 인덱스 저장/로드 기능
- **상태**: ✅ 완료 및 테스트 통과
- **API 엔드포인트**: 
  - `POST /search/upload` - 이미지 업로드 & 검색
  - `POST /build_index` - 인덱스 구축
  - `GET /index/info` - 인덱스 정보 조회

### ✅ 모듈 2: PatchCore Anomaly Detection
- **파일**: `modules/anomaly_detector.py`
- **기능**:
  - PatchCore 기반 이상 영역 검출
  - 제품별 메모리뱅크 관리 (prod1, prod2, prod3)
  - 정상 이미지 자동 선택 (유사도 매칭 통합)
  - Heatmap, Mask, Overlay 이미지 생성
  - Comparison 이미지 (정상 vs 불량 비교)
- **상태**: ✅ 완료 및 테스트 통과
- **API 엔드포인트**:
  - `POST /detect_anomaly` - 이상 검출 수행
  - `GET /anomaly/image/{result_id}/{filename}` - 결과 이미지 서빙

### ✅ 모듈 3: RAG 기반 매뉴얼 검색
- **파일**: `modules/vlm/rag_manager.py`
- **기능**:
  - PDF 매뉴얼 파싱 및 벡터화
  - 불량별 섹션 분리 (발생 원인 / 조치 가이드)
  - FAISS 벡터 DB 구축 및 캐싱
  - 키워드 기반 관련 매뉴얼 검색
- **상태**: ✅ 완료 (pickle deserialization 이슈 해결)
- **설정**:
  - 임베딩 모델: jhgan/ko-sbert-nli
  - 벡터 DB 경로: `/home/dmillion/llm_chal_vlm/manual_store`
  - PDF 경로: `/home/dmillion/llm_chal_vlm/manual_store/prod1_menual.pdf`

### ⚠️ 모듈 4: LLM 기반 대응 매뉴얼 생성
- **파일**: `modules/vlm/llm_inference.py`
- **기능**:
  - 텍스트 기반 LLM 추론
  - RAG 검색 결과 + 이상 검출 결과 통합 분석
  - 구조화된 대응 매뉴얼 생성
- **상태**: ⚠️ 구현 완료, 테스트 대기 중
- **모델**: mistralai/Mistral-7B-Instruct-v0.2 (4-bit 양자화)
- **주의**: VLM(이미지+텍스트) 대신 LLM(텍스트 전용) 사용

### ❌ VLM (Vision Language Model) - 보류
- **파일**: `modules/vlm/vlm_inference.py`
- **문제**: 
  - LlavaNextProcessor 초기화 실패 (`image_token` 인자 오류)
  - tokenizers 라이브러리 버전 충돌 (0.15.2 vs 0.19.1)
  - sentence-transformers 의존성 문제
- **대안**: LLM(텍스트 전용)으로 대체
- **상태**: ❌ 보류 (의존성 해결 후 재시도 예정)

## 웹 인터페이스

### ✅ FastAPI REST API 서버
- **파일**: `web/api_server.py`
- **기능**:
  - 유사도 검색 API
  - 이상 검출 API
  - 불량 등록 API
  - 통합 분석 파이프라인 (`/generate_manual_advanced`)
- **상태**: ✅ 작동 중
- **포트**: 5000
- **헬스체크**: `GET /health2`

### ✅ 웹 UI (HTML/JavaScript)
- **파일**: `web/matching.html`
- **기능**:
  - 탭 기반 UI (유사도 검색 / 이상 영역 검출 / RAG 분석)
  - 이미지 업로드 및 결과 표시
  - 불량 이미지 등록 모달
  - 실시간 로그 표시
- **상태**: ✅ 작동 중
- **접속**: http://dm-alb-112319279-991b4e0889c4.kr.lb.naverncp.com/

## 데이터 구조

### 이미지 데이터
```
data/
├── def_split/              # 불량 이미지 (검색 인덱스)
│   ├── prod1_burr_001.jpeg
│   ├── prod1_hole_001.jpeg
│   └── prod1_scratch_001.jpeg
├── patchCore/             # PatchCore 메모리뱅크
│   ├── prod1/
│   │   ├── ok/           # 정상 이미지 (파일명: prod1_ok_*.jpeg)
│   │   └── bank.pt       # 메모리뱅크
│   ├── prod2/
│   └── prod3/
└── uploads/               # 업로드된 테스트 이미지
```

### 매뉴얼 데이터
```
manual_store/
├── prod1_menual.pdf       # PDF 매뉴얼
├── index.faiss           # FAISS 벡터 인덱스
└── index.pkl             # 메타데이터
```

### 설정 파일
```
web/
├── defect_config.json    # 불량 유형 설정
└── defect_mapping.json   # 불량명 매핑 (EN/KO)
```

## 통합 워크플로우

### 자동 분석 파이프라인 (`/generate_manual_advanced`)

```
1. 이미지 업로드
   ↓
2. 유사도 검색 (CLIP + FAISS)
   → 제품명/불량명 자동 추출
   ↓
3. PatchCore 이상 검출
   → 정상 이미지 자동 선택
   → Heatmap, Mask, Overlay 생성
   ↓
4. RAG 매뉴얼 검색
   → 불량별 원인/조치 검색
   ↓
5. LLM 분석
   → 통합 대응 매뉴얼 생성
   ↓
6. 결과 반환 (JSON)
```

## 주요 이슈 및 해결

### ✅ 해결된 이슈
1. **CLIP 검색 메서드 불일치**
   - 문제: `search_with_index()` 메서드 없음
   - 해결: `search()` 메서드 사용으로 통일

2. **이미지 경로 문제**
   - 문제: 상대 경로 `../data/def_split/...` 처리 실패
   - 해결: `/api/image/` 엔드포인트에서 경로 정규화

3. **파일명 파싱 오류**
   - 문제: `cast_ok_*` 파일명이 불량으로 인식됨
   - 해결: `prod1_ok_*`로 파일명 변경

4. **RAG FAISS deserialization 오류**
   - 문제: `allow_dangerous_deserialization` 필요
   - 해결: `FAISS.load_local(..., allow_dangerous_deserialization=True)`

### ⚠️ 진행 중 이슈
1. **VLM 로드 실패**
   - 문제: tokenizers 버전 충돌
   - 임시 대응: LLM(텍스트 전용)으로 대체
   - 향후 계획: 의존성 재정리 또는 다른 VLM 모델 시도

## 성능 지표

### 유사도 검색
- 인덱스 크기: 30~50개 이미지
- 검색 속도: ~0.1초 (FAISS)
- 모델: CLIP ViT-B-32 (512차원)

### PatchCore 이상 검출
- 메모리뱅크 크기: 614 패치 (prod1 기준)
- 검출 속도: ~1~2초
- 정확도: 테스트 중

### RAG 검색
- 벡터 DB 크기: ~100 청크
- 검색 속도: ~0.5초
- 임베딩: ko-sbert-nli (768차원)

### LLM 생성
- 모델: Mistral-7B (4-bit)
- 생성 속도: ~3~5초 (512 토큰)
- GPU 메모리: ~6GB

## 다음 단계

### 단기 (1주일)
- [ ] LLM 매뉴얼 생성 품질 테스트
- [ ] 웹 UI 개선 (통계, 히스토리)
- [ ] 불량 통계 대시보드 추가
- [ ] API 문서 자동 생성 (Swagger)

### 중기 (1개월)
- [ ] VLM 의존성 문제 해결
- [ ] 제품별 매뉴얼 분리 (prod1, prod2, prod3)
- [ ] 배치 처리 기능 추가
- [ ] 로깅 및 모니터링 강화

### 장기 (3개월)
- [ ] Active Learning 파이프라인
- [ ] 자동 재학습 시스템
- [ ] 모바일 앱 연동
- [ ] 실시간 생산라인 통합

## 미사용 파일 정리 대상

### Legacy 모듈 (원본 프로젝트)
```
modules/
├── clip_search.py          # → similarity_matcher.py로 대체
├── vlm_local.py           # → modules/vlm/vlm_inference.py로 대체
├── region_detect.py       # 사용 안 함 (PatchCore 사용)
├── ssim_utils.py          # 사용 안 함
├── shape_diff.py          # 사용 안 함
├── image_processor.py     # 사용 안 함
├── preprocess.py          # 사용 안 함
├── object_guidance.py     # 사용 안 함
├── prompts.py             # 사용 안 함
├── evaluate_yolo_seg.py   # YOLO 미사용
├── train_yolo_seg.py      # YOLO 미사용
├── train_val_pipeline.py  # 학습 파이프라인 미사용
└── split_train_to_test.py # 데이터 분할 미사용

루트 파일:
├── main.py                # CLI 스크립트 (API 서버로 대체)
├── config.py              # 레거시 설정 (미사용)
```

### 사용 중인 핵심 모듈
```
modules/
├── similarity_matcher.py    # ✅ 유사도 검색
├── anomaly_detector.py      # ✅ 이상 검출
├── patchCore/              # ✅ PatchCore 구현
└── vlm/                    # ✅ RAG + LLM
    ├── rag_manager.py
    ├── llm_inference.py
    ├── defect_mapper.py
    ├── prompt_builder.py
    └── vlm_inference.py    # ⚠️ 보류

web/
├── api_server.py           # ✅ FastAPI 서버
├── matching.html           # ✅ 웹 UI
└── static/                 # ✅ CSS/JS
```

## 참고 자료
- **설계서**: `Markdown/유사이미지_검색_솔루션_설계서.md`
- **인프라 정보**: `NCP_인프라_구축_완료_및_설정_내역_안내.pdf`
- **GitHub**: https://github.com/scschwan/llm_chal_vlm.git
- **ALB 접속**: http://dm-alb-112319279-991b4e0889c4.kr.lb.naverncp.com/

---

**마지막 업데이트**: 2025-11-09  
**작성자**: 프로젝트 팀  
**버전**: 2.0
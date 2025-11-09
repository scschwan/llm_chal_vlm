# 유사이미지 검색 솔루션 개발 현황

## 프로젝트 개요
- **목표**: CLIP 기반 유사이미지 검색 → Anomaly Detection → LLM 대응 매뉴얼 생성 파이프라인 구축
- **환경**: Naver Cloud Platform, Rocky Linux 8.10, Python 3.9
- **개발 방식**: Python 모듈 형태로 개발하여 외부 웹서버(Spring AI 등)에서 호출 가능하도록 구성

## 개발 구조
```
웹 개발자: Spring AI 기반 WAS 개발 (별도)
     ↓ (API 호출)
Python 모듈: 
  1. TOP-K 유사도 매칭 ✅ 완료
  2. 특징점 추출 (Anomaly Detection) ✅ 완료
  3. VLM/LLM 대응 매뉴얼 생성 ⏳ 예정
```

## NCP 인프라 정보
- **GPU 서버 접속**: `ssh -p 2022 root@dm-nlb-112319415-f8e0a97d0b99.kr.lb.naverncp.com`
- **ALB 주소**: `http://dm-alb-112319279-991b4e0889c4.kr.lb.naverncp.com`
- **API 서버 포트**: 5000 (의도적 설정, 8080은 헬스체크 전용)
- **헬스체크 엔드포인트**: `/health2` (경로: `/health2`, 포트: 5000)
- **Object Storage**: `dm-obs` (VPC 전용, 사설 도메인 사용)

## 개발 모듈 현황

### 1. TOP-K 유사도 매칭 모듈 ✅ 완료
- **기반**: `modules/clip_search.py` → `modules/similarity_matcher.py`로 확장
- **모델**: ViT-B-32/openai (CLIP)
- **기능**: 
  - CLIP 기반 이미지 임베딩
  - FAISS 인덱스 활용 고속 검색
  - TOP-K 유사 이미지 반환
  - 인덱스 저장/로드 기능
- **출력**: JSON 형태로 유사 이미지 경로 및 유사도 스코어
- **API 엔드포인트**:
  - `POST /search/upload`: 파일 업로드 검색
  - `POST /search`: 경로 기반 검색
  - `POST /build_index`: 인덱스 구축
  - `GET /index/info`: 인덱스 상태 확인

### 2. 특징점 추출 모듈 (Anomaly Detection) ✅ 완료
- **방법**: PatchCore 기반 Anomaly Detection
- **경로**: `modules/anomaly_detector.py`
- **메모리뱅크 위치**: `/home/dmillion/llm_chal_vlm/data/patchCore`
- **지원 제품**: prod1, prod2, prod3
- **기능**:
  - 제품명 자동 추출 (파일명 기반)
  - 정상 이미지 자동 검색 (raw data에서 CLIP 유사도 기반)
  - Heatmap, Mask, Overlay 이미지 생성
  - 비교 이미지 생성 (정상 이미지 vs 오버레이)
- **출력**: JSON 형태로 이상 점수, 이미지 URL
- **API 엔드포인트**:
  - `POST /detect_anomaly`: 이상 검출 수행
  - `GET /anomaly/image/{result_id}/{filename}`: 결과 이미지 서빙

**기술 결정 사항**:
- **기준 이미지 선정**: 정상 이미지 Raw Data에서 CLIP 유사도 기반 자동 선정
- **경로**: `data/patchCore/{product_name}/ok/` 폴더의 정상 이미지
- **장점**: 
  - 기존 CLIP 검색 모듈 재사용
  - 실제 이미지와 비교하여 직관적
  - 사용자에게 비교 대상 명확히 제시

### 3. VLM/LLM 대응 매뉴얼 생성 모듈 ⏳ 예정
- **방법**: RAG (Retrieval-Augmented Generation)
- **계획**:
  - PDF 매뉴얼 벡터 DB 구축 (LangChain)
  - 차이점 기반 관련 매뉴얼 검색
  - LLaVA 또는 오픈소스 LLM으로 대응 방안 생성
- **출력**: JSON 형태로 대응 매뉴얼 텍스트
- **상태**: 미구현

## 웹 인터페이스 (matching.html)

### 현재 기능
1. **유사도 검색 탭**
   - 이미지 업로드 (드래그 & 드롭 지원)
   - TOP-K 슬라이더 (1~20개)
   - 검색 결과 그리드 표시
   - TOP-1 스왑 기능: TOP-2~5 클릭 시 TOP-1과 교체
   - 인덱스 관리 (상태 확인, 재구축)

2. **이상 영역 검출 탭**
   - 자동 정상 기준 이미지 선정
   - 이상 영역 검출 실행
   - 결과 표시:
     - 정상 기준 이미지
     - 이상 영역 마스크
     - 비교 이미지 (정상 vs 오버레이)

3. **불량 이미지 등록**
   - 제품명/불량명 선택 (설정 파일 기반)
   - 자동 파일명 생성: `{product}_{defect}_{seqno:03d}.jpg`
   - SEQ 번호 자동 관리 (폴더 스캔 방식)
   - 등록 후 인덱스 자동 재구축

### UI/UX 개선 사항
- CSS/JS 인라인 포함 (ALB 정적 파일 서빙 제약 해결)
- 반응형 디자인
- 드래그 & 드롭 지원
- 실시간 상태 메시지

## 파일 구조
```
llm_chal_vlm/
├── web/
│   ├── matching.html              # 메인 UI (CSS/JS 인라인)
│   ├── api_server.py              # FastAPI 백엔드 (포트 5000)
│   ├── defect_config.json         # 제품/불량 설정
│   ├── uploads/                   # 업로드 임시 저장
│   ├── index_cache/               # CLIP 인덱스 캐시
│   └── anomaly_outputs/           # 이상 검출 결과
├── modules/
│   ├── similarity_matcher.py      # CLIP 유사도 검색
│   ├── anomaly_detector.py        # PatchCore 이상 검출
│   └── patchCore/                 # PatchCore 구현
├── data/
│   ├── def_split/                 # 불량 이미지 DB (검색 대상)
│   └── patchCore/                 # 메모리뱅크
│       ├── prod1/
│       │   ├── ok/                # 정상 이미지 raw data
│       │   ├── memory_bank.pt
│       │   ├── bank_config.json
│       │   └── tau.json
│       ├── prod2/
│       └── prod3/
└── defect_menual.pdf              # 불량 대응 매뉴얼 (RAG 예정)
```

## 설정 파일

### defect_config.json
```json
{
  "products": {
    "prod1": {
      "name": "제품1",
      "defects": ["hole", "burr", "scratch"]
    },
    "prod2": {
      "name": "제품2", 
      "defects": ["crack", "dent"]
    },
    "prod3": {
      "name": "제품3",
      "defects": ["stain", "discolor"]
    }
  }
}
```

## 불량 이미지 파일명 규칙

### 형식
```
{product_name}_{defect_name}_{seqno:03d}.{ext}
```

### 예시
```
prod1_hole_001.jpg
prod1_hole_002.jpg
prod1_burr_001.png
prod2_crack_015.jpg
```

### SEQ 번호 관리
- **방식**: 폴더 스캔 방식 (실시간 조회)
- **로직**: 
  1. `data/def_split/` 폴더에서 `{product}_{defect}_*` 패턴 검색
  2. 파일명에서 SEQ 번호 추출 (마지막 `_` 뒤 숫자)
  3. 최대값 + 1을 새 SEQ로 사용
- **장점**:
  - 별도 config 파일 불필요
  - 실제 파일과 항상 동기화
  - 파일 복구 시 자동 인식

## API 서버 실행

### 서버 시작
```bash
cd /home/dmillion/llm_chal_vlm/web
python api_server.py
```

### 접속 주소
- **내부**: `http://localhost:5000`
- **외부 (ALB)**: `http://dm-alb-112319279-991b4e0889c4.kr.lb.naverncp.com`

### 주요 엔드포인트
```
GET  /health2                    # 헬스체크
GET  /                           # matching.html 서빙
POST /search/upload              # 이미지 업로드 검색
POST /detect_anomaly             # 이상 검출
POST /register_defect            # 불량 이미지 등록
GET  /index/info                 # 인덱스 상태
POST /build_index                # 인덱스 재구축
GET  /defect/stats/{prod}/{def}  # 불량 통계
GET  /api/image/{path}           # 이미지 서빙
```

## 해결된 기술 이슈

### 1. 경로 문제
- **문제**: api_server.py의 상대 경로가 실행 위치에 따라 달라짐
- **해결**: 절대 경로 사용 (`project_root / "data" / "patchCore"`)

### 2. 제품명 추출
- **문제**: 파일명에서 제품명 추출 실패 시 에러
- **해결**: 메모리뱅크 디렉토리 확인 후 기본값 사용

### 3. 정적 파일 서빙
- **문제**: ALB 환경에서 CSS/JS 파일 404 에러
- **해결**: HTML 파일에 CSS/JS 인라인 포함

### 4. 불량 등록 422 에러
- **문제**: FastAPI Query 파라미터와 FormData 불일치
- **해결**: `Query(...)` → `Form(...)` 변경

### 5. 포트 설정
- **결정**: API 서버 5000 포트 사용 (의도적)
- **이유**: 
  - 8080 포트는 ALB 헬스체크 전용
  - 5000 포트로 서비스 구동 후 ALB를 통해 외부 접근
  - 헬스체크 경로: `/health2`

## 다음 개발 단계

### 우선순위 1: VLM/LLM 매뉴얼 생성 모듈 (다음 세션)
- [ ] RAG 파이프라인 구축
  - PDF 매뉴얼 로드 및 청킹
  - 한국어 임베딩 모델 적용 (jhgan/ko-sbert-nli)
  - FAISS 벡터 DB 구축
- [ ] LLM 통합
  - LLaVA 또는 EXAONE 모델 선택
  - 프롬프트 엔지니어링
  - 이미지 + 텍스트 컨텍스트 결합
- [ ] API 엔드포인트 추가
  - `POST /generate_manual`: 대응 매뉴얼 생성
- [ ] UI 탭 추가
  - `manual_mapping.html` 또는 matching.html 3번째 탭

### 우선순위 2: 설정 관리 페이지
- [ ] `defect_config_manager.html` 생성
- [ ] 제품/불량 CRUD 기능
- [ ] API 엔드포인트:
  - `GET /config/defects`: 설정 조회
  - `POST /config/defects`: 설정 저장
  - `PUT /config/products/{id}`: 제품 수정
  - `DELETE /config/products/{id}`: 제품 삭제

### 우선순위 3: 통계 및 모니터링
- [ ] `dashboard.html` 구현
- [ ] 불량 통계 시각화
- [ ] 검색 이력 관리
- [ ] 이상 검출 이력 관리

## 참고 사항

### 중요 경로
- **프로젝트 루트**: `/home/dmillion/llm_chal_vlm`
- **API 서버 실행 위치**: `/home/dmillion/llm_chal_vlm/web`
- **불량 이미지 DB**: `/home/dmillion/llm_chal_vlm/data/def_split`
- **메모리뱅크**: `/home/dmillion/llm_chal_vlm/data/patchCore`

### 데이터 파일 확인
```bash
# 불량 이미지 수 확인
ls -l /home/dmillion/llm_chal_vlm/data/def_split/ | grep prod1_hole | wc -l

# 정상 이미지 수 확인
ls -l /home/dmillion/llm_chal_vlm/data/patchCore/prod1/ok/ | wc -l

# 메모리뱅크 확인
ls -la /home/dmillion/llm_chal_vlm/data/patchCore/prod1/
```

### Python 의존성
```bash
# 주요 패키지
pip install fastapi uvicorn python-multipart
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install open-clip-torch faiss-cpu pillow numpy opencv-python
```

### 서버 재시작
```bash
# 기존 프로세스 종료
pkill -f api_server.py

# 재시작
cd /home/dmillion/llm_chal_vlm/web
nohup python api_server.py > server.log 2>&1 &

# 로그 확인
tail -f server.log
```

## 추후 고려사항

### 성능 최적화
- [ ] CLIP 인덱스 증분 업데이트 (전체 재구축 회피)
- [ ] 메모리뱅크 캐싱 개선
- [ ] 비동기 처리 (이미지 업로드, 인덱스 구축)
- [ ] GPU 메모리 관리

### 보안
- [ ] API 인증/인가
- [ ] 파일 업로드 크기 제한
- [ ] 입력 검증 강화
- [ ] HTTPS 적용

### 확장성
- [ ] 다중 GPU 지원
- [ ] 분산 처리 (Celery)
- [ ] 데이터베이스 연동 (검색 이력, 사용자 관리)
- [ ] 로깅 및 모니터링 시스템

---

**마지막 업데이트**: 2025-11-09
**개발 상태**: 모듈 1, 2 완료 / 모듈 3 예정
**다음 세션 목표**: VLM/LLM 대응 매뉴얼 생성 모듈 구현
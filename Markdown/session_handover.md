# Phase 2 개발 일정표
# 서버 배포 기능 구현

**기간**: 1주 (5 작업일)  
**목표**: 비동기 배치 서비스 구현 및 모델 재구축 기능 완성

---

## 개발 항목

### 1. CLIP 임베딩 재구축 (3.9) - 2일

#### 1.1 백엔드 개발 (1.5일)
- [ ] 라우터 생성: `web/routers/admin/deployment.py`
- [ ] API 엔드포인트:
  - `POST /api/admin/deployment/clip/normal` - 정상 이미지 인덱스
  - `POST /api/admin/deployment/clip/defect` - 불량 이미지 인덱스
  - `GET /api/admin/deployment/status/{task_id}` - 진행 상태 조회
  - `GET /api/admin/deployment/logs` - 배포 이력
- [ ] Object Storage 다운로드 → 로컬 배포 로직
- [ ] 비동기 배치 서비스:
  - asyncio 기반 병렬 다운로드 (최대 10개 동시)
  - CLIP 임베딩 생성
  - FAISS 인덱스 저장
- [ ] 진행 상태 추적 (Redis or DB)
- [ ] WebSocket 실시간 진행률 전달

#### 1.2 프론트엔드 개발 (0.5일)
- [ ] 화면 생성: `web/pages/admin/admin_deploy_clip.html`
- [ ] JavaScript: `web/static/js/admin/deployment_clip.js`
- [ ] 기능:
  - 인덱스 유형 선택 (정상/불량)
  - 재구축 시작 버튼
  - 실시간 진행률 표시
  - 배포 이력 테이블

### 2. PatchCore 메모리뱅크 생성 (3.10) - 1.5일

#### 2.1 백엔드 개발 (1일)
- [ ] API 엔드포인트 추가:
  - `POST /api/admin/deployment/patchcore` - 전체 메모리뱅크 생성
- [ ] 스크립트 실행 로직:
```python
  방법1: bash /home/dmillion/llm_chal_vlm/build_patchcore.sh
  방법2: python modules/patchCore/build_bank.py
```
- [ ] 비동기 프로세스 실행 (asyncio.create_subprocess_exec)
- [ ] 실시간 로그 파싱 및 진행 상태 업데이트
- [ ] DB 배포 이력 저장

#### 2.2 프론트엔드 개발 (0.5일)
- [ ] 화면 생성: `web/pages/admin/admin_deploy_patchcore.html`
- [ ] JavaScript: `web/static/js/admin/deployment_patchcore.js`
- [ ] 기능:
  - 전체 재생성 시작 버튼
  - 제품별 진행 상태 표시
  - 실시간 로그 출력
  - 배포 이력 테이블

### 3. 공통 유틸리티 (1.5일)

#### 3.1 Object Storage 매니저 (0.5일)
- [ ] 모듈 생성: `web/utils/object_storage.py`
- [ ] ObjectStorageManager 클래스:
  - create_folder()
  - upload_file()
  - download_file()
  - delete_file()
  - list_objects()
  - get_url()

#### 3.2 비동기 배치 서비스 (0.5일)
- [ ] 모듈 생성: `web/utils/async_batch.py`
- [ ] 기능:
  - 병렬 다운로드 (ThreadPoolExecutor)
  - 진행 상태 관리
  - 에러 핸들링
  - 재시도 로직

#### 3.3 WebSocket 통신 (0.5일)
- [ ] WebSocket 엔드포인트 구현
- [ ] 실시간 진행률 브로드캐스팅
- [ ] 프론트엔드 WebSocket 연결

---

## 세부 작업 내역

### Day 1: CLIP 재구축 백엔드 (전반)

**작업 항목:**
1. `web/utils/object_storage.py` 생성
   - boto3 기반 S3 클라이언트
   - 기본 CRUD 함수 구현

2. `web/routers/admin/deployment.py` 생성
   - CLIP 정상/불량 인덱스 API
   - 진행 상태 조회 API

3. Object Storage → 로컬 다운로드 로직
   - 병렬 다운로드 구현
   - 지정 경로 배치

**예상 산출물:**
- `web/utils/object_storage.py`
- `web/routers/admin/deployment.py` (일부)

---

### Day 2: CLIP 재구축 백엔드 (후반) + 프론트엔드

**작업 항목:**
1. CLIP 임베딩 생성 로직
   - 배치 단위 임베딩
   - FAISS 인덱스 저장

2. 비동기 작업 관리
   - asyncio 기반 배치 서비스
   - 진행 상태 DB 저장

3. 프론트엔드 개발
   - `admin_deploy_clip.html` 생성
   - 실시간 진행률 UI
   - 배포 이력 테이블

**예상 산출물:**
- `web/routers/admin/deployment.py` (완성)
- `web/pages/admin/admin_deploy_clip.html`
- `web/static/js/admin/deployment_clip.js`

---

### Day 3: PatchCore 메모리뱅크 백엔드

**작업 항목:**
1. PatchCore 배포 API
   - 스크립트 실행 엔드포인트
   - 비동기 프로세스 실행

2. 실시간 로그 파싱
   - 표준 출력 스트리밍
   - 진행 상태 파싱
   - DB 업데이트

3. 에러 핸들링
   - 스크립트 실패 처리
   - 롤백 로직

**예상 산출물:**
- `web/routers/admin/deployment.py` (PatchCore 추가)

---

### Day 4: PatchCore 프론트엔드 + WebSocket

**작업 항목:**
1. 프론트엔드 개발
   - `admin_deploy_patchcore.html` 생성
   - 실시간 진행 상태 UI
   - 로그 출력 영역

2. WebSocket 통신
   - 서버 WebSocket 엔드포인트
   - 클라이언트 연결 및 이벤트 핸들링
   - 실시간 진행률 업데이트

**예상 산출물:**
- `web/pages/admin/admin_deploy_patchcore.html`
- `web/static/js/admin/deployment_patchcore.js`
- WebSocket 통신 구현

---

### Day 5: 통합 테스트 및 디버깅

**작업 항목:**
1. 전체 플로우 테스트
   - 이미지 업로드 → CLIP 재구축
   - 정상 이미지 업로드 → PatchCore 생성
   - 진행 상태 추적 검증

2. 에러 케이스 테스트
   - 네트워크 오류
   - 스크립트 실패
   - 중복 실행 방지

3. 성능 최적화
   - 병렬 다운로드 성능 측정
   - 메모리 사용량 체크
   - 타임아웃 설정

**산출물:**
- 테스트 보고서
- 버그 수정 패치

---

## 기술 스택

### 백엔드
- **FastAPI**: 비동기 웹 프레임워크
- **boto3**: AWS S3 호환 Object Storage 클라이언트
- **asyncio**: 비동기 프로세스 실행
- **subprocess**: 쉘 스크립트 실행
- **WebSocket**: 실시간 통신

### 프론트엔드
- **HTML/CSS/JavaScript**: 기본 웹 기술
- **WebSocket API**: 실시간 진행률 수신
- **Fetch API**: RESTful API 호출

### 인프라
- **Naver Cloud Platform Object Storage**
  - Endpoint: https://kr.object.ncloudstorage.com
  - Bucket: dm-obs (환경변수)
- **Rocky Linux 8.10**
- **Python 3.9**

---

## 주요 파일 목록

### 신규 생성 파일
```
web/
├── routers/admin/
│   └── deployment.py                  # 배포 API 라우터
├── pages/admin/
│   ├── admin_deploy_clip.html         # CLIP 재구축 화면
│   └── admin_deploy_patchcore.html    # PatchCore 생성 화면
├── static/js/admin/
│   ├── deployment_clip.js             # CLIP 재구축 JS
│   └── deployment_patchcore.js        # PatchCore 생성 JS
└── utils/
    ├── object_storage.py              # Object Storage 유틸
    └── async_batch.py                 # 비동기 배치 서비스
```

### 수정 파일
```
web/
├── api_server.py                      # 라우터 등록 추가
├── routers/admin/__init__.py          # deployment 라우터 임포트
└── pages/admin/admin_layout.html      # 네비게이션 메뉴 추가
```

---

## 환경 변수
```bash
# Object Storage 인증
NCP_ACCESS_KEY=your_access_key
NCP_SECRET_KEY=your_secret_key
NCP_BUCKET=dm-obs

# Database
DB_HOST=localhost
DB_PORT=3306
DB_USER=dmillion
DB_PASSWORD=your_password
DB_NAME=defect_db
```

---

## 배포 전 체크리스트

- [ ] Object Storage 인증 정보 설정
- [ ] DB 테이블 생성 (deployment_logs)
- [ ] /data/patchCore/ 디렉토리 구조 확인
- [ ] build_patchcore.sh 실행 권한 확인
- [ ] CLIP 모델 로드 테스트
- [ ] PatchCore 모듈 임포트 테스트
- [ ] 네트워크 정책 (Object Storage 접근 허용)

---

## 리스크 관리

### 예상 리스크
1. **Object Storage 다운로드 속도**
   - 완화: 병렬 다운로드 (최대 10개 동시)
   - 완화: 로컬 캐싱

2. **PatchCore 메모리뱅크 생성 시간**
   - 완화: 진행 상태 실시간 표시
   - 완화: 백그라운드 작업 큐

3. **동시 재구축 요청**
   - 완화: 작업 큐 구현
   - 완화: 진행 중 중복 실행 방지

4. **스크립트 실행 실패**
   - 완화: 에러 로그 상세 기록
   - 완화: 자동 재시도 (최대 3회)

---

## 성공 기준

### 기능 요구사항
- [✅] CLIP 정상/불량 인덱스 재구축 성공
- [✅] PatchCore 전체 메모리뱅크 생성 성공
- [✅] 실시간 진행률 표시 (WebSocket)
- [✅] 배포 이력 조회 및 로그 확인

### 성능 요구사항
- [ ] CLIP 재구축 시간: 650장 기준 5분 이내
- [ ] PatchCore 생성 시간: 전체 제품 기준 10분 이내
- [ ] Object Storage 다운로드: 100MB 기준 1분 이내

### 안정성 요구사항
- [ ] 에러 발생 시 자동 복구 또는 명확한 에러 메시지
- [ ] 네트워크 오류 시 재시도 로직
- [ ] 스크립트 실패 시 롤백 처리

---

**작성일**: 2025-11-14  
**작성자**: Claude  
**버전**: 1.0
# Phase 3 개발 일정표
# 설정 및 모니터링 기능 구현

**기간**: 3-4일 (작업일)  
**목표**: 이미지 전처리 설정, 모델 선택, 대시보드 구현

---

## 이전 세션 완료 내역

### Phase 2: 서버 배포 기능 ✅ 완료

**완성된 파일:**
1. `web/routers/admin/__init__.py` - 라우터 패키지 초기화
2. `web/routers/admin/deployment.py` - CLIP/PatchCore 배포 API
3. `web/pages/admin/admin_deploy_clip.html` - CLIP 재구축 화면
4. `web/pages/admin/admin_deploy_patchcore.html` - PatchCore 생성 화면
5. `web/static/admin/css/admin_deploy_clip.css` - CLIP 화면 스타일
6. `web/static/admin/css/admin_deploy_patchcore.css` - PatchCore 화면 스타일
7. `web/static/admin/js/admin_deploy_clip.js` - CLIP 화면 로직
8. `web/static/admin/js/admin_deploy_patchcore.js` - PatchCore 화면 로직

**수정된 파일:**
- `web/database/crud.py` - 배포 로그 관련 함수 추가
- `web/database/models.py` - DeploymentLog 테이블 모델 추가
- `web/api_server.py` - deployment 라우터 등록 및 페이지 서빙 추가

**DB 작업:**
- `deployment_logs` 테이블 생성 완료

**구현된 기능:**
- CLIP 정상/불량 이미지 인덱스 재구축
- PatchCore 전체 메모리뱅크 생성
- 실시간 진행 상태 추적 (폴링 방식)
- 제품별 생성 상태 시각화
- 실시간 로그 출력
- 배포 이력 조회

**알려진 이슈:**
- TopKSimilarityMatcher import 오류 수정 완료
- 페이지 서빙 라우트 수동 추가 필요 (완료)

---

## 개발 항목

### 1. 이미지 전처리 설정 (3.7) - 1.5일

#### 1.1 백엔드 개발 (1일)
- [ ] 라우터 생성: `web/routers/admin/preprocessing.py`
- [ ] API 엔드포인트:
  - `GET /api/admin/preprocessing` - 현재 설정 조회
  - `POST /api/admin/preprocessing` - 설정 저장
  - `PUT /api/admin/preprocessing/{id}` - 설정 수정
  - `DELETE /api/admin/preprocessing/{id}` - 설정 삭제
- [ ] 전처리 옵션:
  - 이미지 리사이즈 (해상도 설정)
  - 정규화 방법
  - 증강 기법 (회전, 반전, 밝기 조정)
  - 노이즈 제거
- [ ] DB 테이블: `preprocessing_configs`
- [ ] 설정 적용 검증 로직

#### 1.2 프론트엔드 개발 (0.5일)
- [ ] 화면 생성: `web/pages/admin/admin_preprocessing.html`
- [ ] CSS: `web/static/admin/css/admin_preprocessing.css`
- [ ] JavaScript: `web/static/admin/js/admin_preprocessing.js`
- [ ] 기능:
  - 전처리 파라미터 설정 폼
  - 미리보기 기능
  - 설정 저장/불러오기
  - 설정 프리셋 관리

### 2. 모델 선택 (3.8) - 1.5일

#### 2.1 백엔드 개발 (1일)
- [ ] 라우터 생성: `web/routers/admin/model.py`
- [ ] API 엔드포인트:
  - `GET /api/admin/models` - 사용 가능한 모델 목록
  - `GET /api/admin/models/current` - 현재 선택된 모델
  - `POST /api/admin/models/select` - 모델 선택
  - `GET /api/admin/models/{model_id}/info` - 모델 상세 정보
- [ ] 지원 모델:
  - CLIP: ViT-B-32, ViT-B-16, ViT-L-14
  - PatchCore: WideResNet50, ResNet18
- [ ] DB 테이블: `model_configs`
- [ ] 모델 전환 로직

#### 2.2 프론트엔드 개발 (0.5일)
- [ ] 화면 생성: `web/pages/admin/admin_model.html`
- [ ] CSS: `web/static/admin/css/admin_model.css`
- [ ] JavaScript: `web/static/admin/js/admin_model.js`
- [ ] 기능:
  - 모델 선택 UI
  - 모델 성능 비교 표
  - 현재 사용 중인 모델 표시
  - 모델 정보 표시 (파라미터 수, 정확도 등)

### 3. 통합 대시보드 (3.1) - 1일

#### 3.1 백엔드 개발 (0.5일)
- [ ] 라우터 생성: `web/routers/admin/dashboard.py`
- [ ] API 엔드포인트:
  - `GET /api/admin/dashboard/stats` - 전체 통계
  - `GET /api/admin/dashboard/products` - 제품별 현황
  - `GET /api/admin/dashboard/defects` - 불량 유형별 통계
  - `GET /api/admin/dashboard/recent` - 최근 활동 내역
- [ ] 집계 데이터:
  - 제품별 정상/불량 이미지 수
  - 전체 검사 건수
  - 불량 감지 건수
  - 조치 완료/미조치 건수

#### 3.2 프론트엔드 개발 (0.5일)
- [ ] 화면 생성: `web/pages/admin/admin_dashboard.html`
- [ ] CSS: `web/static/admin/css/admin_dashboard.css`
- [ ] JavaScript: `web/static/admin/js/admin_dashboard.js`
- [ ] 기능:
  - 주요 지표 카드 (총 제품 수, 검사 수, 불량 감지 수)
  - 제품별 데이터셋 현황 테이블
  - 불량 유형별 검출 현황 차트 (Chart.js)
  - 최근 조치 내역 리스트

---

## 세부 작업 내역

### Day 1: 이미지 전처리 설정 백엔드

**작업 항목:**
1. `web/routers/admin/preprocessing.py` 생성
   - 전처리 설정 CRUD API
   - 설정 검증 로직

2. DB 테이블 생성
   - `preprocessing_configs` 테이블
   - 스키마: id, name, resize_width, resize_height, normalize, augmentation, created_at

3. 전처리 로직 통합
   - 기존 모듈과 연동
   - 설정 적용 테스트

**예상 산출물:**
- `web/routers/admin/preprocessing.py`
- DB 마이그레이션 스크립트

---

### Day 2: 이미지 전처리 설정 프론트엔드 + 모델 선택 백엔드

**작업 항목:**
1. 전처리 설정 프론트엔드
   - `admin_preprocessing.html` 생성
   - 설정 폼 UI
   - 미리보기 기능

2. 모델 선택 백엔드
   - `web/routers/admin/model.py` 생성
   - 모델 목록/선택 API
   - DB 테이블 생성

**예상 산출물:**
- `web/pages/admin/admin_preprocessing.html`
- `web/static/admin/css/admin_preprocessing.css`
- `web/static/admin/js/admin_preprocessing.js`
- `web/routers/admin/model.py`

---

### Day 3: 모델 선택 프론트엔드 + 대시보드

**작업 항목:**
1. 모델 선택 프론트엔드
   - `admin_model.html` 생성
   - 모델 선택 UI
   - 모델 정보 표시

2. 대시보드 백엔드
   - `web/routers/admin/dashboard.py` 생성
   - 통계 데이터 API
   - 집계 쿼리 최적화

3. 대시보드 프론트엔드 (기본)
   - `admin_dashboard.html` 생성
   - 주요 지표 카드
   - 테이블 레이아웃

**예상 산출물:**
- `web/pages/admin/admin_model.html`
- `web/static/admin/css/admin_model.css`
- `web/static/admin/js/admin_model.js`
- `web/routers/admin/dashboard.py`
- `web/pages/admin/admin_dashboard.html` (부분)

---

### Day 4: 대시보드 완성 및 통합 테스트

**작업 항목:**
1. 대시보드 차트 구현
   - Chart.js 통합
   - 불량 유형별 차트
   - 시계열 그래프

2. 전체 통합 테스트
   - 모든 화면 연동 테스트
   - 네비게이션 메뉴 통합
   - 에러 처리 점검

3. 성능 최적화
   - API 응답 시간 측정
   - 쿼리 최적화

**예상 산출물:**
- 완성된 대시보드 화면
- 통합 테스트 결과
- 버그 수정 패치

---

## 기술 스택

### 백엔드
- **FastAPI**: RESTful API
- **MariaDB**: 데이터 저장
- **SQLAlchemy**: ORM

### 프론트엔드
- **HTML/CSS/JavaScript**: 기본 웹 기술
- **Chart.js**: 데이터 시각화
- **Fetch API**: RESTful API 호출

### 데이터베이스
- **preprocessing_configs**: 전처리 설정 저장
- **model_configs**: 모델 선택 설정 저장

---

## 주요 파일 목록

### 신규 생성 파일
```
web/
├── routers/admin/
│   ├── preprocessing.py           # 전처리 설정 API
│   ├── model.py                   # 모델 선택 API
│   └── dashboard.py               # 대시보드 API
├── pages/admin/
│   ├── admin_preprocessing.html   # 전처리 설정 화면
│   ├── admin_model.html           # 모델 선택 화면
│   └── admin_dashboard.html       # 대시보드 화면
├── static/admin/css/
│   ├── admin_preprocessing.css
│   ├── admin_model.css
│   └── admin_dashboard.css
└── static/admin/js/
    ├── admin_preprocessing.js
    ├── admin_model.js
    └── admin_dashboard.js
```

### 수정 파일
```
web/
├── api_server.py                  # 라우터 등록 및 페이지 서빙 추가
└── database/
    ├── models.py                  # 테이블 모델 추가
    └── crud.py                    # CRUD 함수 추가
```

---

## DB 테이블 스키마

### preprocessing_configs
```sql
CREATE TABLE IF NOT EXISTS preprocessing_configs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL COMMENT '설정 이름',
    resize_width INT DEFAULT 224 COMMENT '리사이즈 너비',
    resize_height INT DEFAULT 224 COMMENT '리사이즈 높이',
    normalize BOOLEAN DEFAULT TRUE COMMENT '정규화 여부',
    augmentation JSON NULL COMMENT '증강 설정',
    is_active BOOLEAN DEFAULT FALSE COMMENT '활성화 여부',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_active (is_active)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='전처리 설정';
```

### model_configs
```sql
CREATE TABLE IF NOT EXISTS model_configs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    model_type VARCHAR(50) NOT NULL COMMENT 'CLIP or PatchCore',
    model_name VARCHAR(100) NOT NULL COMMENT '모델 이름',
    model_path VARCHAR(255) NULL COMMENT '모델 경로',
    is_active BOOLEAN DEFAULT FALSE COMMENT '활성화 여부',
    parameters JSON NULL COMMENT '모델 파라미터',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_type_active (model_type, is_active)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='모델 설정';
```

---

## 환경 변수
```bash
# Database
DB_HOST=localhost
DB_PORT=3306
DB_USER=dmillion
DB_PASSWORD=your_password
DB_NAME=defect_db

# Chart.js CDN (프론트엔드에서 사용)
# <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
```

---

## 배포 전 체크리스트

- [ ] DB 테이블 생성 (preprocessing_configs, model_configs)
- [ ] 라우터 등록 확인 (api_server.py)
- [ ] 페이지 서빙 라우트 추가
- [ ] Chart.js CDN 연결 확인
- [ ] 기존 모듈과의 통합 테스트

---

## 성공 기준

### 기능 요구사항
- [ ] 전처리 설정 저장/불러오기 성공
- [ ] 모델 선택 및 전환 성공
- [ ] 대시보드 통계 데이터 정확성
- [ ] 차트 정상 표시

### 성능 요구사항
- [ ] 대시보드 로딩 시간: 2초 이내
- [ ] API 응답 시간: 500ms 이내
- [ ] 차트 렌더링: 1초 이내

### 안정성 요구사항
- [ ] 설정 변경 시 기존 데이터 유지
- [ ] 모델 전환 시 서비스 중단 없음
- [ ] 에러 발생 시 명확한 메시지

---

**작성일**: 2025-11-14  
**작성자**: Claude  
**버전**: 2.0  
**이전 완료**: Phase 2 (서버 배포 기능)  
**다음 작업**: Phase 3 (설정 및 모니터링 기능)
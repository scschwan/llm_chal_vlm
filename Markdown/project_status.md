# 유사이미지 검색 솔루션 개발 현황 (업데이트)

## 프로젝트 개요
- **목표**: CLIP 기반 유사이미지 검색 → PatchCore Anomaly Detection → LLM 대응 매뉴얼 생성
- **환경**: Naver Cloud Platform, Rocky Linux 8.10
- **아키텍처**: 분리된 서버 구조 (API Server + LLM Server)

## 시스템 아키텍처

### 1. 서버 구조
```
┌─────────────────────────────────────────────┐
│         웹 클라이언트 (브라우저)              │
│  - matching.html (작업자 화면)               │
│  - dashboard.html (관리자 화면)              │
│  - defect_analysis.html (불량 분석)          │
└─────────────────────────────────────────────┘
                    ↓ HTTP
┌─────────────────────────────────────────────┐
│       API Server (FastAPI - Port 5000)       │
│  - CLIP 유사도 검색                          │
│  - PatchCore Anomaly Detection              │
│  - 이미지 처리 및 ROI 추출                   │
│  - LLM Server로 프롬프트 전달 (콜백)        │
└─────────────────────────────────────────────┘
                    ↓ HTTP (내부 통신)
┌─────────────────────────────────────────────┐
│       LLM Server (FastAPI - Port 5001)       │
│  - Python 3.10 전용 venv310 가상환경         │
│  - Hugging Face Transformers (LLaVA 등)    │
│  - 프롬프트 기반 텍스트 생성                 │
│  - 스트리밍 응답 (콜백)                      │
└─────────────────────────────────────────────┘
```

### 2. 통신 흐름
```
사용자 이미지 업로드
    ↓
API Server (5000)
    ├─ CLIP 유사 이미지 검색 (TOP-K)
    ├─ PatchCore Anomaly Detection 실행 (수동 트리거)
    ├─ ROI 추출 및 분석
    └─ 프롬프트 구성 → LLM Server (5001) 호출
                            ↓
                    LLM Server (5001)
                    └─ LLM 추론 → 응답 콜백
    ↓
API Server가 최종 응답 수신
    ↓
웹 클라이언트에 결과 전달
```

## 개발 환경

### API Server (web/api_server.py)
- **Python**: 3.9
- **포트**: 5000
- **주요 라이브러리**:
  - FastAPI
  - PyTorch (CUDA 지원)
  - CLIP (openai/ViT-B-32)
  - PatchCore (Anomaly Detection)
  - FAISS (벡터 검색)
  - OpenCV (이미지 처리)

### LLM Server (llm_server/llm_server.py)
- **Python**: 3.10
- **포트**: 5001
- **가상환경**: venv310 (독립 환경)
- **주요 라이브러리**:
```
accelerate==1.11.0
fastapi==0.111.0
transformers==4.45.2
torch==2.9.0
bitsandbytes==0.48.2
huggingface-hub==0.24.6
```

### LLM Server 주요 특징
1. **독립된 Python 3.10 환경**: API Server와 의존성 충돌 방지
2. **GPU 지원**: CUDA 12.8.x, cuDNN 9.10
3. **모델 로딩**: Hugging Face Transformers 기반
4. **비동기 처리**: FastAPI 비동기 엔드포인트
5. **스트리밍 응답**: 콜백 방식으로 실시간 응답 전달

## 주요 기능 모듈

### 1. TOP-K 유사도 매칭 (`modules/clip_search.py`)
- **상태**: ✅ 완료
- **기능**:
  - CLIP 기반 이미지 임베딩
  - FAISS 인덱스 활용 고속 검색
  - TOP-K 유사 이미지 반환
  - 유사도 스코어 제공

### 2. Anomaly Detection (`modules/patchcore_module.py`)
- **상태**: ✅ 완료
- **기능**:
  - PatchCore 기반 이상 탐지
  - 제품별 메모리 뱅크 (prod1, prod2, prod3)
  - 수동 트리거 방식 (자동 실행 없음)
  - 정상 참조 이미지 자동 선택
  - 히트맵 생성 및 시각화

### 3. LLM 매뉴얼 생성 (`llm_server/llm_server.py`)
- **상태**: 🔄 개발중
- **기능**:
  - RAG 기반 매뉴얼 검색 (예정)
  - LLaVA 또는 오픈소스 LLM 추론
  - 프롬프트 기반 응답 생성
  - 스트리밍 방식 응답

## 데이터 구조

### 이미지 데이터 디렉토리
```
data/
├── prod1/               # 제품1
│   ├── normal/         # 정상 이미지
│   └── defect/         # 불량 이미지
├── prod2/               # 제품2
│   ├── normal/
│   └── defect/
└── prod3/               # 제품3
    ├── normal/
    └── defect/
```

### 이미지 명명 규칙
```
{제품명}_{불량타입}_{순번}.jpg

예시:
prod1_scratch_001.jpg
prod2_dent_005.jpg
prod3_normal_010.jpg
```

## 웹 인터페이스

### 작업자 화면 (matching.html)
- **기능**:
  - 이미지 업로드
  - TOP-K 유사 이미지 표시
  - Anomaly Detection 수동 실행
  - 불량 유형 등록
  - 결과 확인

### 관리자 화면 (dashboard.html)
- **기능**:
  - 제품/불량 유형 관리
  - 통계 대시보드
  - 데이터셋 관리
  - 시스템 설정

## NCP 인프라 정보

### 네트워크 구성
- **VPC**: dm-vpc (10.200.0.0/16)
- **서브넷**:
  - Public: dm-pub-sub (10.200.0.0/24)
  - LB: dm-lb-sub (10.200.1.0/24)
  - NAT: dm-nat-sub (10.200.2.0/24)
  - Private: dm-pri-sub (10.200.3.0/24)

### 접속 정보
- **GPU 서버 SSH**: `ssh -p 2022 root@dm-nlb-112319415-f8e0a97d0b99.kr.lb.naverncp.com`
- **ALB 주소**: `http://dm-alb-112319279-991b4e0889c4.kr.lb.naverncp.com:80`
- **서비스 포트**: 
  - API Server: 5000 → ALB 8080 → 외부 80
  - LLM Server: 5001 (내부 전용)
- **헬스체크**: `/health` (포트 8080)

### Object Storage
- **버킷명**: dm-obs
- **VPC 사설 도메인**: kr.object.private.ncloudstorage.com
- **접근 제어**: GPU 서버만 접근 가능

## 데이터베이스 설계

### 핵심 테이블 (10개)
1. **users**: 사용자 계정 관리
2. **products**: 제품 정보
3. **manuals**: 대응 매뉴얼 문서
4. **defect_types**: 불량 유형 정의
5. **images**: 이미지 메타데이터
6. **search_history**: 검색 이력
7. **response_history**: LLM 응답 이력
8. **model_parameters**: 모델 파라미터 설정
9. **deployment_logs**: 배포 로그
10. **system_config**: 시스템 설정

## 현재 개발 상태

### ✅ 완료된 작업
- [x] CLIP 유사도 검색 모듈
- [x] PatchCore Anomaly Detection 모듈
- [x] FastAPI 웹 서버 구축 (5000, 5001)
- [x] 작업자 웹 인터페이스
- [x] 관리자 대시보드 (기본)
- [x] 제품별 메모리 뱅크 구축
- [x] NCP 인프라 배포

### 🔄 진행중
- [ ] LLM Server 완전한 통합
- [ ] RAG 기반 매뉴얼 검색 구현
- [ ] 데이터베이스 마이그레이션
- [ ] 관리자 기능 확장

### ⏳ 예정
- [ ] 배치 처리 (다중 이미지, ZIP 파일)
- [ ] 통계 대시보드 고도화
- [ ] 모바일 반응형 UI
- [ ] 사용자 권한 관리

## 개발 가이드

### API Server 실행
```bash
cd /path/to/llm_chal_vlm/web
python3.9 api_server.py
```

### LLM Server 실행
```bash
cd /path/to/llm_chal_vlm/llm_server
source venv310/bin/activate
python llm_server.py
```

### 의존성 설치
```bash
# API Server (Python 3.9)
pip install -r web/requirements.txt

# LLM Server (Python 3.10 - venv310)
cd llm_server
python3.10 -m venv venv310
source venv310/bin/activate
pip install fastapi transformers torch accelerate bitsandbytes
```

## 문제 해결 이력

### PyTorch CUDA 이슈
- **문제**: CUDA 호환성 오류
- **해결**: GPU 지원 PyTorch 재설치

### FP16 정밀도 오류
- **문제**: PatchCore FP16 연산 오류
- **해결**: FP16 비활성화

### ALB 로드밸런서 연동
- **문제**: localhost 직접 접근 불가
- **해결**: 상대 경로 URL 사용

### 경로 문제
- **해결**: 절대 경로로 통일

## 참고 문서
- [시스템 아키텍처](./system_architecture.md)
- [데이터베이스 스키마](./database_schema.md)
- [TOP-K 테스트 가이드](./TOP-K%20TEST_GUIDE.md)
- [PatchCore 가이드](./readme_patchcore.md)
- [세션 인수인계](./session_handover.md)
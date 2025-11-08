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
  1. TOP-K 유사도 매칭
  2. 특징점 추출 (YOLO/Anomaly Detection)
  3. VLM/LLM 대응 매뉴얼 생성
```

## 임시 테스트 환경
- **web 폴더**: 임시로 만든 HTML + FastAPI 기반 테스트 페이지
- **용도**: 웹 개발자의 WAS가 완성되기 전까지 기능 검증용

## 개발 모듈 (3개)

### 1. TOP-K 유사도 매칭 모듈
- **기반**: 기존 `modules/clip_search.py` 활용 및 개선
- **기능**: 
  - CLIP 기반 이미지 임베딩
  - FAISS 인덱스 활용 고속 검색
  - TOP-K 유사 이미지 반환
- **출력**: JSON 형태로 유사 이미지 경로 및 유사도 스코어

### 2. 특징점 추출 모듈
- **방법**: YOLO Segmentation 또는 Anomaly Detection
- **기능**:
  - 입력 이미지와 유사 이미지 간 차이 영역 검출
  - ROI(Region of Interest) 추출
  - Bounding Box 또는 Segmentation Mask 생성
- **출력**: JSON 형태로 ROI 좌표 및 차이 정보

### 3. VLM/LLM 대응 매뉴얼 생성 모듈
- **방법**: RAG (Retrieval-Augmented Generation)
- **기능**:
  - PDF 매뉴얼 벡터 DB 구축
  - 차이점 기반 관련 매뉴얼 검색
  - LLaVA 또는 오픈소스 LLM으로 대응 방안 생성
- **출력**: JSON 형태로 대응 매뉴얼 텍스트

## 기존 코드베이스 분석 결과

### 핵심 모듈
1. **modules/clip_search.py**: CLIP 기반 검색 엔진 (완성도 높음)
2. **modules/region_detect.py**: OpenCV 기반 차이 검출
3. **modules/evaluate_yolo_seg.py**: YOLO Segmentation 평가
4. **modules/vlm_local.py**: VLM 래퍼 (플레이스홀더)

### 개선 필요 사항
1. **clip_search.py**: 인덱스 저장/로드 기능 추가 필요
2. **region_detect.py**: `boxes_to_prompt_hints()` 버그 수정 필요
3. **vlm_local.py**: 실제 LLaVA 구현 필요

## 개발 우선순위
1. ✅ TOP-K 유사도 매칭 모듈 (진행중)
2. ⏳ 특징점 추출 모듈
3. ⏳ VLM/LLM 대응 매뉴얼 생성 모듈

## NCP 인프라 정보
- **GPU 서버 접속**: `ssh -p 2022 root@dm-nlb-112319415-f8e0a97d0b99.kr.lb.naverncp.com`
- **ALB 주소**: `http://dm-alb-112319279-991b4e0889c4.kr.lb.naverncp.com:80`
- **서비스 포트**: 8080 (헬스체크: `/health`)
- **Object Storage**: `dm-obs` (VPC 전용, 사설 도메인 사용)

## 다음 단계
1. TOP-K 유사도 모듈 완성 및 테스트
2. FastAPI 엔드포인트 구현
3. 특징점 추출 모듈 개발 시작

## 개발 현황 업데이트 (2025-11-08)

### 완료된 사항
- TOP-K 유사도 매칭 모듈 구축 및 배포 완료
- FastAPI 서버 NCP 환경에서 정상 작동 확인
- ALB 로드밸런싱 설정 완료 및 헬스체크 통과

### 기술 결정 사항
- **API 서버 포트**: 5000번 포트 사용 (의도적 변경)
  - 이유: 8080 포트는 ALB 헬스체크 전용
  - 5000 포트로 서비스 구동 후 ALB를 통해 외부 접근
- **헬스체크 엔드포인트**: `/health2` 사용
  - FastAPI 서버에서 인덱스 상태 확인 포함

### 진행 중
- Anomaly Detection 모듈 통합 및 테스트
  - 경로 이슈 해결 중: 상대경로 → 절대경로 변경
  - 제품명 추출 로직 개선 필요
  - 메모리뱅크 경로: `/home/dmillion/llm_chal_vlm/data/patchCore`
  - 사용 가능 제품: prod1, prod2, prod3

### 다음 단계
1. Anomaly Detection 제품명 자동 추출 로직 강화
2. matching.html에 제품 선택 UI 추가
3. 전체 파이프라인 통합 테스트
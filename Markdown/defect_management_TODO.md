# 불량 관리 시스템 개발 계획

## 개요
불량 이미지 등록 및 제품/불량 설정을 관리하는 웹 인터페이스가 필요합니다.

## 현재 상태
- `web/defect_config.json`: 제품별 불량 목록 설정 파일
- matching.html: 불량 이미지 등록 기능 구현 완료
- api_server.py: `/register_defect` 엔드포인트 구현 완료

## TODO: 설정 관리 페이지 개발

### 1. defect_config_manager.html 생성
**기능:**
- 제품 추가/수정/삭제
- 불량 유형 추가/수정/삭제
- 제품별 불량 목록 관리
- JSON 파일 저장/로드

### 2. API 엔드포인트 추가 (api_server.py)
```python
@app.get("/config/defects")  # 설정 조회
@app.post("/config/defects")  # 설정 저장
@app.put("/config/products/{product_id}")  # 제품 수정
@app.delete("/config/products/{product_id}")  # 제품 삭제
```

### 3. UI 요구사항
- 제품 목록 테이블
- 각 제품별 불량 유형 관리
- Drag & Drop으로 순서 변경
- 실시간 미리보기

### 4. 보안 고려사항
- 관리자 인증 필요
- 설정 변경 이력 로깅
- 백업 기능

## 우선순위
- Phase 1: 기본 CRUD 기능 (현재 필요)
- Phase 2: 이력 관리 및 백업
- Phase 3: 권한 관리

## 연관 파일
- `web/defect_config.json`: 설정 데이터
- `web/matching.html`: 설정 사용하는 메인 UI
- `web/api_server.py`: 백엔드 API
# 유사도 검색 시스템 테스트 가이드

## 준비 사항

### 1. 파일 배치
```bash
# 프로젝트 구조
llm_chal_vlm/
├── modules/
│   └── similarity_matcher.py    # 새로 추가
├── web/
│   ├── api_server.py           # 업데이트
│   └── matching.html           # 업데이트
└── data/
    └── ok_front/               # 갤러리 이미지 (정상 이미지)
```

### 2. 필요 패키지 설치
```bash
pip install torch torchvision open_clip_torch pillow faiss-cpu fastapi uvicorn python-multipart
```

## 테스트 단계

### STEP 1: 인덱스 구축

인덱스를 먼저 구축해야 검색이 가능합니다.

```bash
# Python으로 직접 구축
cd /root/llm_chal_vlm

python3 << 'EOF'
import sys
sys.path.insert(0, '.')

from modules.similarity_matcher import create_matcher

# 매처 생성
matcher = create_matcher(
    model_id="ViT-B-32/openai",
    device="cuda",  # GPU 사용 (없으면 "cpu")
    use_fp16=True,
    verbose=True
)

# 인덱스 구축 (정상 이미지 디렉토리 지정)
matcher.build_index("./data/ok_front")

# 인덱스 저장
matcher.save_index("./web/index_cache")

print("✅ 인덱스 구축 완료!")
EOF
```

### STEP 2: API 서버 실행

```bash
cd /root/llm_chal_vlm/web

# 서버 실행
python3 api_server.py
```

서버가 정상 실행되면 다음 메시지가 표시됩니다:
```
==================================================
TOP-K 유사도 매칭 API 서버 시작
==================================================
✅ 기존 인덱스 로드 완료: XXX개 이미지
==================================================
INFO:     Uvicorn running on http://0.0.0.0:8080
```

### STEP 3: 웹 브라우저에서 테스트

1. **브라우저 열기**
   ```
   http://your-server-ip:8080/matching.html
   ```
   또는 로컬에서:
   ```
   http://localhost:8080/matching.html
   ```

2. **이미지 업로드**
   - "📸" 영역 클릭 또는 이미지를 드래그하여 업로드

3. **검색 실행**
   - "검색할 유사 이미지 개수" 슬라이더로 TOP-K 설정 (1~20)
   - "🔍 유사 이미지 검색" 버튼 클릭

4. **결과 확인**
   - 우측에 유사 이미지들이 표시됨
   - TOP-1은 큰 카드로 표시
   - 각 이미지의 유사도 점수 확인

## API 직접 테스트 (curl)

### 헬스체크
```bash
curl http://localhost:8080/health
```

### 인덱스 구축 (API로)
```bash
curl -X POST http://localhost:8080/build_index \
  -H "Content-Type: application/json" \
  -d '{
    "gallery_dir": "/root/llm_chal_vlm/data/ok_front",
    "save_index": true,
    "index_save_dir": "./index_cache"
  }'
```

### 이미지 업로드 검색
```bash
curl -X POST "http://localhost:8080/search/upload?top_k=5" \
  -F "file=@/path/to/test_image.jpg"
```

### 인덱스 정보 조회
```bash
curl http://localhost:8080/index/info
```

## 트러블슈팅

### 1. "인덱스가 구축되지 않았습니다" 오류
**원인**: 인덱스를 먼저 구축하지 않음
**해결**: STEP 1을 먼저 실행

### 2. "매처가 초기화되지 않았습니다" 오류
**원인**: API 서버가 제대로 시작되지 않음
**해결**: 서버 로그 확인 및 재시작

### 3. "API 서버에 연결할 수 없습니다" 경고
**원인**: API 서버가 실행되지 않았거나 포트가 다름
**해결**: 
- API 서버 실행 확인
- matching.html의 `API_BASE_URL` 수정

### 4. 이미지가 표시되지 않음
**원인**: 이미지 경로 접근 권한 문제
**해결**: 
- 갤러리 이미지 경로 확인
- 파일 권한 확인 (`chmod` 사용)

### 5. CUDA Out of Memory
**원인**: GPU 메모리 부족
**해결**:
```python
# similarity_matcher.py에서
matcher = create_matcher(
    device="cpu",  # CPU로 전환
    use_fp16=False
)
```

## 성능 확인

### 검색 속도
- FAISS + GPU: 수백~수천 이미지에서 < 100ms
- FAISS + CPU: < 500ms
- Torch (FAISS 없음): < 1초

### 메모리 사용량
- 모델 로드: ~2GB (GPU) / ~1GB (CPU)
- 인덱스 (1000 이미지): ~200MB

## 다음 단계

✅ TOP-K 유사도 검색 완료

다음 개발:
1. 특징점 추출 모듈 (YOLO/Anomaly Detection)
2. VLM/LLM 대응 매뉴얼 생성 모듈

## 문제 발생 시 체크리스트

- [ ] Python 3.9 이상 설치 확인
- [ ] 필요 패키지 모두 설치됨
- [ ] data/ok_front 디렉토리에 이미지 존재
- [ ] 인덱스가 구축됨 (index_cache 디렉토리 존재)
- [ ] API 서버가 8080 포트에서 실행 중
- [ ] 방화벽에서 8080 포트 허용됨
- [ ] GPU 사용 시 CUDA 정상 작동

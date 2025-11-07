# TOP-K 유사도 매칭 모듈 사용 가이드

## 개요
CLIP 기반 이미지 유사도 검색 모듈 및 FastAPI 서버

## 파일 구성
```
similarity_matcher.py  # 핵심 검색 모듈
api_server.py          # FastAPI REST API 서버
```

## 1. 모듈 직접 사용 (Python)

### 설치 필요 패키지
```bash
pip install torch torchvision open_clip_torch pillow faiss-cpu
```

### 기본 사용법
```python
from similarity_matcher import create_matcher

# 1. 매처 생성
matcher = create_matcher(
    model_id="ViT-B-32/openai",
    device="cuda",  # 또는 "cpu"
    use_fp16=True,
    verbose=True
)

# 2. 인덱스 구축 (최초 1회)
matcher.build_index("./data/ok_front")

# 3. 인덱스 저장 (선택)
matcher.save_index("./index_cache")

# 4. 검색
result = matcher.search("./data/def_front/sample.jpg", top_k=5)

# 5. 결과 확인
print(result.to_json())
```

### 인덱스 재사용
```python
# 저장된 인덱스 로드
matcher.load_index("./index_cache")

# 바로 검색 가능
result = matcher.search(query_image, top_k=5)
```

## 2. API 서버 사용

### 서버 실행
```bash
# 개발 모드 (자동 재시작)
python api_server.py

# 프로덕션 모드
uvicorn api_server:app --host 0.0.0.0 --port 8080 --workers 4
```

### API 엔드포인트

#### 1. 헬스체크
```bash
GET /health
```

**응답 예시:**
```json
{
  "status": "healthy",
  "message": "API 서버가 정상 작동 중입니다",
  "index_built": true,
  "gallery_size": 1500
}
```

#### 2. 인덱스 구축
```bash
POST /build_index
Content-Type: application/json

{
  "gallery_dir": "/path/to/ok_images",
  "save_index": true,
  "index_save_dir": "./index_cache"
}
```

**응답 예시:**
```json
{
  "status": "success",
  "gallery_dir": "/path/to/ok_images",
  "num_images": 1500,
  "embedding_dim": 512,
  "faiss_enabled": true,
  "index_saved": true,
  "index_save_path": "./index_cache"
}
```

#### 3. 이미지 경로로 검색
```bash
POST /search
Content-Type: application/json

{
  "query_image_path": "/path/to/defect.jpg",
  "top_k": 5
}
```

**응답 예시:**
```json
{
  "status": "success",
  "query_image": "/path/to/defect.jpg",
  "top_k_results": [
    {
      "rank": 1,
      "image_path": "/path/to/ok_images/ok_001.jpg",
      "image_name": "ok_001.jpg",
      "similarity_score": 0.9523
    },
    {
      "rank": 2,
      "image_path": "/path/to/ok_images/ok_042.jpg",
      "image_name": "ok_042.jpg",
      "similarity_score": 0.9401
    }
  ],
  "total_gallery_size": 1500,
  "model_info": "ViT-B-32/openai"
}
```

#### 4. 파일 업로드로 검색
```bash
POST /search/upload
Content-Type: multipart/form-data

file: [이미지 파일]
top_k: 5
```

**curl 예시:**
```bash
curl -X POST "http://localhost:8080/search/upload?top_k=5" \
  -F "file=@defect_image.jpg"
```

#### 5. 인덱스 정보 조회
```bash
GET /index/info
```

**응답 예시:**
```json
{
  "status": "index_built",
  "gallery_size": 1500,
  "model_id": "ViT-B-32/openai",
  "device": "cuda",
  "faiss_enabled": true,
  "sample_paths": [
    "/path/to/ok_images/ok_001.jpg",
    "/path/to/ok_images/ok_002.jpg"
  ]
}
```

## 3. Spring AI에서 호출 예시 (Java)

### RestTemplate 사용
```java
import org.springframework.web.client.RestTemplate;
import org.springframework.http.*;
import org.springframework.core.io.FileSystemResource;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;

public class SimilaritySearchClient {
    
    private final RestTemplate restTemplate;
    private final String apiBaseUrl = "http://localhost:8080";
    
    public SimilaritySearchClient() {
        this.restTemplate = new RestTemplate();
    }
    
    // 이미지 경로로 검색
    public SearchResponse searchByPath(String imagePath, int topK) {
        String url = apiBaseUrl + "/search";
        
        SearchRequest request = new SearchRequest(imagePath, topK);
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        
        HttpEntity<SearchRequest> entity = new HttpEntity<>(request, headers);
        
        ResponseEntity<SearchResponse> response = restTemplate.postForEntity(
            url, entity, SearchResponse.class
        );
        
        return response.getBody();
    }
    
    // 파일 업로드로 검색
    public SearchResponse searchByUpload(File imageFile, int topK) {
        String url = apiBaseUrl + "/search/upload?top_k=" + topK;
        
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);
        
        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("file", new FileSystemResource(imageFile));
        
        HttpEntity<MultiValueMap<String, Object>> entity = 
            new HttpEntity<>(body, headers);
        
        ResponseEntity<SearchResponse> response = restTemplate.postForEntity(
            url, entity, SearchResponse.class
        );
        
        return response.getBody();
    }
}
```

## 4. 성능 최적화 팁

### GPU 사용
```python
# CUDA 사용 (권장)
matcher = create_matcher(device="cuda", use_fp16=True)
```

### 인덱스 사전 구축
```bash
# 서버 시작 전 인덱스 미리 구축
python -c "
from similarity_matcher import create_matcher
matcher = create_matcher()
matcher.build_index('./data/ok_front')
matcher.save_index('./index_cache')
"
```

### 멀티 워커 배포
```bash
# Gunicorn 사용 (프로덕션)
gunicorn api_server:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8080 \
  --timeout 120
```

## 5. NCP 배포 가이드

### systemd 서비스 등록
```bash
# /etc/systemd/system/similarity-api.service
[Unit]
Description=Similarity Matching API
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/llm_chal_vlm
ExecStart=/usr/bin/python3.9 /root/llm_chal_vlm/api_server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# 서비스 등록 및 시작
sudo systemctl daemon-reload
sudo systemctl enable similarity-api
sudo systemctl start similarity-api
sudo systemctl status similarity-api
```

### nginx 리버스 프록시 (선택)
```nginx
# /etc/nginx/conf.d/similarity-api.conf
server {
    listen 8080;
    server_name localhost;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 300s;
    }
}
```

## 6. 트러블슈팅

### CUDA Out of Memory
```python
# FP16 사용 + 배치 크기 줄이기
matcher = create_matcher(use_fp16=True)
```

### FAISS 설치 실패
```bash
# CPU 버전 설치
pip install faiss-cpu

# GPU 버전 (CUDA 11.x)
pip install faiss-gpu
```

### 인덱스 로드 실패
```python
# 모델 ID가 일치하는지 확인
# 저장 시와 로드 시 동일한 model_id 사용
```

## 7. 다음 단계

현재 완료: ✅ TOP-K 유사도 매칭 모듈

다음 개발 예정:
- 특징점 추출 모듈 (YOLO/Anomaly Detection)
- VLM/LLM 대응 매뉴얼 생성 모듈

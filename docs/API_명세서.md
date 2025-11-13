# 제조 불량 검출 시스템 API 명세서

**버전**: 3.0.0  
**작성일**: 2025-01-15  
**Base URL**: `http://{server_ip}:5000`

---

## 목차

1. [이미지 업로드 API](#1-이미지-업로드-api)
2. [유사도 검색 API](#2-유사도-검색-api)
3. [이상 검출 API](#3-이상-검출-api)
4. [매뉴얼 생성 API](#4-매뉴얼-생성-api)
5. [관리자 기능 API](#5-관리자-기능-api)
6. [시스템 API](#6-시스템-api)

---

## 1. 이미지 업로드 API

### 1.1 이미지 업로드

**Endpoint**: `POST /upload/image`

**설명**: 검사할 이미지를 서버에 업로드합니다.

**Request**:
- Content-Type: `multipart/form-data`

| 파라미터 | 타입 | 필수 | 설명 |
|---------|------|------|------|
| file | File | O | 이미지 파일 (jpg, jpeg, png, webp, bmp) |

**Response**:
```json
{
  "status": "success",
  "filename": "test_image.jpg",
  "file_path": "/path/to/uploads/test_image.jpg",
  "file_size": 1048576,
  "file_size_mb": 1.0,
  "message": "파일 업로드 완료"
}
```

**Response Fields**:
| 필드 | 타입 | 설명 |
|------|------|------|
| status | string | 처리 상태 (success/error) |
| filename | string | 업로드된 파일명 |
| file_path | string | 서버 내 파일 경로 |
| file_size | integer | 파일 크기 (bytes) |
| file_size_mb | float | 파일 크기 (MB) |
| message | string | 처리 메시지 |

**Error Response**:
```json
{
  "detail": "지원하지 않는 파일 형식입니다. 허용: .jpg, .jpeg, .png, .webp, .bmp"
}
```

---

### 1.2 업로드 파일 목록 조회

**Endpoint**: `GET /upload/list`

**설명**: 현재 업로드된 파일 목록을 조회합니다.

**Request**: 없음

**Response**:
```json
{
  "status": "success",
  "files": [
    {
      "filename": "test_image.jpg",
      "file_path": "/path/to/uploads/test_image.jpg",
      "file_size": 1048576,
      "modified_at": 1705123456.789
    }
  ],
  "total_count": 1
}
```

**Response Fields**:
| 필드 | 타입 | 설명 |
|------|------|------|
| status | string | 처리 상태 |
| files | array | 파일 목록 |
| files[].filename | string | 파일명 |
| files[].file_path | string | 파일 경로 |
| files[].file_size | integer | 파일 크기 (bytes) |
| files[].modified_at | float | 수정 시각 (Unix timestamp) |
| total_count | integer | 전체 파일 개수 |

---

### 1.3 업로드 디렉토리 정리

**Endpoint**: `DELETE /upload/clean`

**설명**: 업로드된 모든 파일을 삭제합니다.

**Request**: 없음

**Response**:
```json
{
  "status": "success",
  "deleted_count": 5,
  "message": "5개 파일 삭제 완료"
}
```

---

### 1.4 특정 파일 삭제

**Endpoint**: `DELETE /upload/file/{filename}`

**설명**: 특정 파일을 삭제합니다.

**Request**:
| 파라미터 | 타입 | 필수 | 설명 |
|---------|------|------|------|
| filename | string | O | 삭제할 파일명 (URL Path) |

**Response**:
```json
{
  "status": "success",
  "filename": "test_image.jpg",
  "message": "파일 삭제 완료"
}
```

---

## 2. 유사도 검색 API

### 2.1 유사 이미지 검색

**Endpoint**: `POST /search/similarity`

**설명**: CLIP 기반 유사도 검색으로 불량 이미지 데이터베이스에서 TOP-K 유사 이미지를 검색합니다.

**Request**:
```json
{
  "query_image_path": "uploads/test_image.jpg",
  "top_k": 5
}
```

**Request Fields**:
| 필드 | 타입 | 필수 | 기본값 | 설명 |
|------|------|------|--------|------|
| query_image_path | string | O | - | 쿼리 이미지 경로 |
| top_k | integer | X | 5 | 반환할 결과 개수 (1~20) |

**Response**:
```json
{
  "status": "success",
  "query_image": "/path/to/test_image.jpg",
  "top_k_results": [
    {
      "image_path": "/path/to/prod1_hole_001.jpg",
      "similarity": 0.9823,
      "rank": 1,
      "product": "prod1",
      "defect": "hole",
      "sequence": "001"
    }
  ],
  "total_gallery_size": 2083,
  "model_info": {
    "model_id": "ViT-B-32/openai",
    "device": "cuda"
  }
}
```

**Response Fields**:
| 필드 | 타입 | 설명 |
|------|------|------|
| status | string | 처리 상태 |
| query_image | string | 쿼리 이미지 경로 |
| top_k_results | array | 검색 결과 배열 |
| top_k_results[].image_path | string | 유사 이미지 경로 |
| top_k_results[].similarity | float | 유사도 점수 (0~1) |
| top_k_results[].rank | integer | 순위 |
| top_k_results[].product | string | 제품명 |
| top_k_results[].defect | string | 불량 유형 |
| top_k_results[].sequence | string | 시퀀스 번호 |
| total_gallery_size | integer | 전체 갤러리 크기 |
| model_info | object | 모델 정보 |

**비고**:
- 자동으로 불량 이미지 인덱스로 전환하여 검색합니다.
- 이미지 파일명 규칙: `{제품명}_{불량유형}_{시퀀스}.jpg`

---

### 2.2 검색 인덱스 상태 조회

**Endpoint**: `GET /search/index/status`

**설명**: 현재 로드된 검색 인덱스의 상태를 조회합니다.

**Request**: 없음

**Response**:
```json
{
  "status": "success",
  "index_built": true,
  "gallery_count": 2083,
  "index_type": "defect"
}
```

**Response Fields**:
| 필드 | 타입 | 설명 |
|------|------|------|
| status | string | 처리 상태 |
| index_built | boolean | 인덱스 구축 여부 |
| gallery_count | integer | 인덱스 내 이미지 개수 |
| index_type | string | 인덱스 유형 (defect/normal) |

---

## 3. 이상 검출 API

### 3.1 이상 검출 수행

**Endpoint**: `POST /anomaly/detect`

**설명**: PatchCore 기반 이상 검출을 수행하고 결과 이미지(마스크, 오버레이)를 생성합니다.

**Request**:
```json
{
  "test_image_path": "uploads/test_image.jpg",
  "product_name": "prod1",
  "top1_defect_image": "data/def_split/prod1_hole_001.jpg"
}
```

**Request Fields**:
| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| test_image_path | string | O | 검사 이미지 경로 |
| product_name | string | O | 제품명 |
| top1_defect_image | string | X | TOP-1 불량 이미지 경로 (표시용) |

**Response**:
```json
{
  "status": "success",
  "product_name": "prod1",
  "image_score": 0.8542,
  "pixel_tau": 0.90,
  "image_tau": 0.85,
  "is_anomaly": true,
  "reference_normal_path": "data/patchCore/prod1/normal_001.jpg",
  "top1_defect_path": "data/def_split/prod1_hole_001.jpg",
  "mask_url": "/anomaly/image/test_image/mask.png",
  "overlay_url": "/anomaly/image/test_image/overlay.png",
  "comparison_url": "/anomaly/image/test_image/comparison.png"
}
```

**Response Fields**:
| 필드 | 타입 | 설명 |
|------|------|------|
| status | string | 처리 상태 |
| product_name | string | 제품명 |
| image_score | float | 이상 점수 (0~1) |
| pixel_tau | float | 픽셀 임계값 |
| image_tau | float | 이미지 임계값 |
| is_anomaly | boolean | 이상 판정 여부 |
| reference_normal_path | string | 정상 기준 이미지 경로 |
| top1_defect_path | string | TOP-1 불량 이미지 경로 |
| mask_url | string | 마스크 이미지 URL |
| overlay_url | string | 오버레이 이미지 URL |
| comparison_url | string | 비교 이미지 URL |

**비고**:
- 자동으로 정상 이미지 인덱스로 전환하여 검출합니다.
- 정상 기준 이미지는 유사도 매칭으로 자동 선택됩니다.

---

### 3.2 결과 이미지 조회

**Endpoint**: `GET /anomaly/image/{result_id}/{filename}`

**설명**: 이상 검출 결과 이미지를 조회합니다.

**Request**:
| 파라미터 | 타입 | 필수 | 설명 |
|---------|------|------|------|
| result_id | string | O | 결과 ID (이미지명) |
| filename | string | O | 파일명 (mask.png, overlay.png, comparison.png) |

**Response**: 이미지 파일 (PNG)

---

## 4. 매뉴얼 생성 API

### 4.1 대응 매뉴얼 생성

**Endpoint**: `POST /manual/generate`

**설명**: RAG 기반 매뉴얼 검색과 LLM 분석을 통해 대응 매뉴얼을 생성합니다.

**Request**:
```json
{
  "product": "prod1",
  "defect": "hole",
  "anomaly_score": 0.8542,
  "is_anomaly": true,
  "model_type": "hyperclovax",
  "image_path": "uploads/test_image.jpg"
}
```

**Request Fields**:
| 필드 | 타입 | 필수 | 기본값 | 설명 |
|------|------|------|--------|------|
| product | string | O | - | 제품명 |
| defect | string | O | - | 불량 유형 (영문명) |
| anomaly_score | float | O | - | 이상 점수 |
| is_anomaly | boolean | O | - | 이상 판정 여부 |
| model_type | string | X | hyperclovax | 모델 타입 (hyperclovax/exaone/llava) |
| image_path | string | X | null | 이미지 경로 (llava 사용 시 필수) |

**Response**:
```json
{
  "status": "success",
  "product": "prod1",
  "defect_en": "hole",
  "defect_ko": "기공",
  "full_name_ko": "주조 기공",
  "manual_context": {
    "원인": [
      "주조 시 금속 내부의 가스가 빠져나가지 못하고 갇혀 형성됨",
      "주조 온도가 너무 높거나 냉각 속도가 부적절한 경우"
    ],
    "조치": [
      "주조 온도를 10~15°C 낮춰 재시도",
      "탈기 처리 시간을 2분 연장",
      "냉각 속도를 점진적으로 조절"
    ]
  },
  "llm_analysis": "【현상 (What)】\n제품 표면에 직경 약 2mm 크기의 기공이 발견되었습니다...",
  "anomaly_score": 0.8542,
  "is_anomaly": true,
  "model_type": "hyperclovax",
  "processing_time": 3.45
}
```

**Response Fields**:
| 필드 | 타입 | 설명 |
|------|------|------|
| status | string | 처리 상태 |
| product | string | 제품명 |
| defect_en | string | 불량 유형 (영문) |
| defect_ko | string | 불량 유형 (한글) |
| full_name_ko | string | 불량 전체 명칭 |
| manual_context | object | RAG 검색 결과 |
| manual_context.원인 | array | 원인 목록 |
| manual_context.조치 | array | 조치 목록 |
| llm_analysis | string | LLM 분석 결과 (4개 섹션) |
| anomaly_score | float | 이상 점수 |
| is_anomaly | boolean | 이상 판정 |
| model_type | string | 사용한 모델 타입 |
| processing_time | float | 처리 시간 (초) |

**비고**:
- LLM 서버가 실행 중이어야 합니다 (http://localhost:5001)
- 모델별 특징:
  - `hyperclovax`: HyperCLOVAX (네이버)
  - `exaone`: EXAONE 3.5 (LG)
  - `llava`: LLaVA (VLM, 이미지 필수)

---

## 5. 관리자 기능 API

### 5.1 불량 이미지 등록

**Endpoint**: `POST /register_defect`

**설명**: 새로운 불량 이미지를 시스템에 등록하고 인덱스를 재구축합니다.

**Request**:
- Content-Type: `multipart/form-data`

| 파라미터 | 타입 | 필수 | 설명 |
|---------|------|------|------|
| file | File | O | 불량 이미지 파일 |
| product_name | string | O | 제품명 |
| defect_name | string | O | 불량 유형 |

**Response**:
```json
{
  "status": "success",
  "saved_path": "/path/to/data/def_split/prod1_hole_004.jpg",
  "filename": "prod1_hole_004.jpg",
  "product_name": "prod1",
  "defect_name": "hole",
  "seqno": 4,
  "index_rebuilt": true
}
```

**Response Fields**:
| 필드 | 타입 | 설명 |
|------|------|------|
| status | string | 처리 상태 |
| saved_path | string | 저장된 파일 경로 |
| filename | string | 생성된 파일명 |
| product_name | string | 제품명 |
| defect_name | string | 불량 유형 |
| seqno | integer | 시퀀스 번호 |
| index_rebuilt | boolean | 인덱스 재구축 여부 |

**비고**:
- 파일명 자동 생성: `{제품명}_{불량유형}_{시퀀스}.jpg`
- 시퀀스 번호는 자동으로 증가합니다.

---

### 5.2 매핑 상태 조회

**Endpoint**: `GET /mapping/status`

**설명**: 제품/불량 매핑 정보를 조회합니다.

**Request**: 없음

**Response**:
```json
{
  "status": "active",
  "products": {
    "prod1": {
      "defect_count": 3,
      "defects": ["hole", "burr", "scratch"]
    },
    "leather": {
      "defect_count": 6,
      "defects": ["hole", "burr", "scratch", "fold", "stain", "color"]
    }
  }
}
```

**Response Fields**:
| 필드 | 타입 | 설명 |
|------|------|------|
| status | string | 매핑 상태 (active/disabled) |
| products | object | 제품별 정보 |
| products.{product}.defect_count | integer | 불량 유형 개수 |
| products.{product}.defects | array | 불량 유형 목록 |

---

### 5.3 매핑 재로드

**Endpoint**: `POST /mapping/reload`

**설명**: defect_mapping.json 파일을 재로드합니다.

**Request**: 없음

**Response**:
```json
{
  "status": "success",
  "message": "매핑 파일 재로드 완료",
  "available_products": ["prod1", "grid", "carpet", "leather"]
}
```

**비고**:
- defect_mapping.json 파일 수정 후 호출해야 합니다.

---

## 6. 시스템 API

### 6.1 헬스체크

**Endpoint**: `GET /health2`

**설명**: API 서버 상태를 확인합니다 (ALB 헬스체크용).

**Request**: 없음

**Response**:
```json
{
  "status": "healthy",
  "message": "API 서버가 정상 작동 중입니다",
  "index_built": true,
  "gallery_size": 2083,
  "matcher_initialized": true,
  "detector_initialized": true
}
```

**Response Fields**:
| 필드 | 타입 | 설명 |
|------|------|------|
| status | string | 서버 상태 (healthy/unhealthy) |
| message | string | 상태 메시지 |
| index_built | boolean | 인덱스 구축 여부 |
| gallery_size | integer | 갤러리 크기 |
| matcher_initialized | boolean | 유사도 매처 초기화 여부 |
| detector_initialized | boolean | 이상 검출기 초기화 여부 |

---

## 부록

### A. 에러 코드

| HTTP 상태 코드 | 설명 |
|---------------|------|
| 200 | 정상 처리 |
| 400 | 잘못된 요청 (파라미터 오류) |
| 404 | 리소스를 찾을 수 없음 |
| 500 | 서버 내부 오류 |
| 503 | 외부 서비스 연결 실패 (LLM 서버 등) |

### B. 에러 응답 형식

```json
{
  "detail": "오류 메시지"
}
```

### C. 데이터 디렉토리 구조

```
data/
├── def_split/              # 불량 이미지
│   ├── prod1_hole_001.jpg
│   ├── prod1_burr_001.jpg
│   └── ...
├── patchCore/              # 정상 이미지 (제품별)
│   ├── prod1/
│   ├── grid/
│   ├── carpet/
│   └── leather/
└── ...
```

### D. 지원 모델

| 모델 | 타입 | 설명 |
|------|------|------|
| HyperCLOVAX | LLM | 네이버 하이퍼클로바X |
| EXAONE 3.5 | LLM | LG EXAONE 3.5 |
| LLaVA | VLM | 멀티모달 비전-언어 모델 |

### E. 환경 정보

- **서버 OS**: Rocky Linux 8.10
- **Python**: 3.9
- **GPU**: Tesla T4
- **포트**: 5000 (API), 5001 (LLM)

---

**문서 버전**: 1.0  
**최종 수정**: 2025-01-15

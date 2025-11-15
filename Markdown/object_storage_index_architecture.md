# Object Storage 기반 인덱스 구축 및 메타데이터 관리 아키텍처

## 1. 전체 아키텍처 개요

```
┌─────────────────────────────────────────────────────────────┐
│                    기존 코드 (유지)                          │
├─────────────────────────────────────────────────────────────┤
│ modules/similarity_matcher.py                               │
│ - TopKSimilarityMatcher (기존 클래스)                       │
│ - 로컬 파일 기반 인덱스 구축                                 │
│ - 파일명 파싱 기반 메타데이터                                │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                    신규 코드 (추가)                          │
├─────────────────────────────────────────────────────────────┤
│ modules/similarity_matcher_v2.py (NEW)                      │
│ - TopKSimilarityMatcherV2 (신규 클래스)                     │
│ - DB 기반 메타데이터 인덱스 구축                             │
│ - Object Storage URL 포함                                   │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                   API 라우터 수정                            │
├─────────────────────────────────────────────────────────────┤
│ web/routers/search.py                                       │
│ - 기존: _matcher_ref (TopKSimilarityMatcher)               │
│ - 신규: _matcher_v2_ref (TopKSimilarityMatcherV2)          │
│ - 환경변수/설정으로 버전 선택                                │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 신규 모듈 구조: `similarity_matcher_v2.py`

### 2.1 핵심 차이점

| 항목 | 기존 (V1) | 신규 (V2) |
|------|-----------|-----------|
| 메타데이터 소스 | 파일명 파싱 | DB 조회 |
| 이미지 경로 | 로컬 경로만 | 로컬 + Object Storage URL |
| 인덱스 구축 입력 | 디렉토리 경로 | DB 세션 + 이미지 타입 |
| 검색 결과 | 파일명 파싱 정보 | DB 메타데이터 전체 |

### 2.2 클래스 구조

```python
# modules/similarity_matcher_v2.py

class ImageMetadata:
    """이미지 메타데이터 구조체"""
    image_id: int
    local_path: str
    storage_url: str
    product_code: str
    product_name: str
    defect_code: str
    defect_name: str
    image_type: str

class TopKSimilarityMatcherV2:
    """
    DB 기반 메타데이터 + Object Storage 지원 매처
    """
    
    def __init__(self, ...):
        # 기존과 동일한 CLIP 모델 초기화
        pass
    
    def build_index_from_db(
        self, 
        db_session, 
        image_type: str  # 'normal' or 'defect'
    ) -> Dict:
        """
        DB에서 이미지 메타데이터 조회 → 인덱스 구축
        
        Process:
        1. DB에서 images 테이블 조회 (image_type 필터)
        2. 로컬 파일 존재 확인
        3. 배치 임베딩 생성
        4. 메타데이터와 함께 저장
        """
        pass
    
    def search(
        self, 
        query_image_path: str, 
        top_k: int = 5
    ) -> SearchResultV2:
        """
        검색 결과에 전체 메타데이터 포함
        
        Returns:
            SearchResultV2(
                results=[
                    {
                        "image_id": 123,
                        "similarity_score": 0.95,
                        "local_path": "/data/def_split/...",
                        "storage_url": "https://...",
                        "product_code": "prod1",
                        "product_name": "제품1",
                        "defect_code": "scratch",
                        "defect_name": "긁힘"
                    }
                ]
            )
        """
        pass
    
    def save_index(self, save_dir: Path):
        """인덱스 + 메타데이터 저장"""
        # gallery_metadata: List[ImageMetadata]
        pass
    
    def load_index(self, load_dir: Path):
        """인덱스 + 메타데이터 로드"""
        pass
```

---

## 3. 인덱스 구축 프로세스

### 3.1 서버 기동 시 자동 재구축

```python
# api_server.py

async def initialize_clip_indexes_v2():
    """
    서버 기동 시 DB 기반 인덱스 재구축
    """
    
    INDEX_DIR = Path("/home/dmillion/llm_chal_vlm/web/index_cache_v2")
    
    try:
        print("[STARTUP] DB 기반 인덱스 재구축 시작...")
        
        db = next(get_db())
        
        # 1. 불량 이미지 인덱스
        matcher_defect = TopKSimilarityMatcherV2(...)
        info = matcher_defect.build_index_from_db(
            db_session=db,
            image_type='defect'
        )
        matcher_defect.save_index(INDEX_DIR / "defect")
        
        # 2. 정상 이미지 인덱스
        matcher_normal = TopKSimilarityMatcherV2(...)
        info = matcher_normal.build_index_from_db(
            db_session=db,
            image_type='normal'
        )
        matcher_normal.save_index(INDEX_DIR / "normal")
        
        print(f"[STARTUP] ✅ 인덱스 재구축 완료")
        
    except Exception as e:
        print(f"[STARTUP] ⚠️ DB 조회 실패, 저장된 인덱스 로드 시도")
        
        # Fallback: 저장된 인덱스 로드
        if (INDEX_DIR / "defect" / "index_data.pt").exists():
            print("[STARTUP] ✅ 저장된 인덱스 로드 완료")
        else:
            raise RuntimeError("인덱스 파일 없음")
```

### 3.2 메타데이터 조회 최적화

```python
# modules/similarity_matcher_v2.py

def _fetch_image_metadata_from_db(
    db_session, 
    image_type: str
) -> List[ImageMetadata]:
    """
    DB에서 이미지 메타데이터 일괄 조회 (JOIN 최적화)
    
    SQL:
    SELECT 
        i.image_id,
        i.file_path as local_path,
        i.storage_url,
        p.product_code,
        p.product_name,
        COALESCE(d.defect_code, 'normal') as defect_code,
        COALESCE(d.defect_name_ko, '정상') as defect_name,
        i.image_type
    FROM images i
    INNER JOIN products p ON i.product_id = p.product_id
    LEFT JOIN defect_types d ON i.defect_type_id = d.defect_type_id
    WHERE i.image_type = %s
    ORDER BY i.image_id
    """
    
    from sqlalchemy.orm import joinedload
    
    query = db_session.query(Image).filter(
        Image.image_type == image_type
    ).options(
        joinedload(Image.product),
        joinedload(Image.defect_type)
    )
    
    db_images = query.all()
    
    metadata_list = []
    for img in db_images:
        metadata_list.append(ImageMetadata(
            image_id=img.image_id,
            local_path=img.file_path,
            storage_url=img.storage_url,
            product_code=img.product.product_code,
            product_name=img.product.product_name,
            defect_code=img.defect_type.defect_code if img.defect_type else 'normal',
            defect_name=img.defect_type.defect_name_ko if img.defect_type else '정상',
            image_type=img.image_type
        ))
    
    return metadata_list
```

---

## 4. 인덱스 저장 구조

### 4.1 저장 파일 구조

```
/home/dmillion/llm_chal_vlm/web/index_cache_v2/
├── defect/
│   ├── faiss_index.bin          # FAISS 인덱스
│   ├── embeddings.pt            # 임베딩 벡터
│   └── metadata.json            # 메타데이터
│       [
│         {
│           "image_id": 123,
│           "local_path": "/data/def_split/...",
│           "storage_url": "https://...",
│           "product_code": "prod1",
│           "product_name": "제품1",
│           "defect_code": "scratch",
│           "defect_name": "긁힘"
│         }
│       ]
└── normal/
    ├── faiss_index.bin
    ├── embeddings.pt
    └── metadata.json
```

### 4.2 저장/로드 함수

```python
def save_index(self, save_dir: Path):
    """인덱스 + 메타데이터 저장"""
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. FAISS 인덱스
    if self.faiss_index:
        faiss.write_index(
            self.faiss_index, 
            str(save_dir / "faiss_index.bin")
        )
    
    # 2. 임베딩
    torch.save(
        self.gallery_embs, 
        save_dir / "embeddings.pt"
    )
    
    # 3. 메타데이터 (JSON)
    metadata_list = [
        {
            "image_id": m.image_id,
            "local_path": m.local_path,
            "storage_url": m.storage_url,
            "product_code": m.product_code,
            "product_name": m.product_name,
            "defect_code": m.defect_code,
            "defect_name": m.defect_name,
            "image_type": m.image_type
        }
        for m in self.gallery_metadata
    ]
    
    with open(save_dir / "metadata.json", 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, ensure_ascii=False, indent=2)
```

---

## 5. API 라우터 수정

### 5.1 버전 선택 메커니즘

```python
# web/routers/search.py

# 전역 변수
_matcher_v1_ref = None  # 기존 버전
_matcher_v2_ref = None  # 신규 버전
_use_v2 = False         # 버전 선택 플래그

def init_search_router(
    matcher_v1, 
    matcher_v2,
    use_v2: bool = False,
    ...
):
    """라우터 초기화"""
    global _matcher_v1_ref, _matcher_v2_ref, _use_v2
    
    _matcher_v1_ref = matcher_v1
    _matcher_v2_ref = matcher_v2
    _use_v2 = use_v2
    
    print(f"[SEARCH ROUTER] 버전: {'V2 (DB)' if use_v2 else 'V1 (파일명)'}")

def get_active_matcher():
    """활성 매처 반환"""
    return _matcher_v2_ref if _use_v2 else _matcher_v1_ref
```

### 5.2 검색 API 수정

```python
@router.post("/similarity")
async def search_similar_images(request: SearchRequest):
    """
    유사 이미지 검색
    
    V1: 파일명 파싱 기반
    V2: DB 메타데이터 기반 (Object Storage URL 포함)
    """
    
    matcher = get_active_matcher()
    
    if matcher is None:
        raise HTTPException(500, "매처 초기화 안됨")
    
    # 검색 수행
    result = matcher.search(
        str(query_path), 
        top_k=request.top_k
    )
    
    # V2인 경우 이미 메타데이터 포함됨
    if _use_v2:
        return JSONResponse(content={
            "status": "success",
            "query_image": str(query_path),
            "results": result.results,  # DB 메타데이터 전체 포함
            "version": "v2"
        })
    
    # V1인 경우 기존 로직
    else:
        results_with_info = []
        for item in result.top_k_results:
            filename = Path(item["image_path"]).stem
            parts = filename.split("_")
            
            results_with_info.append({
                **item,
                "product": parts[0],
                "defect": parts[1] if len(parts) >= 2 else "unknown"
            })
        
        return JSONResponse(content={
            "status": "success",
            "results": results_with_info,
            "version": "v1"
        })
```

---

## 6. 관리자 페이지 - 인덱스 재구축

### 6.1 재구축 API

```python
# web/routers/admin/deployment.py

@router.post("/index/rebuild")
async def rebuild_indexes_v2(
    background_tasks: BackgroundTasks,
    index_type: str  # 'defect' or 'normal' or 'all'
):
    """
    DB 기반 인덱스 재구축 (V2)
    """
    
    import uuid
    task_id = str(uuid.uuid4())
    
    background_tasks.add_task(
        rebuild_index_v2_task,
        task_id=task_id,
        index_type=index_type
    )
    
    return {
        "task_id": task_id,
        "status": "started"
    }

async def rebuild_index_v2_task(task_id: str, index_type: str):
    """인덱스 재구축 백그라운드 작업"""
    
    try:
        db = next(get_db())
        INDEX_DIR = Path("/home/dmillion/llm_chal_vlm/web/index_cache_v2")
        
        if index_type in ['defect', 'all']:
            matcher = TopKSimilarityMatcherV2(...)
            matcher.build_index_from_db(db, 'defect')
            matcher.save_index(INDEX_DIR / "defect")
        
        if index_type in ['normal', 'all']:
            matcher = TopKSimilarityMatcherV2(...)
            matcher.build_index_from_db(db, 'normal')
            matcher.save_index(INDEX_DIR / "normal")
        
        print(f"[REBUILD] ✅ 완료: {index_type}")
        
    except Exception as e:
        print(f"[REBUILD] ❌ 실패: {e}")
```

---

## 7. 환경 설정 및 전환 전략

### 7.1 설정 파일

```python
# config.py (NEW)

import os

class Config:
    # 인덱스 버전 선택
    USE_INDEX_V2 = os.getenv("USE_INDEX_V2", "false").lower() == "true"
    
    # 인덱스 경로
    INDEX_DIR_V1 = "/home/dmillion/llm_chal_vlm/web/index_cache"
    INDEX_DIR_V2 = "/home/dmillion/llm_chal_vlm/web/index_cache_v2"
```

### 7.2 점진적 전환 계획

```
Phase 1: V2 개발 및 테스트
- similarity_matcher_v2.py 구현
- 병렬 운영 (V1/V2 동시 사용 가능)
- USE_INDEX_V2=false (기본값)

Phase 2: V2 검증
- 관리자 페이지에서 V2 인덱스 수동 재구축
- API 테스트 (Postman 등)
- 결과 비교 검증

Phase 3: V2 전환
- USE_INDEX_V2=true 설정
- 서버 재시작 → V2 자동 구축
- 모니터링

Phase 4: V1 제거 (선택)
- V2 안정화 후 V1 코드 제거
```

---

## 8. 주요 개선 포인트

### 8.1 성능 최적화

```python
# DB 조회 1회 → 메모리 캐싱
# 기존: 이미지당 파일명 파싱 (N회)
# 신규: DB JOIN 쿼리 1회 → 전체 메타데이터 로드
```

### 8.2 확장성

```python
# 메타데이터 추가 용이
class ImageMetadata:
    # 기본 필드
    image_id: int
    local_path: str
    storage_url: str
    
    # 확장 필드 (언제든 추가 가능)
    upload_date: datetime
    file_size: int
    resolution: str
    preprocessing_config: dict
```

### 8.3 유지보수성

```
- 파일명 변경 무관 (DB 기준)
- 제품/불량 정보 DB 수정만으로 반영
- 코드 변경 없이 메타데이터 확장
```

---

## 9. 마이그레이션 체크리스트

- [ ] `similarity_matcher_v2.py` 구현
- [ ] `Image` 모델에 Product, DefectType relationship 추가
- [ ] DB 메타데이터 조회 함수 작성
- [ ] 인덱스 저장/로드 로직 구현
- [ ] `search.py` 라우터 버전 분기 추가
- [ ] 관리자 페이지 재구축 API 추가
- [ ] 환경변수 설정 (`USE_INDEX_V2`)
- [ ] 테스트 케이스 작성
- [ ] 성능 비교 테스트
- [ ] 문서화

---

## 10. 추가 고려사항

### 10.1 Object Storage 연동

```python
# 작업자 페이지에서 이미지 업로드 시
# 1. 웹서버 → Object Storage 직접 업로드
# 2. 웹서버 → FastAPI 호출
POST /api/worker/image/register
{
  "storage_url": "https://kr.object.ncloudstorage.com/...",
  "image_type": "query_temp",
  "product_code": "prod1",
  "defect_code": "scratch"
}

# 3. FastAPI → DB Insert (임시 이미지)
# 4. 임시 이미지는 24시간 후 자동 삭제
```

### 10.2 임시 이미지 처리

```python
# 유사도 검색/이상 검출 시
# 1. Object Storage URL 받음
# 2. 임시 다운로드 /tmp/query_xxx.jpg
# 3. 처리 완료
# 4. 임시 파일 삭제 (또는 스케줄러)
```

### 10.3 결과 이미지 처리

```python
# 이상 검출 결과 (히트맵, 오버레이)
# Option 1: Object Storage 업로드 후 URL 반환 (권장)
# Option 2: Base64 인코딩 직접 반환
# Option 3: 로컬 저장 후 별도 API로 조회
```

---

## 부록: DB 스키마 변경

### A.1 Image 모델 Relationship 추가

```python
# web/database/models.py

from sqlalchemy.orm import relationship

class Image(Base):
    __tablename__ = "images"
    
    # ... 기존 컬럼 ...
    
    # Relationship 추가
    product = relationship("Product", foreign_keys=[product_id])
    defect_type = relationship("DefectType", foreign_keys=[defect_type_id])
```

### A.2 필요한 인덱스

```sql
-- 검색 성능 최적화
CREATE INDEX idx_images_type ON images(image_type);
CREATE INDEX idx_images_product ON images(product_id);
CREATE INDEX idx_images_defect ON images(defect_type_id);
```

---

**문서 버전**: 1.0  
**작성일**: 2025-01-15  
**최종 수정일**: 2025-01-15

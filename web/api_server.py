"""
TOP-K 유사도 매칭 + Anomaly Detection API 서버
FastAPI 기반으로 외부 웹서버에서 호출 가능한 REST API 제공
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import os
import sys
import shutil
from pathlib import Path
import uvicorn

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# modules 폴더의 모듈 import
from modules.similarity_matcher import TopKSimilarityMatcher, create_matcher
from modules.anomaly_detector import AnomalyDetector, create_detector


# ====================
# Pydantic 모델
# ====================

class BuildIndexRequest(BaseModel):
    """인덱스 구축 요청"""
    gallery_dir: str = Field(..., description="갤러리 이미지 디렉토리 경로")
    save_index: bool = Field(False, description="인덱스를 파일로 저장할지 여부")
    index_save_dir: Optional[str] = Field(None, description="인덱스 저장 경로")


class SearchRequest(BaseModel):
    """검색 요청 (이미지 경로 기반)"""
    query_image_path: str = Field(..., description="쿼리 이미지 경로")
    top_k: int = Field(5, ge=1, le=50, description="상위 K개 결과")


class SearchResponse(BaseModel):
    """검색 응답"""
    status: str
    query_image: str
    top_k_results: List[dict]
    total_gallery_size: int
    model_info: str


class AnomalyDetectRequest(BaseModel):
    """이상 검출 요청"""
    test_image_path: str = Field(..., description="테스트 이미지 경로")
    reference_image_path: Optional[str] = Field(None, description="기준 이미지 경로 (TOP-1)")
    product_name: Optional[str] = Field(None, description="제품명 (자동 추출 가능)")


class AnomalyDetectResponse(BaseModel):
    """이상 검출 응답"""
    status: str
    product_name: str
    image_score: float
    pixel_tau: float
    image_tau: float
    is_anomaly: bool
    heatmap_url: str
    mask_url: str
    overlay_url: str
    comparison_url: Optional[str] = None


class HealthResponse(BaseModel):
    """헬스체크 응답"""
    status: str
    message: str
    index_built: bool
    gallery_size: int


# ====================
# FastAPI 앱 생성
# ====================

app = FastAPI(
    title="유사도 매칭 + Anomaly Detection API",
    description="CLIP 기반 이미지 유사도 검색 + PatchCore 이상 검출 서비스",
    version="2.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====================
# 전역 변수
# ====================

# 매처 및 디텍터 인스턴스
matcher: Optional[TopKSimilarityMatcher] = None
detector: Optional[AnomalyDetector] = None

# 설정
UPLOAD_DIR = Path("./uploads")
INDEX_DIR = Path("./index_cache")
ANOMALY_OUTPUT_DIR = Path("./anomaly_outputs")

UPLOAD_DIR.mkdir(exist_ok=True)
INDEX_DIR.mkdir(exist_ok=True)
ANOMALY_OUTPUT_DIR.mkdir(exist_ok=True)


# ====================
# 라이프사이클 이벤트
# ====================

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 초기화"""
    global matcher, detector
    
    print("=" * 60)
    print("유사도 매칭 + Anomaly Detection API 서버 시작")
    print("=" * 60)
    
    # 1. 유사도 매처 생성
    matcher = create_matcher(
        model_id="ViT-B-32/openai",
        device="auto",
        use_fp16=False,  # 안정성 우선
        verbose=True
    )
    
    # 기존 인덱스 로드 시도
    if (INDEX_DIR / "index_data.pt").exists():
        try:
            matcher.load_index(str(INDEX_DIR))
            print(f"✅ 기존 인덱스 로드 완료: {len(matcher.gallery_paths)}개 이미지")
        except Exception as e:
            print(f"⚠️  기존 인덱스 로드 실패: {e}")
    else:
        print("ℹ️  저장된 인덱스 없음")
    
    # 인덱스가 없으면 자동 구축 시도
    if not matcher.index_built:
        default_gallery = Path("../data/def_split")  # 변경된 경로
        
        if default_gallery.exists():
            print(f"🔄 자동 인덱스 구축 시작: {default_gallery}")
            try:
                info = matcher.build_index(str(default_gallery))
                matcher.save_index(str(INDEX_DIR))
                print(f"✅ 자동 인덱스 구축 완료: {info['num_images']}개 이미지")
            except Exception as e:
                print(f"❌ 자동 인덱스 구축 실패: {e}")
        else:
            print(f"⚠️  기본 갤러리 디렉토리 없음: {default_gallery}")
    
    # 2. Anomaly Detector 생성
    try:
        detector = create_detector(
            bank_base_dir=str(project_root / "data" / "patchCore"),
            device="auto",
            verbose=True
        )
        print("✅ Anomaly Detector 초기화 완료")
    except Exception as e:
        print(f"⚠️  Anomaly Detector 초기화 실패: {e}")
        detector = None
    
    print("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """서버 종료 시 정리"""
    print("\n서버 종료 중...")


# ====================
# API 엔드포인트 - 유사도 검색
# ====================

@app.get("/health2", response_model=HealthResponse)
async def health_check():
    """헬스체크 엔드포인트"""
    return HealthResponse(
        status="healthy",
        message="API 서버가 정상 작동 중입니다",
        index_built=matcher.index_built if matcher else False,
        gallery_size=len(matcher.gallery_paths) if matcher and matcher.index_built else 0
    )


@app.post("/build_index")
async def build_index(request: BuildIndexRequest):
    """갤러리 이미지 인덱스 구축"""
    if matcher is None:
        raise HTTPException(status_code=500, detail="매처가 초기화되지 않았습니다")
    
    gallery_dir = Path(request.gallery_dir)
    if not gallery_dir.exists():
        raise HTTPException(status_code=404, detail=f"디렉토리를 찾을 수 없습니다: {gallery_dir}")
    
    try:
        info = matcher.build_index(str(gallery_dir))
        
        if request.save_index:
            save_dir = request.index_save_dir or str(INDEX_DIR)
            matcher.save_index(save_dir)
            info["index_saved"] = True
            info["index_save_path"] = save_dir
        else:
            info["index_saved"] = False
        
        return JSONResponse(content=info)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"인덱스 구축 실패: {str(e)}")


@app.post("/search", response_model=SearchResponse)
async def search_by_path(request: SearchRequest):
    """이미지 경로로 유사 이미지 검색"""
    if matcher is None:
        raise HTTPException(status_code=500, detail="매처가 초기화되지 않았습니다")
    
    if not matcher.index_built:
        raise HTTPException(status_code=400, detail="인덱스가 구축되지 않았습니다")
    
    query_path = Path(request.query_image_path)
    if not query_path.exists():
        raise HTTPException(status_code=404, detail=f"이미지를 찾을 수 없습니다: {query_path}")
    
    try:
        result = matcher.search(str(query_path), top_k=request.top_k)
        
        return SearchResponse(
            status="success",
            query_image=result.query_image,
            top_k_results=result.top_k_results,
            total_gallery_size=result.total_gallery_size,
            model_info=result.model_info
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"검색 실패: {str(e)}")


@app.post("/search/upload")
async def search_by_upload(
    file: UploadFile = File(...),
    top_k: int = Query(5, ge=1, le=50)
):
    """업로드된 이미지로 유사 이미지 검색"""
    if matcher is None:
        raise HTTPException(status_code=500, detail="매처가 초기화되지 않았습니다")
    
    if not matcher.index_built:
        raise HTTPException(status_code=400, detail="인덱스가 구축되지 않았습니다")
    
    allowed_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"지원하지 않는 파일 형식: {file_ext}")
    
    try:
        # 임시 저장
        temp_path = UPLOAD_DIR / file.filename
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 검색
        result = matcher.search(str(temp_path), top_k=top_k)
        
        # 임시 파일은 유지 (이상 검출에 사용될 수 있음)
        # temp_path.unlink()
        
        return JSONResponse(content=result.to_dict())
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"검색 실패: {str(e)}")


@app.get("/index/info")
async def get_index_info():
    """현재 인덱스 정보 조회"""
    if matcher is None:
        raise HTTPException(status_code=500, detail="매처가 초기화되지 않았습니다")
    
    if not matcher.index_built:
        return JSONResponse(content={
            "status": "no_index",
            "message": "인덱스가 구축되지 않았습니다"
        })
    
    return JSONResponse(content={
        "status": "index_built",
        "gallery_size": len(matcher.gallery_paths),
        "model_id": matcher.model_id,
        "device": matcher.device,
        "faiss_enabled": matcher.faiss_index is not None,
        "sample_paths": matcher.gallery_paths[:5]
    })


# ====================
# API 엔드포인트 - Anomaly Detection
# ====================

@app.post("/detect_anomaly", response_model=AnomalyDetectResponse)
async def detect_anomaly(request: AnomalyDetectRequest):
    """이상 검출 수행"""
    if detector is None:
        raise HTTPException(status_code=500, detail="Anomaly Detector가 초기화되지 않았습니다")
    
    test_path = Path(request.test_image_path)
    if not test_path.exists():
        raise HTTPException(status_code=404, detail=f"테스트 이미지를 찾을 수 없습니다: {test_path}")
    
    try:
        # 출력 디렉토리 생성
        output_dir = ANOMALY_OUTPUT_DIR / test_path.stem
        output_dir.mkdir(exist_ok=True)
        
        # 이상 검출
        if request.reference_image_path:
            # 기준 이미지와 함께 검출
            ref_path = Path(request.reference_image_path)
            if not ref_path.exists():
                raise HTTPException(status_code=404, detail=f"기준 이미지를 찾을 수 없습니다: {ref_path}")
            
            result = detector.detect_with_reference(
                test_image_path=str(test_path),
                reference_image_path=str(ref_path),
                product_name=request.product_name,
                output_dir=str(output_dir)
            )
        else:
            # 테스트 이미지만으로 검출
            result = detector.detect(
                test_image_path=str(test_path),
                product_name=request.product_name,
                output_dir=str(output_dir)
            )
        
        # URL 생성 (상대 경로)
        return AnomalyDetectResponse(
            status="success",
            product_name=result["product_name"],
            image_score=result["image_score"],
            pixel_tau=result["pixel_tau"],
            image_tau=result["image_tau"],
            is_anomaly=result["is_anomaly"],
            heatmap_url=f"/anomaly/image/{test_path.stem}/heatmap.png",
            mask_url=f"/anomaly/image/{test_path.stem}/mask.png",
            overlay_url=f"/anomaly/image/{test_path.stem}/overlay.png",
            comparison_url=f"/anomaly/image/{test_path.stem}/comparison.png" if "comparison_path" in result else None
        )
    
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"이상 검출 실패: {str(e)}")


@app.post("/detect_anomaly/upload")
async def detect_anomaly_upload(
    test_file: UploadFile = File(...),
    reference_file: Optional[UploadFile] = File(None),
    product_name: Optional[str] = None
):
    """업로드된 이미지로 이상 검출"""
    if detector is None:
        raise HTTPException(status_code=500, detail="Anomaly Detector가 초기화되지 않았습니다")
    
    try:
        # 테스트 이미지 저장
        test_path = UPLOAD_DIR / test_file.filename
        with test_path.open("wb") as buffer:
            shutil.copyfileobj(test_file.file, buffer)
        
        # 기준 이미지 저장 (있는 경우)
        ref_path = None
        if reference_file:
            ref_path = UPLOAD_DIR / reference_file.filename
            with ref_path.open("wb") as buffer:
                shutil.copyfileobj(reference_file.file, buffer)
        
        # 출력 디렉토리
        output_dir = ANOMALY_OUTPUT_DIR / test_path.stem
        output_dir.mkdir(exist_ok=True)
        
        # 이상 검출
        if ref_path:
            result = detector.detect_with_reference(
                test_image_path=str(test_path),
                reference_image_path=str(ref_path),
                product_name=product_name,
                output_dir=str(output_dir)
            )
        else:
            result = detector.detect(
                test_image_path=str(test_path),
                product_name=product_name,
                output_dir=str(output_dir)
            )
        
        return JSONResponse(content={
            "status": "success",
            "product_name": result["product_name"],
            "image_score": result["image_score"],
            "is_anomaly": result["is_anomaly"],
            "heatmap_url": f"/anomaly/image/{test_path.stem}/heatmap.png",
            "mask_url": f"/anomaly/image/{test_path.stem}/mask.png",
            "overlay_url": f"/anomaly/image/{test_path.stem}/overlay.png",
            "comparison_url": f"/anomaly/image/{test_path.stem}/comparison.png" if "comparison_path" in result else None
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"이상 검출 실패: {str(e)}")


@app.get("/anomaly/image/{result_id}/{filename}")
async def serve_anomaly_image(result_id: str, filename: str):
    """이상 검출 결과 이미지 제공"""
    file_path = ANOMALY_OUTPUT_DIR / result_id / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="이미지를 찾을 수 없습니다")
    
    return FileResponse(file_path, media_type="image/png")


# ====================
# 이미지 서빙 및 정적 파일
# ====================

@app.get("/api/image/{image_path:path}")
async def serve_image(image_path: str):
    """이미지 파일 제공 엔드포인트"""
    try:
        file_path = Path(image_path)
        
        if not file_path.exists() or not file_path.is_file():
            raise HTTPException(status_code=404, detail="이미지를 찾을 수 없습니다")
        
        allowed_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
        if file_path.suffix.lower() not in allowed_extensions:
            raise HTTPException(status_code=400, detail="이미지 파일이 아닙니다")
        
        return FileResponse(file_path, media_type=f"image/{file_path.suffix[1:]}")
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"이미지 로드 실패: {str(e)}")


@app.delete("/uploads/clean")
async def clean_uploads():
    """업로드 디렉토리 정리"""
    try:
        for file in UPLOAD_DIR.glob("*"):
            if file.is_file():
                file.unlink()
        
        return JSONResponse(content={
            "status": "success",
            "message": "업로드 디렉토리가 정리되었습니다"
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"정리 실패: {str(e)}")


# HTML 파일 서빙
WEB_DIR = Path(__file__).parent

@app.get("/matching.html")
async def serve_matching():
    """matching.html 서빙"""
    return FileResponse(WEB_DIR / "matching.html")

@app.get("/")
async def root():
    """루트 접근 시 matching.html로 리다이렉트"""
    return FileResponse(WEB_DIR / "matching.html")


# ====================
# 서버 실행
# ====================

if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=5000,  # 변경된 포트
        reload=True,
        log_level="info"
    )
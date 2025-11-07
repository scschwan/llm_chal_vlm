"""
TOP-K 유사도 매칭 API 서버
FastAPI 기반으로 외부 웹서버에서 호출 가능한 REST API 제공
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
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
    title="TOP-K 유사도 매칭 API",
    description="CLIP 기반 이미지 유사도 검색 서비스",
    version="1.0.0"
)

# CORS 설정 (다른 도메인에서 접근 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====================
# 전역 변수
# ====================

# 매처 인스턴스 (서버 시작 시 초기화)
matcher: Optional[TopKSimilarityMatcher] = None

# 설정
UPLOAD_DIR = Path("./uploads")
INDEX_DIR = Path("./index_cache")
UPLOAD_DIR.mkdir(exist_ok=True)
INDEX_DIR.mkdir(exist_ok=True)


# ====================
# 라이프사이클 이벤트
# ====================

# 기존 startup_event 함수 수정

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 매처 초기화"""
    global matcher
    
    print("=" * 50)
    print("TOP-K 유사도 매칭 API 서버 시작")
    print("=" * 50)
    
    # 매처 생성
    matcher = create_matcher(
        model_id="ViT-B-32/openai",
        device="auto",
        #use_fp16=True,
        use_fp16=False,
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
        default_gallery = Path("../data/ok_front")  # 기본 갤러리 경로
        
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
            print("   /build_index 엔드포인트로 수동 구축이 필요합니다")
    
    print("=" * 50)


@app.on_event("shutdown")
async def shutdown_event():
    """서버 종료 시 정리"""
    print("\n서버 종료 중...")


# ====================
# API 엔드포인트
# ====================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    헬스체크 엔드포인트
    ALB/NLB 헬스체크용
    """
    return HealthResponse(
        status="healthy",
        message="API 서버가 정상 작동 중입니다",
        index_built=matcher.index_built if matcher else False,
        gallery_size=len(matcher.gallery_paths) if matcher and matcher.index_built else 0
    )


@app.post("/build_index")
async def build_index(request: BuildIndexRequest):
    """
    갤러리 이미지 인덱스 구축
    
    Request Body:
    - gallery_dir: 갤러리 디렉토리 경로
    - save_index: 인덱스 저장 여부
    - index_save_dir: 저장 경로 (기본: ./index_cache)
    """
    if matcher is None:
        raise HTTPException(status_code=500, detail="매처가 초기화되지 않았습니다")
    
    gallery_dir = Path(request.gallery_dir)
    if not gallery_dir.exists():
        raise HTTPException(status_code=404, detail=f"디렉토리를 찾을 수 없습니다: {gallery_dir}")
    
    try:
        # 인덱스 구축
        info = matcher.build_index(str(gallery_dir))
        
        # 인덱스 저장 (요청 시)
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
    """
    이미지 경로로 유사 이미지 검색
    
    Request Body:
    - query_image_path: 쿼리 이미지 경로
    - top_k: 상위 K개 (기본: 5)
    """
    if matcher is None:
        raise HTTPException(status_code=500, detail="매처가 초기화되지 않았습니다")
    
    if not matcher.index_built:
        raise HTTPException(status_code=400, detail="인덱스가 구축되지 않았습니다. /build_index를 먼저 호출하세요")
    
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
    """
    업로드된 이미지로 유사 이미지 검색
    
    Form Data:
    - file: 이미지 파일 (multipart/form-data)
    - top_k: 상위 K개 (기본: 5)
    """
    if matcher is None:
        raise HTTPException(status_code=500, detail="매처가 초기화되지 않았습니다")
    
    if not matcher.index_built:
        raise HTTPException(status_code=400, detail="인덱스가 구축되지 않았습니다")
    
    # 파일 확장자 검증
    allowed_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"지원하지 않는 파일 형식입니다: {file_ext}"
        )
    
    try:
        # 임시 저장
        temp_path = UPLOAD_DIR / file.filename
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 검색
        result = matcher.search(str(temp_path), top_k=top_k)
        
        # 임시 파일 삭제
        temp_path.unlink()
        
        return JSONResponse(content=result.to_dict())
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"검색 실패: {str(e)}")


@app.get("/api/image/{image_path:path}")
async def serve_image(image_path: str):
    """
    이미지 파일 제공 엔드포인트
    검색 결과 이미지를 브라우저에서 볼 수 있도록 제공
    """
    try:
        # URL 디코딩된 경로
        file_path = Path(image_path)
        
        # 파일 존재 확인
        if not file_path.exists() or not file_path.is_file():
            raise HTTPException(status_code=404, detail="이미지를 찾을 수 없습니다")
        
        # 이미지 파일인지 확인
        allowed_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
        if file_path.suffix.lower() not in allowed_extensions:
            raise HTTPException(status_code=400, detail="이미지 파일이 아닙니다")
        
        return FileResponse(
            file_path,
            media_type=f"image/{file_path.suffix[1:]}"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"이미지 로드 실패: {str(e)}")


@app.get("/index/info")
async def get_index_info():
    """
    현재 인덱스 정보 조회
    """
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
        "sample_paths": matcher.gallery_paths[:5]  # 샘플 경로 5개
    })


@app.delete("/uploads/clean")
async def clean_uploads():
    """
    업로드 디렉토리 정리
    """
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


# ====================
# 정적 파일 서빙
# ====================

# HTML 파일들을 서빙하기 위한 정적 파일 마운트

# 현재 스크립트의 디렉토리 경로
CURRENT_DIR = Path(__file__).parent

# HTML 파일들을 서빙하기 위한 정적 파일 마운트
try:
    app.mount("/static", StaticFiles(directory=str(CURRENT_DIR), html=True), name="static")
except Exception as e:
    print(f"⚠️  정적 파일 마운트 실패: {e}")


# 루트 경로에서 matching.html로 리다이렉트
from fastapi.responses import RedirectResponse

@app.get("/")
async def root():
    """루트 접근 시 matching.html로 리다이렉트"""
    return RedirectResponse(url="/static/matching.html")

@app.get("/matching.html")
async def matching_page():
    """matching.html 직접 서빙"""
    html_path = CURRENT_DIR / "matching.html"
    if html_path.exists():
        return FileResponse(html_path)
    raise HTTPException(status_code=404, detail="matching.html을 찾을 수 없습니다")


# ====================
# 서버 실행
# ====================

if __name__ == "__main__":
    # 개발 서버 실행
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8080,
        reload=True,  # 코드 변경 시 자동 재시작
        log_level="info"
    )
"""
메인 API 서버 - 라우터 통합
"""


from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict,List, Optional
import os
import sys
import shutil
from pathlib import Path
import uvicorn
from fastapi.staticfiles import StaticFiles
import httpx


import time  # 기존 import에 추가


# 프로젝트 루트 경로
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 기존 imports
from modules.similarity_matcher import TopKSimilarityMatcher, create_matcher
from modules.anomaly_detector import AnomalyDetector, create_detector
from modules.vlm import RAGManager, DefectMapper, PromptBuilder

# 라우터 imports
from routers.upload import router as upload_router, init_upload_router
from routers.search import router as search_router, init_search_router


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
    version="3.0.0"
)

# 디렉토리 설정
WEB_DIR = Path(__file__).parent
STATIC_DIR = WEB_DIR / "static"
PAGES_DIR = WEB_DIR / "pages"
UPLOAD_DIR = WEB_DIR / "uploads"
INDEX_DIR = WEB_DIR / "index_cache"
ANOMALY_OUTPUT_DIR = WEB_DIR / "anomaly_outputs"

# 디렉토리 생성
STATIC_DIR.mkdir(exist_ok=True)
(STATIC_DIR / "css").mkdir(exist_ok=True)
(STATIC_DIR / "js").mkdir(exist_ok=True)
PAGES_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)
INDEX_DIR.mkdir(exist_ok=True)
ANOMALY_OUTPUT_DIR.mkdir(exist_ok=True)

# Static 파일 마운트
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

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

matcher: Optional[TopKSimilarityMatcher] = None
detector: Optional[AnomalyDetector] = None
current_index_type: Optional[str] = None

vlm_components = {
    "rag": None,
    "vlm": None,
    "mapper": None,
    "prompt_builder": PromptBuilder()
}

# ====================
# 라우터 등록
# ====================

# 업로드 라우터 초기화 및 등록
init_upload_router(UPLOAD_DIR)
init_search_router(matcher, INDEX_DIR, project_root)

# 라우터 등록 부분에 추가
app.include_router(upload_router)
app.include_router(search_router)

# TODO: 다른 라우터들도 추가
# app.include_router(anomaly_router)
# app.include_router(manual_router)

# ====================
# 라이프사이클 이벤트
# ====================

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 초기화"""
    global matcher, detector, current_index_type
    
    print("=" * 60)
    print("유사도 매칭 + Anomaly Detection API 서버 시작")
    print("=" * 60)
    
    # 1. 유사도 매처 생성
    matcher = create_matcher(
        model_id="ViT-B-32/openai",
        device="auto",
        use_fp16=False,
        verbose=True
    )
    
    # 2. 두 인덱스 모두 미리 구축
    print("\n" + "="*60)
    print("인덱스 사전 구축 시작")
    print("="*60)
    
    # 2-1. 불량 이미지 인덱스 구축
    defect_dir = project_root / "data" / "def_split"
    defect_index_path = INDEX_DIR / "defect"
    defect_index_path.mkdir(parents=True, exist_ok=True)
    
    if defect_dir.exists():
        try:
            print(f"\n[1/2] 불량 이미지 인덱스 구축 중...")
            print(f"      경로: {defect_dir}")
            
            info = matcher.build_index(str(defect_dir))
            matcher.save_index(str(defect_index_path))
            
            print(f"      ✅ 완료: {info['num_images']}개 이미지")
        except Exception as e:
            print(f"      ❌ 실패: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n[1/2] ⚠️  불량 이미지 디렉토리 없음: {defect_dir}")
    
    # 2-2. 정상 이미지 통합 인덱스 구축
    normal_base_dir = project_root / "data" / "patchCore"
    normal_index_path = INDEX_DIR / "normal"
    normal_index_path.mkdir(parents=True, exist_ok=True)
    
    if normal_base_dir.exists():
        try:
            print(f"\n[2/2] 정상 이미지 통합 인덱스 구축 중...")
            print(f"      기본 경로: {normal_base_dir}")
            
            # 모든 제품 폴더 탐색
            product_dirs = [d for d in normal_base_dir.iterdir() if d.is_dir()]
            
            if not product_dirs:
                print(f"      ⚠️  제품 폴더를 찾을 수 없습니다")
            else:
                print(f"      발견된 제품: {[d.name for d in product_dirs]}")
                
                # 통합 인덱스 구축 (하위 폴더 재귀 탐색)
                info = matcher.build_index(str(normal_base_dir))
                matcher.save_index(str(normal_index_path))
                
                print(f"      ✅ 완료: {info['num_images']}개 이미지 (통합)")
                
                # 제품별 이미지 개수 표시
                for prod_dir in product_dirs:
                    prod_images = list(prod_dir.glob("*.jpg")) + list(prod_dir.glob("*.png"))
                    print(f"         - {prod_dir.name}: {len(prod_images)}개")
                    
        except Exception as e:
            print(f"      ❌ 실패: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n[2/2] ⚠️  정상 이미지 기본 디렉토리 없음: {normal_base_dir}")
    
    print("\n" + "="*60)
    print("인덱스 사전 구축 완료")
    print("="*60)
    
    # 3. 기본 인덱스를 불량 이미지로 설정
    try:
        print("\n🔄 기본 인덱스 로드 중 (불량 이미지)...")
        if (defect_index_path / "index_data.pt").exists():
            matcher.load_index(str(defect_index_path))
            current_index_type = "defect"
            print(f"✅ 불량 이미지 인덱스 로드 완료: {len(matcher.gallery_paths)}개")
        else:
            print("⚠️  저장된 불량 인덱스를 찾을 수 없습니다")
    except Exception as e:
        print(f"⚠️  기본 인덱스 로드 실패: {e}")

    # 4. Anomaly Detector 생성
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
    
    # 5. VLM 컴포넌트 초기화
    # init_vlm_components() - 기존 함수 재사용
    print("✅ VLM Component 초기화 완료")
    print("=" * 60)
    print("서버 초기화 완료")
    print("=" * 60 + "\n")


@app.on_event("shutdown")
async def shutdown_event():
    """서버 종료 시 정리"""
    print("\n서버 종료 중...")


# ====================
# 기본 라우트 (페이지 서빙)
# ====================

@app.get("/")
async def root():
    """루트 접근 시 업로드 페이지로"""
    return FileResponse(PAGES_DIR / "upload.html")


@app.get("/upload.html")
async def serve_upload():
    """업로드 페이지"""
    return FileResponse(PAGES_DIR / "upload.html")


@app.get("/search.html")
async def serve_search():
    """검색 페이지"""
    html_path = PAGES_DIR / "search.html"
    if not html_path.exists():
        raise HTTPException(404, "검색 페이지가 아직 구현되지 않았습니다")
    return FileResponse(html_path)


@app.get("/anomaly.html")
async def serve_anomaly():
    """이상 검출 페이지"""
    html_path = PAGES_DIR / "anomaly.html"
    if not html_path.exists():
        raise HTTPException(404, "이상 검출 페이지가 아직 구현되지 않았습니다")
    return FileResponse(html_path)


@app.get("/manual.html")
async def serve_manual():
    """매뉴얼 페이지"""
    html_path = PAGES_DIR / "manual.html"
    if not html_path.exists():
        raise HTTPException(404, "매뉴얼 페이지가 아직 구현되지 않았습니다")
    return FileResponse(html_path)


# ====================
# 기존 API 엔드포인트 유지 (하위 호환성)
# ====================

@app.get("/index/status")
async def get_index_status():
    """현재 인덱스 상태 조회"""
    if matcher is None:
        return {
            "status": "error",
            "message": "매처가 초기화되지 않았습니다",
            "current_index_type": None,
            "gallery_count": 0
        }
    
    return {
        "status": "success",
        "current_index_type": current_index_type,
        "gallery_count": len(matcher.gallery_paths) if matcher.gallery_paths else 0,
        "index_built": matcher.index_built,
        "model_id": matcher.model_id if hasattr(matcher, 'model_id') else None
    }

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
async def build_index(request: dict):
    """인덱스 재구축"""

    
    class BuildIndexRequest(BaseModel):
        gallery_dir: str = Field(..., description="갤러리 디렉토리")
        save_index: bool = Field(True, description="인덱스 저장 여부")
    
    if matcher is None:
        raise HTTPException(500, "매처가 초기화되지 않았습니다")
    
    req = BuildIndexRequest(**request)
    gallery_dir = Path(req.gallery_dir)
    
    if not gallery_dir.exists():
        raise HTTPException(404, f"디렉토리를 찾을 수 없습니다: {gallery_dir}")
    
    try:
        info = matcher.build_index(str(gallery_dir))
        
        if req.save_index:
            save_dir = INDEX_DIR / "defect"
            matcher.save_index(str(save_dir))
            info["index_saved"] = True
            info["index_save_path"] = str(save_dir)
        
        return info
    
    except Exception as e:
        raise HTTPException(500, f"인덱스 구축 실패: {str(e)}")



@app.get("/api/image/{image_path:path}")
async def serve_image(image_path: str):
    """이미지 파일 제공 엔드포인트"""
    try:
        # 상대 경로 정규화
        if image_path.startswith("../"):
            image_path = image_path.replace("../", "")
        
        # 경로 처리
        if image_path.startswith("uploads/"):
            file_path = UPLOAD_DIR / image_path.replace("uploads/", "")
        elif image_path.startswith("data/"):
            file_path = project_root / image_path
        else:
            # 기본적으로 project_root 기준
            file_path = project_root / image_path
        
        print(f"[IMAGE] 이미지 서빙 시도: {file_path}")
        
        if not file_path.exists():
            raise HTTPException(404, f"이미지를 찾을 수 없습니다: {file_path}")
        
        return FileResponse(str(file_path))
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[IMAGE] 이미지 서빙 오류: {e}")
        raise HTTPException(500, str(e))


# 불량 등록 API (기존 코드 유지)
@app.post("/register_defect")
async def register_defect(
    file: UploadFile = File(...),
    product_name: str = Form(...),
    defect_name: str = Form(...)
):
    """불량 이미지 등록"""
    defect_dir = project_root / "data" / "def_split"
    defect_dir.mkdir(parents=True, exist_ok=True)
    
    # 현재 등록된 파일 중 최대 seqno 찾기
    existing_files = list(defect_dir.glob(f"{product_name}_{defect_name}_*"))
    
    max_seqno = 0
    for existing_file in existing_files:
        try:
            stem = existing_file.stem
            parts = stem.split('_')
            if len(parts) >= 3:
                seqno = int(parts[-1])
                max_seqno = max(max_seqno, seqno)
        except (ValueError, IndexError):
            continue
    
    # 새로운 seqno
    new_seqno = max_seqno + 1
    
    # 파일명 생성
    ext = Path(file.filename).suffix
    new_filename = f"{product_name}_{defect_name}_{new_seqno:03d}{ext}"
    save_path = defect_dir / new_filename
    
    # 저장
    with save_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # 인덱스 재구축
    index_rebuilt = False
    if matcher and matcher.index_built:
        try:
            defect_index_path = INDEX_DIR / "defect"
            matcher.build_index(str(defect_dir))
            matcher.save_index(str(defect_index_path))
            index_rebuilt = True
        except Exception as e:
            print(f"[REGISTER] 인덱스 재구축 실패: {e}")
    
    return JSONResponse(content={
        "status": "success",
        "saved_path": str(save_path),
        "filename": new_filename,
        "product_name": product_name,
        "defect_name": defect_name,
        "seqno": new_seqno,
        "index_rebuilt": index_rebuilt
    })

# ====================
# 서버 실행
# ====================

if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=5000,
        reload=True,
        log_level="info"
    )
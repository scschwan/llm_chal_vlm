"""
유사도 검색 관련 API 라우터
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from pathlib import Path

router = APIRouter(prefix="/search", tags=["search"])

# ✅ 전역 변수 (api_server.py에서 주입받을 참조)
_matcher_ref = None
_index_dir_ref = None
_project_root_ref = None


def init_search_router(similarity_matcher, index_dir, proj_root):
    """라우터 초기화"""
    global _matcher_ref, _index_dir_ref, _project_root_ref
    _matcher_ref = similarity_matcher
    _index_dir_ref = index_dir
    _project_root_ref = proj_root
    print(f"[SEARCH ROUTER] 초기화 완료: matcher={_matcher_ref is not None}")


def get_matcher():
    """매처 참조 반환"""
    return _matcher_ref


def get_index_dir():
    """인덱스 디렉토리 반환"""
    return _index_dir_ref


def get_project_root():
    """프로젝트 루트 반환"""
    return _project_root_ref


class SearchRequest(BaseModel):
    """검색 요청"""
    query_image_path: str = Field(..., description="쿼리 이미지 경로")
    top_k: int = Field(5, ge=1, le=20, description="상위 K개 결과")


async def ensure_defect_index():
    """불량 이미지 인덱스로 전환"""
    matcher = get_matcher()
    INDEX_DIR = get_index_dir()
    project_root = get_project_root()
    
    if matcher is None:
        raise HTTPException(500, "매처가 초기화되지 않았습니다")
    
    defect_index_path = INDEX_DIR / "defect"
    
    try:
        # 저장된 인덱스가 있으면 로드
        if (defect_index_path / "index_data.pt").exists():
            print(f"[SEARCH] 불량 이미지 인덱스 로드: {defect_index_path}")
            matcher.load_index(str(defect_index_path))
            return {
                "status": "success",
                "index_type": "defect",
                "gallery_count": len(matcher.gallery_paths) if matcher.gallery_paths else 0,
                "message": "불량 이미지 인덱스 로드 완료"
            }
        else:
            # 인덱스가 없으면 새로 구축
            defect_dir = project_root / "data" / "def_split"
            
            if not defect_dir.exists():
                raise FileNotFoundError(f"불량 이미지 디렉토리가 없습니다: {defect_dir}")
            
            print(f"[SEARCH] 불량 이미지 인덱스 구축 중: {defect_dir}")
            info = matcher.build_index(str(defect_dir))
            
            # 인덱스 저장
            defect_index_path.mkdir(parents=True, exist_ok=True)
            matcher.save_index(str(defect_index_path))
            
            return {
                "status": "success",
                "index_type": "defect",
                "gallery_count": info["num_images"],
                "message": "불량 이미지 인덱스 구축 완료"
            }
    
    except Exception as e:
        print(f"[SEARCH] 인덱스 로드 실패: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"인덱스 로드 실패: {str(e)}")


@router.post("/similarity")
async def search_similar_images(request: SearchRequest):
    """유사 이미지 검색"""
    matcher = get_matcher()
    project_root = get_project_root()
    
    if matcher is None:
        raise HTTPException(500, "유사도 매처가 초기화되지 않았습니다")
    
    # 불량 이미지 인덱스로 자동 전환
    await ensure_defect_index()
    
    if not matcher.index_built:
        raise HTTPException(400, "인덱스가 구축되지 않았습니다")
    
    # 이미지 경로 정규화
    query_path = Path(request.query_image_path)
    
    # 상대 경로면 절대 경로로 변환
    if not query_path.is_absolute():
        if str(query_path).startswith("uploads/"):
            query_path = project_root / "web" / query_path
        else:
            query_path = project_root / query_path
    
    if not query_path.exists():
        raise HTTPException(404, f"이미지를 찾을 수 없습니다: {query_path}")
    
    try:
        print(f"[SEARCH] 유사도 검색 시작: {query_path.name}")
        
        # 유사도 검색 수행
        result = matcher.search(str(query_path), top_k=request.top_k)
        
        # 제품/불량 정보 추출
        results_with_info = []
        for item in result.top_k_results:
            filename = Path(item["image_path"]).stem
            parts = filename.split("_")
            
            product = parts[0] if len(parts) >= 1 else "unknown"
            defect = parts[1] if len(parts) >= 2 else "unknown"
            seq = parts[2] if len(parts) >= 3 else "000"
            
            results_with_info.append({
                **item,
                "product": product,
                "defect": defect,
                "sequence": seq
            })
        
        print(f"[SEARCH] 검색 완료: {len(results_with_info)}개 결과")
        
        return JSONResponse(content={
            "status": "success",
            "query_image": str(query_path),
            "top_k_results": results_with_info,
            "total_gallery_size": result.total_gallery_size,
            "model_info": result.model_info
        })
    
    except Exception as e:
        print(f"[SEARCH] 검색 실패: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"검색 실패: {str(e)}")


@router.get("/index/status")
async def get_search_index_status():
    """검색 인덱스 상태 조회"""
    matcher = get_matcher()
    
    if matcher is None:
        return {
            "status": "error",
            "message": "매처가 초기화되지 않았습니다",
            "index_built": False,
            "gallery_count": 0
        }
    
    return {
        "status": "success",
        "index_built": matcher.index_built,
        "gallery_count": len(matcher.gallery_paths) if matcher.gallery_paths else 0,
        "index_type": "defect"
    }
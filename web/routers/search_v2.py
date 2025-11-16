"""
유사도 검색 API V2 (DB 메타데이터 기반)

기존 search.py와의 차이점:
- DB에서 메타데이터 조회
- Object Storage URL 포함
- 제품명, 불량명 등 풍부한 메타데이터 제공
"""
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Optional
from sqlalchemy.orm import Session
import tempfile
import os
import shutil

router = APIRouter(prefix="/v2/search", tags=["search_v2"])

# 전역 변수
_matcher_v2_ref = None
_index_dir_v2_ref = None
_project_root_ref = None
_project_config = None

def init_search_v2_router(matcher_v2, index_dir_v2, proj_root ,config):
    """라우터 V2 초기화"""
    global _matcher_v2_ref, _index_dir_v2_ref, _project_root_ref ,_project_config
    _matcher_v2_ref = matcher_v2
    _index_dir_v2_ref = index_dir_v2
    _project_root_ref = proj_root
    _project_config = config
    print(f"[SEARCH V2 ROUTER] 초기화 완료: matcher={_matcher_v2_ref is not None}")


def get_matcher_v2():
    """매처 V2 참조 반환"""
    return _matcher_v2_ref


def get_index_dir_v2():
    """인덱스 디렉토리 V2 반환"""
    return _index_dir_v2_ref


def get_project_root():
    """프로젝트 루트 반환"""
    return _project_root_ref


class SearchRequestV2(BaseModel):
    """검색 요청 V2"""
    query_image_path: str = Field(..., description="쿼리 이미지 경로")
    top_k: int = Field(5, ge=1, le=20, description="상위 K개 결과")
    index_type: Optional[str] = Field("defect", description="인덱스 타입 (normal/defect)")



def download_from_object_storage(storage_url: str, local_path: str) -> bool:
    """
    Object Storage에서 파일 다운로드
    
    Args:
        storage_url: Object Storage URL
        local_path: 로컬 저장 경로
    
    Returns:
        성공 여부
    """
    try:
        import boto3
        from urllib.parse import urlparse
        
        # URL 파싱
        parsed = urlparse(storage_url)
        # 예: https://kr.object.ncloudstorage.com/dm-obs/def_split/prod1_burr_021.jpeg
        # bucket: dm-obs
        # key: def_split/prod1_burr_021.jpeg
        
        path_parts = parsed.path.lstrip('/').split('/', 1)
        if len(path_parts) < 2:
            raise ValueError(f"Invalid storage URL format: {storage_url}")
        
        #bucket_name = path_parts[0]
        bucket_name = os.getenv('NCP_BUCKET', _project_config.get('NCP_BUCKET', 'dm-obs'))
        object_key = path_parts[1]
        
        # S3 클라이언트 생성
        s3 = boto3.client(
            service_name='s3',
            #endpoint_url='https://kr.object.ncloudstorage.com',
            endpoint_url=os.getenv('NCP_STORAGE_BASE_URL', _project_config.get('NCP_STORAGE_BASE_URL', 'https://kr.object.ncloudstorage.com')),
            region_name='kr-standard',
            aws_access_key_id=os.getenv('NCP_ACCESS_KEY', _project_config.get('NCP_ACCESS_KEY', '')),
            aws_secret_access_key=os.getenv('NCP_SECRET_KEY',  _project_config.get('NCP_ACCESS_KEY', ''))
        )
        
        # 파일 다운로드
        s3.download_file(bucket_name, object_key, local_path)
        
        print(f"[OBS] 다운로드 완료: {storage_url} -> {local_path}")
        return True
        
    except Exception as e:
        print(f"[OBS] 다운로드 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


# 현재 로드된 인덱스 타입 추적
_current_index_type_v2 = None


async def ensure_index_loaded(index_type: str):
    """
    지정된 인덱스 타입 로드
    
    Args:
        index_type: 'normal' or 'defect'
    """
    global _current_index_type_v2
    
    matcher = get_matcher_v2()
    INDEX_DIR = get_index_dir_v2()
    
    if matcher is None:
        raise HTTPException(500, "매처 V2가 초기화되지 않았습니다")
    
    # 이미 해당 인덱스가 로드되어 있으면 스킵
    if _current_index_type_v2 == index_type and matcher.index_built:
        print(f"[SEARCH V2] 이미 {index_type} 인덱스가 로드되어 있음")
        return {
            "status": "success",
            "index_type": index_type,
            "cached": True,
            "gallery_count": len(matcher.gallery_metadata)
        }
    
    # 인덱스 경로
    index_path = INDEX_DIR / index_type
    
    if not (index_path / "metadata.json").exists():
        raise HTTPException(
            404, 
            f"{index_type} 인덱스를 찾을 수 없습니다. "
            f"관리자 페이지에서 인덱스를 먼저 구축해주세요."
        )
    
    try:
        print(f"[SEARCH V2] {index_type} 인덱스 로드 중...")
        matcher.load_index(str(index_path))
        _current_index_type_v2 = index_type
        
        print(f"[SEARCH V2] {index_type} 인덱스 로드 완료 ({len(matcher.gallery_metadata)}개)")
        
        return {
            "status": "success",
            "index_type": index_type,
            "cached": False,
            "gallery_count": len(matcher.gallery_metadata)
        }
    
    except Exception as e:
        print(f"[SEARCH V2] 인덱스 로드 실패: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"인덱스 로드 실패: {str(e)}")


@router.post("/similarity")
async def search_similar_images_v2(request: SearchRequestV2):
    """
    유사 이미지 검색 V2 (DB 메타데이터 기반)
    
    Response:
        {
            "status": "success",
            "query_image": "/path/to/query.jpg",
            "index_type": "defect",
            "results": [
                {
                    "image_id": 123,
                    "similarity_score": 0.95,
                    "local_path": "/home/dmillion/llm_chal_vlm/data/def_split/prod1_burr_021.jpeg",
                    "storage_url": "https://kr.object.ncloudstorage.com/dm-obs/def_split/prod1_burr_021.jpeg",
                    "product_id": 1,
                    "product_code": "prod1",
                    "product_name": "제품1",
                    "defect_type_id": 3,
                    "defect_code": "burr",
                    "defect_name": "버(Burr)",
                    "image_type": "defect",
                    "file_name": "prod1_burr_021.jpeg"
                }
            ],
            "total_gallery_size": 1000,
            "model_info": {...}
        }
    """
    matcher = get_matcher_v2()
    project_root = get_project_root()
    
    if matcher is None:
        raise HTTPException(500, "유사도 매처 V2가 초기화되지 않았습니다")
    
    # 지정된 인덱스 로드
    await ensure_index_loaded(request.index_type)
    
    query_image_path = request.query_image_path
    temp_file_path = None

    try:
        # ✅ Object Storage URL 체크
        if query_image_path.startswith("http://") or query_image_path.startswith("https://"):
            print(f"[SEARCH V2] Object Storage URL 감지: {query_image_path}")
            
            # 임시 파일 생성
            temp_dir = Path(tempfile.gettempdir()) / "obs_downloads"
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            # 파일명 추출
            from urllib.parse import urlparse
            parsed_url = urlparse(query_image_path)
            file_name = Path(parsed_url.path).name
            temp_file_path = temp_dir / file_name
            
           # Object Storage에서 다운로드
            from web.utils.object_storage import get_obs_manager
            from web.utils.session_helper import generate_session_id,create_obs_session_folder,upload_origin_to_obs
            obs = get_obs_manager()
            
            # URL에서 s3_key 추출
            path_parts = [p for p in parsed_url.path.split('/') if p]
            # dm-obs 이후 경로
            bucket_idx = path_parts.index(obs.bucket) if obs.bucket in path_parts else -1
            if bucket_idx >= 0:
                s3_key = '/'.join(path_parts[bucket_idx + 1:])
            else:
                s3_key = '/'.join(path_parts)
            
            if not obs.download_file(s3_key, str(temp_file_path)):
                raise HTTPException(500, "Object Storage에서 이미지 다운로드 실패")
            
            query_path = temp_file_path
            print(f"[SEARCH V2] 임시 파일 생성: {query_path}")
            
            # 2. 세션 ID 생성
            session_id = generate_session_id()
            print(f"[SEARCH V2] 세션 생성: {session_id}")
            
            # 3. OBS에 세션 폴더 생성
            create_obs_session_folder(session_id)
            
            # 4. origin.{확장자}로 OBS에 업로드
            uploaded_origin_url = upload_origin_to_obs(str(temp_file_path), session_id)
            print(f"[SEARCH V2] 원본 업로드: {uploaded_origin_url}")
            
            # 5. 로컬 세션 폴더에도 origin.{확장자}로 저장
            session_dir = Path("/home/dmillion/llm_chal_vlm/uploads") / session_id
            session_dir.mkdir(parents=True, exist_ok=True)
            
            file_ext = Path(file_name).suffix
            origin_filename = f"origin{file_ext}"
            origin_local_path = session_dir / origin_filename
            shutil.copy(str(temp_file_path), str(origin_local_path))
            
            print(f"[SEARCH V2] 로컬 원본 저장: {origin_local_path}")
        
        else:
            # 로컬 경로 처리 (기존 로직)
            query_path = Path(query_image_path)
            
            # 상대 경로면 절대 경로로 변환
            if not query_path.is_absolute():
                if str(query_path).startswith("uploads/"):
                    query_path = project_root / "web" / query_path
                else:
                    query_path = project_root / query_path
        
        if not query_path.exists():
            raise HTTPException(404, f"이미지를 찾을 수 없습니다: {query_path}")
        
        print(f"[SEARCH V2] 유사도 검색 시작: {query_path.name}, top_k={request.top_k}")
        
        # 유사도 검색 수행 (메타데이터 포함)
        result = matcher.search(str(query_path), top_k=request.top_k)
        
        print(f"[SEARCH V2] 검색 완료: {len(result.results)}개 결과")
        
        return JSONResponse(content={
            "status": "success",
            "session_id": session_id,  # ← 추가
            "uploaded_origin_url": uploaded_origin_url,  # ← 추가
            "query_image": str(query_path),
            "query_image_source": "object_storage" if temp_file_path else "local",
            "index_type": result.index_type,
            "results": result.results,
            "total_gallery_size": result.total_gallery_size,
            "model_info": result.model_info,
            "version": "v2"
        })
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"[SEARCH V2] 검색 실패: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"검색 실패: {str(e)}")
    
    finally:
        # ✅ 임시 파일 정리
        if temp_file_path and temp_file_path.exists():
            try:
                temp_file_path.unlink()
                print(f"[SEARCH V2] 임시 파일 삭제: {temp_file_path}")
            except Exception as e:
                print(f"[SEARCH V2] 임시 파일 삭제 실패: {e}")


@router.get("/index/status")
async def get_index_status_v2():
    """
    인덱스 상태 조회 V2
    
    Response:
        {
            "status": "success",
            "index_built": true,
            "index_type": "defect",
            "gallery_count": 1000,
            "has_metadata": true
        }
    """
    matcher = get_matcher_v2()
    
    if matcher is None:
        return {
            "status": "error",
            "message": "매처 V2가 초기화되지 않았습니다",
            "index_built": False,
            "gallery_count": 0
        }
    
    return {
        "status": "success",
        "index_built": matcher.index_built,
        "index_type": matcher.index_type,
        "gallery_count": len(matcher.gallery_metadata) if matcher.gallery_metadata else 0,
        "has_metadata": matcher.gallery_metadata is not None,
        "version": "v2"
    }


@router.post("/index/switch")
async def switch_index_type(index_type: str):
    """
    인덱스 타입 전환
    
    Args:
        index_type: 'normal' or 'defect'
    
    Response:
        {
            "status": "success",
            "previous_type": "normal",
            "current_type": "defect",
            "gallery_count": 1000
        }
    """
    global _current_index_type_v2
    
    if index_type not in ["normal", "defect"]:
        raise HTTPException(400, "index_type은 'normal' 또는 'defect'만 가능합니다")
    
    previous_type = _current_index_type_v2
    
    # 인덱스 로드
    result = await ensure_index_loaded(index_type)
    
    return {
        "status": "success",
        "previous_type": previous_type,
        "current_type": index_type,
        "gallery_count": result.get("gallery_count", 0),
        "cached": result.get("cached", False)
    }
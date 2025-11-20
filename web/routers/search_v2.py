"""
유사도 검색 API V2 (DB 메타데이터 기반)

기존 search.py와의 차이점:
- DB에서 메타데이터 조회
- Object Storage URL 포함
- 제품명, 불량명 등 풍부한 메타데이터 제공
"""
from fastapi import APIRouter, HTTPException, Depends , UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Optional
from sqlalchemy import text
from sqlalchemy.orm import Session
from web.database.connection import get_db
from web.database.models import SearchHistory
from datetime import datetime
import shutil

router = APIRouter(prefix="/v2/search", tags=["search_v2"])

# 전역 변수
_matcher_v2_ref = None
_index_dir_v2_ref = None
_project_root_ref = None


def init_search_v2_router(matcher_v2, index_dir_v2, proj_root):
    """라우터 V2 초기화"""
    global _matcher_v2_ref, _index_dir_v2_ref, _project_root_ref
    _matcher_v2_ref = matcher_v2
    _index_dir_v2_ref = index_dir_v2
    _project_root_ref = proj_root
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

import json

@router.post("/similarity")
async def search_similar_images_v2(request: SearchRequestV2,  db: Session = Depends(get_db)  ):
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
        print(f"[SEARCH V2] 유사도 검색 시작: {query_path.name}, top_k={request.top_k}")
        
        start_time = datetime.now()
        
        # 유사도 검색 수행 (메타데이터 포함)
        result = matcher.search(str(query_path), top_k=request.top_k)

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        print(f"[SEARCH V2] 검색 완료: {len(result.results)}개 결과")


        # ========== DB 저장 ==========
        #db: Session = next(get_db())
        #db: Session = Depends(get_db)  # ✅ FastAPI가 자동으로 세션 관리
        search_id = None
        top1_similarity = None
        
        try:
            if result.results and len(result.results) > 0:
                top1 = result.results[0]
                top1_similarity = top1['similarity_score']
                
                # search_history 테이블에 저장
                search_history = SearchHistory(
                    searched_at=datetime.now(),
                    uploaded_image_path=str(query_path),
                    product_code=top1['product_name'],  # 한글 제품명
                    defect_code=top1['defect_name'],    # 한글 불량명
                    top_k_results=json.dumps(result.model_info, ensure_ascii=False),
                    processing_time=processing_time
                )
                
                db.add(search_history)
                db.commit()
                db.refresh(search_history)
                
                search_id = search_history.search_id
                print(f"[SEARCH V2] DB 저장 완료: search_id={search_id}")
        
        except Exception as e:
            print(f"[SEARCH V2] DB 저장 실패: {e}")
            db.rollback()
        finally:
            db.close()
        
        return JSONResponse(content={
            "status": "success",
            "query_image": str(query_path),
            "index_type": result.index_type,
            "results": result.results,
            "total_gallery_size": result.total_gallery_size,
            "model_info": result.model_info,
            "search_id": search_id,              # 추가
            "top1_similarity": top1_similarity,   # 추가
            "version": "v2"
        })
    
    except Exception as e:
        print(f"[SEARCH V2] 검색 실패: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"검색 실패: {str(e)}")


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


@router.post("/register_defect")
async def register_defect(
    product_name: str = Form(...),
    defect_type: str = Form(...),
    file: Optional[UploadFile] = File(None),
    file_path: Optional[str] = Form(None),  # 추가: 기존 파일 경로
    db: Session = Depends(get_db)
):
    try:
        # 1. product_id 조회
        product_query = text("SELECT product_id FROM products WHERE product_code = :product_code")
        product_result = db.execute(product_query, {"product_code": product_name}).fetchone()
        
        if not product_result:
            raise HTTPException(status_code=404, detail=f"제품 '{product_name}'을 찾을 수 없습니다.")
        
        product_id = product_result[0]
        
        # 2. defect_type_id 조회
        defect_query = text("""
            SELECT defect_type_id 
            FROM defect_types 
            WHERE product_id = :product_id AND defect_code = :defect_code
        """)
        defect_result = db.execute(defect_query, {
            "product_id": product_id,
            "defect_code": defect_type
        }).fetchone()
        
        if not defect_result:
            raise HTTPException(status_code=404, detail=f"불량 유형 '{defect_type}'을 찾을 수 없습니다.")
        
        defect_type_id = defect_result[0]
        
        # 3. 파일명 생성 (중복 방지)
        current_time = datetime.now().strftime("%Y%m%d%H%M%S")
        
        # 4. 파일 처리: 기존 파일 경로 사용 또는 새 파일 업로드
        if file_path:
            print(f"\n[FILE_PATH 모드] 기존 파일 복사")
            # 기존 uploads 폴더의 파일 사용
            source_path = Path(file_path)
            print(f"  - source_path: {source_path}")
            print(f"  - source_path.exists(): {source_path.exists()}")
            
            if not source_path.exists():
                raise HTTPException(status_code=404, detail=f"파일을 찾을 수 없습니다: {file_path}")
            
            file_extension = source_path.suffix
            new_filename = f"{product_name}_{defect_type}_{current_time}{file_extension}"
            print(f"  - new_filename: {new_filename}")
            
            # def_split 폴더로 복사
            local_dir = Path("data/def_split")
            local_dir.mkdir(parents=True, exist_ok=True)
            local_path = local_dir / new_filename
            
            print(f"  - local_dir: {local_dir}")
            print(f"  - local_dir.exists(): {local_dir.exists()}")
            print(f"  - local_path: {local_path}")
            
            # 파일 복사 실행
            print(f"  - 파일 복사 시작...")
            shutil.copy2(source_path, local_path)
            print(f"  - 파일 복사 완료")
            print(f"  - local_path.exists(): {local_path.exists()}")
            
            if local_path.exists():
                file_size = local_path.stat().st_size
                print(f"  - file_size: {file_size}")
            else:
                print(f"  - 경고: 복사 후에도 파일이 존재하지 않음!")
                file_size = source_path.stat().st_size
            
        elif file:
            print(f"\n[FILE 업로드 모드] 새 파일 저장")
            file_extension = Path(file.filename).suffix
            new_filename = f"{product_name}_{defect_type}_{current_time}{file_extension}"
            print(f"  - new_filename: {new_filename}")
            
            local_dir = Path("data/def_split")
            local_dir.mkdir(parents=True, exist_ok=True)
            local_path = local_dir / new_filename
            
            print(f"  - local_path: {local_path}")
            
            contents = await file.read()
            print(f"  - contents length: {len(contents)}")
            
            with open(local_path, "wb") as f:
                f.write(contents)
            
            print(f"  - 파일 쓰기 완료")
            print(f"  - local_path.exists(): {local_path.exists()}")
            
            file_size = len(contents)
        else:
            raise HTTPException(status_code=400, detail="파일 또는 파일 경로를 제공해야 합니다.")
        
        # 5. 경로 설정
        absolute_file_path = str(local_path.absolute())
        relative_storage_url = f"/def_split/{new_filename}"
        
        # 6. Object Storage 업로드
        from web.utils.object_storage import ObjectStorageManager
        
        
        try:
            obs_manager = ObjectStorageManager()
            s3_key = f"def_split/{new_filename}"    
            success = obs_manager.upload_file(str(local_path), s3_key)
            if not success:
                print(f"Object Storage 업로드 실패")
        except Exception as e:
            print(f"Object Storage 업로드 예외: {e}")
        
        # 7. DB에 이미지 정보 저장
        insert_query = text("""
            INSERT INTO images (
                product_id, 
                image_type, 
                defect_type_id, 
                file_name, 
                file_path, 
                file_size, 
                storage_url,
                use_yn,
                sync_yn
            ) VALUES (
                :product_id,
                'defect',
                :defect_type_id,
                :file_name,
                :file_path,
                :file_size,
                :storage_url,
                1,
                1
            )
        """)
        
        db.execute(insert_query, {
            "product_id": product_id,
            "defect_type_id": defect_type_id,
            "file_name": new_filename,
            "file_path": absolute_file_path,
            "file_size": file_size,
            "storage_url": relative_storage_url
        })
        
        db.commit()
        
        return {
            "success": True,
            "message": "불량 이미지가 성공적으로 등록되었습니다.",
            "filename": new_filename,
            "product_id": product_id,
            "defect_type_id": defect_type_id,
            "file_path": absolute_file_path,
            "storage_url": relative_storage_url
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"불량 이미지 등록 실패: {e}")
        raise HTTPException(status_code=500, detail=f"이미지 등록 중 오류가 발생했습니다: {str(e)}")
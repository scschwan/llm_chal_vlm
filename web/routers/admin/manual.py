"""
매뉴얼 관리 API 라우터
"""

from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, List
from pathlib import Path
import shutil

import sys
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from web.database.connection import get_db
from web.database import crud
from web.utils.object_storage import get_obs_manager

router = APIRouter(prefix="/api/admin/manual", tags=["admin-manual"])


# ========================================
# Response 모델
# ========================================

class ManualResponse(BaseModel):
    """매뉴얼 응답"""
    manual_id: int
    product_id: int
    product_code: Optional[str] = None
    product_name: Optional[str] = None
    file_name: str
    file_path: str
    file_size: Optional[int]
    vector_indexed: bool
    indexed_at: Optional[str]
    created_at: str
    display_url: Optional[str] = None
    
    class Config:
        from_attributes = True


# ========================================
# API 엔드포인트
# ========================================

@router.post("")
async def upload_manual(
    product_id: int = Form(...),
    file: UploadFile = File(...)
):
    """
    매뉴얼 업로드 (Object Storage)
    """
    # 제품 존재 확인
    db = next(get_db())
    product = crud.get_product(db, product_id)
    if not product:
        raise HTTPException(404, "제품을 찾을 수 없습니다")
    
    # PDF 파일 체크
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(400, "PDF 파일만 업로드 가능합니다")
    
    try:
        obs = get_obs_manager()
        
        # Object Storage 경로 생성
        s3_key = f"manuals/{product_id}/{file.filename}"
        
        # Object Storage 업로드
        success = obs.upload_fileobj(file.file, s3_key)
        if not success:
            raise HTTPException(500, "파일 업로드 실패")
        
        # 파일 크기 조회
        file_info = obs.get_file_info(s3_key)
        file_size = file_info['size'] if file_info else None
        
        # DB에 저장
        manual = crud.create_manual(
            db,
            product_id=product_id,
            file_name=file.filename,
            file_path=s3_key,
            file_size=file_size
        )
        
        return JSONResponse(content={
            "status": "success",
            "message": "매뉴얼 업로드 완료",
            "manual_id": manual.manual_id,
            "file_name": file.filename,
            "file_path": s3_key,
            "display_url": obs.get_url(s3_key)
        })
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"매뉴얼 업로드 실패: {str(e)}")
    finally:
        db.close()


@router.get("", response_model=List[ManualResponse])
def list_manuals(product_id: Optional[int] = None, db: Session = Depends(get_db)):
    """
    매뉴얼 목록 조회
    """
    try:
        obs = get_obs_manager()
        
        if product_id:
            manuals = crud.get_manuals_by_product(db, product_id)
        else:
            manuals = crud.get_all_manuals(db)
        
        result = []
        for m in manuals:
            product = crud.get_product(db, m.product_id)
            result.append(ManualResponse(
                manual_id=m.manual_id,
                product_id=m.product_id,
                product_code=product.product_code if product else None,
                product_name=product.product_name if product else None,
                file_name=m.file_name,
                file_path=m.file_path,
                file_size=m.file_size,
                vector_indexed=bool(m.vector_indexed),
                indexed_at=str(m.indexed_at) if m.indexed_at else None,
                created_at=str(m.created_at),
                display_url=obs.get_url(m.file_path)
            ))
        
        return result
    
    except Exception as e:
        raise HTTPException(500, f"매뉴얼 목록 조회 실패: {str(e)}")


@router.get("/{manual_id}/download")
def download_manual(manual_id: int, db: Session = Depends(get_db)):
    """
    매뉴얼 다운로드
    """
    manual = crud.get_manual(db, manual_id)
    if not manual:
        raise HTTPException(404, "매뉴얼을 찾을 수 없습니다")
    
    try:
        obs = get_obs_manager()
        
        # 임시 다운로드 경로
        temp_dir = Path("/tmp/manual_downloads")
        temp_dir.mkdir(parents=True, exist_ok=True)
        local_path = temp_dir / manual.file_name
        
        # Object Storage에서 다운로드
        success = obs.download_file(manual.file_path, str(local_path))
        if not success:
            raise HTTPException(500, "파일 다운로드 실패")
        
        return FileResponse(
            path=str(local_path),
            filename=manual.file_name,
            media_type="application/pdf"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"매뉴얼 다운로드 실패: {str(e)}")


@router.delete("/{manual_id}")
def delete_manual(manual_id: int, db: Session = Depends(get_db)):
    """
    매뉴얼 삭제 (Object Storage + DB)
    """
    manual = crud.get_manual(db, manual_id)
    if not manual:
        raise HTTPException(404, "매뉴얼을 찾을 수 없습니다")
    
    try:
        obs = get_obs_manager()
        
        # Object Storage에서 삭제
        obs.delete_file(manual.file_path)
        
        # DB에서 삭제
        success = crud.delete_manual(db, manual_id)
        if success:
            return JSONResponse(content={"status": "success", "message": "매뉴얼이 삭제되었습니다"})
        else:
            raise HTTPException(500, "매뉴얼 삭제 실패")
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"매뉴얼 삭제 실패: {str(e)}")
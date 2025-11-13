"""
불량 유형 관리 API 라우터
"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from typing import Optional, List

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from web.database.connection import get_db
from web.database import crud

router = APIRouter(prefix="/api/admin/defect-type", tags=["admin-defect-type"])


# ========================================
# Request/Response 모델
# ========================================

class DefectTypeCreate(BaseModel):
    """불량 유형 생성 요청"""
    product_id: int = Field(..., description="제품 ID")
    defect_code: str = Field(..., min_length=1, max_length=50, description="불량 코드(영문)")
    defect_name_ko: str = Field(..., min_length=1, max_length=100, description="불량 명칭(한글)")
    defect_name_en: Optional[str] = Field(None, max_length=100, description="불량 명칭(영문)")
    full_name_ko: Optional[str] = Field(None, max_length=200, description="불량 전체 명칭")


class DefectTypeUpdate(BaseModel):
    """불량 유형 수정 요청"""
    defect_code: Optional[str] = Field(None, min_length=1, max_length=50)
    defect_name_ko: Optional[str] = Field(None, min_length=1, max_length=100)
    defect_name_en: Optional[str] = None
    full_name_ko: Optional[str] = None
    is_active: Optional[bool] = None


class DefectTypeResponse(BaseModel):
    """불량 유형 응답"""
    defect_type_id: int
    product_id: int
    product_code: Optional[str] = None
    product_name: Optional[str] = None
    defect_code: str
    defect_name_ko: str
    defect_name_en: Optional[str]
    full_name_ko: Optional[str]
    is_active: bool
    created_at: str
    
    class Config:
        from_attributes = True


# ========================================
# API 엔드포인트
# ========================================

@router.post("", response_model=DefectTypeResponse)
def create_defect_type(request: DefectTypeCreate, db: Session = Depends(get_db)):
    """
    불량 유형 등록
    """
    # 제품 존재 확인
    product = crud.get_product(db, request.product_id)
    if not product:
        raise HTTPException(404, "제품을 찾을 수 없습니다")
    
    try:
        defect_type = crud.create_defect_type(
            db,
            product_id=request.product_id,
            defect_code=request.defect_code,
            defect_name_ko=request.defect_name_ko,
            defect_name_en=request.defect_name_en,
            full_name_ko=request.full_name_ko
        )
        
        return DefectTypeResponse(
            defect_type_id=defect_type.defect_type_id,
            product_id=defect_type.product_id,
            product_code=product.product_code,
            product_name=product.product_name,
            defect_code=defect_type.defect_code,
            defect_name_ko=defect_type.defect_name_ko,
            defect_name_en=defect_type.defect_name_en,
            full_name_ko=defect_type.full_name_ko,
            is_active=bool(defect_type.is_active),
            created_at=str(defect_type.created_at)
        )
    
    except Exception as e:
        raise HTTPException(500, f"불량 유형 등록 실패: {str(e)}")


@router.get("", response_model=List[DefectTypeResponse])
def list_defect_types(product_id: Optional[int] = None, is_active: bool = True,
                      db: Session = Depends(get_db)):
    """
    불량 유형 목록 조회
    """
    try:
        if product_id:
            defect_types = crud.get_defect_types_by_product(db, product_id, is_active)
        else:
            defect_types = crud.get_all_defect_types(db, is_active)
        
        result = []
        for dt in defect_types:
            product = crud.get_product(db, dt.product_id)
            result.append(DefectTypeResponse(
                defect_type_id=dt.defect_type_id,
                product_id=dt.product_id,
                product_code=product.product_code if product else None,
                product_name=product.product_name if product else None,
                defect_code=dt.defect_code,
                defect_name_ko=dt.defect_name_ko,
                defect_name_en=dt.defect_name_en,
                full_name_ko=dt.full_name_ko,
                is_active=bool(dt.is_active),
                created_at=str(dt.created_at)
            ))
        
        return result
    
    except Exception as e:
        raise HTTPException(500, f"불량 유형 목록 조회 실패: {str(e)}")


@router.get("/{defect_type_id}", response_model=DefectTypeResponse)
def get_defect_type(defect_type_id: int, db: Session = Depends(get_db)):
    """
    불량 유형 상세 조회
    """
    defect_type = crud.get_defect_type(db, defect_type_id)
    if not defect_type:
        raise HTTPException(404, "불량 유형을 찾을 수 없습니다")
    
    product = crud.get_product(db, defect_type.product_id)
    
    return DefectTypeResponse(
        defect_type_id=defect_type.defect_type_id,
        product_id=defect_type.product_id,
        product_code=product.product_code if product else None,
        product_name=product.product_name if product else None,
        defect_code=defect_type.defect_code,
        defect_name_ko=defect_type.defect_name_ko,
        defect_name_en=defect_type.defect_name_en,
        full_name_ko=defect_type.full_name_ko,
        is_active=bool(defect_type.is_active),
        created_at=str(defect_type.created_at)
    )


@router.put("/{defect_type_id}", response_model=DefectTypeResponse)
def update_defect_type(defect_type_id: int, request: DefectTypeUpdate, 
                       db: Session = Depends(get_db)):
    """
    불량 유형 수정
    """
    existing = crud.get_defect_type(db, defect_type_id)
    if not existing:
        raise HTTPException(404, "불량 유형을 찾을 수 없습니다")
    
    try:
        update_data = {}
        if request.defect_code is not None:
            update_data['defect_code'] = request.defect_code
        if request.defect_name_ko is not None:
            update_data['defect_name_ko'] = request.defect_name_ko
        if request.defect_name_en is not None:
            update_data['defect_name_en'] = request.defect_name_en
        if request.full_name_ko is not None:
            update_data['full_name_ko'] = request.full_name_ko
        if request.is_active is not None:
            update_data['is_active'] = 1 if request.is_active else 0
        
        defect_type = crud.update_defect_type(db, defect_type_id, **update_data)
        product = crud.get_product(db, defect_type.product_id)
        
        return DefectTypeResponse(
            defect_type_id=defect_type.defect_type_id,
            product_id=defect_type.product_id,
            product_code=product.product_code if product else None,
            product_name=product.product_name if product else None,
            defect_code=defect_type.defect_code,
            defect_name_ko=defect_type.defect_name_ko,
            defect_name_en=defect_type.defect_name_en,
            full_name_ko=defect_type.full_name_ko,
            is_active=bool(defect_type.is_active),
            created_at=str(defect_type.created_at)
        )
    
    except Exception as e:
        raise HTTPException(500, f"불량 유형 수정 실패: {str(e)}")


@router.delete("/{defect_type_id}")
def delete_defect_type(defect_type_id: int, db: Session = Depends(get_db)):
    """
    불량 유형 삭제 (비활성화)
    """
    existing = crud.get_defect_type(db, defect_type_id)
    if not existing:
        raise HTTPException(404, "불량 유형을 찾을 수 없습니다")
    
    try:
        success = crud.delete_defect_type(db, defect_type_id)
        if success:
            return JSONResponse(content={"status": "success", "message": "불량 유형이 비활성화되었습니다"})
        else:
            raise HTTPException(500, "불량 유형 삭제 실패")
    
    except Exception as e:
        raise HTTPException(500, f"불량 유형 삭제 실패: {str(e)}")
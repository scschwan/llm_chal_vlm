"""
제품 관리 API 라우터
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

router = APIRouter(prefix="/api/admin/product", tags=["admin-product"])


# ========================================
# Request/Response 모델
# ========================================

class ProductCreate(BaseModel):
    """제품 생성 요청"""
    product_code: str = Field(..., min_length=1, max_length=50, description="제품 코드")
    product_name: str = Field(..., min_length=1, max_length=100, description="제품명")
    description: Optional[str] = Field(None, description="제품 설명")


class ProductUpdate(BaseModel):
    """제품 수정 요청"""
    product_name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = None
    is_active: Optional[bool] = None


class ProductResponse(BaseModel):
    """제품 응답"""
    product_id: int
    product_code: str
    product_name: str
    description: Optional[str]
    is_active: bool
    created_at: str
    updated_at: str
    
    class Config:
        from_attributes = True


# ========================================
# API 엔드포인트
# ========================================

@router.post("", response_model=ProductResponse)
def create_product(request: ProductCreate, db: Session = Depends(get_db)):
    """
    제품 등록
    """
    # 중복 체크
    existing = crud.get_product_by_code(db, request.product_code)
    if existing:
        raise HTTPException(400, f"제품 코드가 이미 존재합니다: {request.product_code}")
    
    try:
        product = crud.create_product(
            db,
            product_code=request.product_code,
            product_name=request.product_name,
            description=request.description
        )
        
        return ProductResponse(
            product_id=product.product_id,
            product_code=product.product_code,
            product_name=product.product_name,
            description=product.description,
            is_active=bool(product.is_active),
            created_at=str(product.created_at),
            updated_at=str(product.updated_at)
        )
    
    except Exception as e:
        raise HTTPException(500, f"제품 등록 실패: {str(e)}")


@router.get("", response_model=List[ProductResponse])
def list_products(skip: int = 0, limit: int = 100, is_active: bool = True, 
                  db: Session = Depends(get_db)):
    """
    제품 목록 조회
    """
    try:
        products = crud.get_products(db, skip=skip, limit=limit, is_active=is_active)
        
        return [
            ProductResponse(
                product_id=p.product_id,
                product_code=p.product_code,
                product_name=p.product_name,
                description=p.description,
                is_active=bool(p.is_active),
                created_at=str(p.created_at),
                updated_at=str(p.updated_at)
            )
            for p in products
        ]
    
    except Exception as e:
        raise HTTPException(500, f"제품 목록 조회 실패: {str(e)}")


@router.get("/{product_id}", response_model=ProductResponse)
def get_product(product_id: int, db: Session = Depends(get_db)):
    """
    제품 상세 조회
    """
    product = crud.get_product(db, product_id)
    if not product:
        raise HTTPException(404, "제품을 찾을 수 없습니다")
    
    return ProductResponse(
        product_id=product.product_id,
        product_code=product.product_code,
        product_name=product.product_name,
        description=product.description,
        is_active=bool(product.is_active),
        created_at=str(product.created_at),
        updated_at=str(product.updated_at)
    )


@router.put("/{product_id}", response_model=ProductResponse)
def update_product(product_id: int, request: ProductUpdate, db: Session = Depends(get_db)):
    """
    제품 수정
    """
    existing = crud.get_product(db, product_id)
    if not existing:
        raise HTTPException(404, "제품을 찾을 수 없습니다")
    
    try:
        update_data = {}
        if request.product_name is not None:
            update_data['product_name'] = request.product_name
        if request.description is not None:
            update_data['description'] = request.description
        if request.is_active is not None:
            update_data['is_active'] = 1 if request.is_active else 0
        
        product = crud.update_product(db, product_id, **update_data)
        
        return ProductResponse(
            product_id=product.product_id,
            product_code=product.product_code,
            product_name=product.product_name,
            description=product.description,
            is_active=bool(product.is_active),
            created_at=str(product.created_at),
            updated_at=str(product.updated_at)
        )
    
    except Exception as e:
        raise HTTPException(500, f"제품 수정 실패: {str(e)}")


@router.delete("/{product_id}")
def delete_product(product_id: int, db: Session = Depends(get_db)):
    """
    제품 삭제 (비활성화)
    """
    existing = crud.get_product(db, product_id)
    if not existing:
        raise HTTPException(404, "제품을 찾을 수 없습니다")
    
    try:
        success = crud.delete_product(db, product_id)
        if success:
            return JSONResponse(content={"status": "success", "message": "제품이 비활성화되었습니다"})
        else:
            raise HTTPException(500, "제품 삭제 실패")
    
    except Exception as e:
        raise HTTPException(500, f"제품 삭제 실패: {str(e)}")
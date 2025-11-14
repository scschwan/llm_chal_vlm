"""
관리자 페이지 - 이미지 전처리 설정 API
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List
from sqlalchemy.orm import Session

from web.database.connection import get_db
from web.database.crud import (
    create_preprocessing,
    get_all_preprocessing_configs,
    get_preprocessing_by_id,
    get_preprocessing_by_product,
    update_preprocessing,
    delete_preprocessing,
    set_active_preprocessing,
    get_products,
    get_preprocessing_configs_with_product
)

router = APIRouter(prefix="/api/admin/preprocessing", tags=["admin-preprocessing"])


class PreprocessingConfigCreate(BaseModel):
    product_id: int
    grayscale: str = 'N'
    histogram: str = 'N'
    contrast: str = 'N'
    smoothing: str = 'N'
    normalize: str = 'N'


class PreprocessingConfigUpdate(BaseModel):
    grayscale: Optional[str] = None
    histogram: Optional[str] = None
    contrast: Optional[str] = None
    smoothing: Optional[str] = None
    normalize: Optional[str] = None


class PreprocessingConfigResponse(BaseModel):
    preprocessing_id: int
    product_id: int
    product_name: Optional[str] = None
    grayscale: str
    histogram: str
    contrast: str
    smoothing: str
    normalize: str
    is_active: bool
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True


@router.get("/", response_model=List[PreprocessingConfigResponse])
async def get_all_configs(db: Session = Depends(get_db)):
    """전체 전처리 설정 조회"""
    try:
        configs = get_preprocessing_configs_with_product(db)
        return configs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/products")
async def get_product_list(db: Session = Depends(get_db)):
    """제품 목록 조회"""
    try:
        products = get_products(db)
        return {"products": [
            {"product_id": p.product_id, "product_name": p.product_name, "product_code": p.product_code}
            for p in products
        ]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/product/{product_id}")
async def get_config_by_product(product_id: int, db: Session = Depends(get_db)):
    """제품별 전처리 설정 조회"""
    try:
        config = get_preprocessing_by_product(db, product_id)
        if not config:
            raise HTTPException(status_code=404, detail="해당 제품의 전처리 설정이 없습니다.")
        
        return {
            "preprocessing_id": config.preprocessing_id,
            "product_id": config.product_id,
            "grayscale": config.grayscale,
            "histogram": config.histogram,
            "contrast": config.contrast,
            "smoothing": config.smoothing,
            "normalize": config.normalize,
            "is_active": bool(config.is_active),
            "created_at": config.created_at.isoformat() if config.created_at else None,
            "updated_at": config.updated_at.isoformat() if config.updated_at else None
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/", response_model=PreprocessingConfigResponse)
async def create_config(config: PreprocessingConfigCreate, db: Session = Depends(get_db)):
    """전처리 설정 생성"""
    try:
        # 유효성 검증
        valid_values = ['Y', 'N']
        if config.grayscale not in valid_values:
            raise HTTPException(status_code=400, detail="grayscale는 Y 또는 N이어야 합니다.")
        if config.histogram not in valid_values:
            raise HTTPException(status_code=400, detail="histogram은 Y 또는 N이어야 합니다.")
        if config.contrast not in valid_values:
            raise HTTPException(status_code=400, detail="contrast는 Y 또는 N이어야 합니다.")
        if config.smoothing not in valid_values:
            raise HTTPException(status_code=400, detail="smoothing은 Y 또는 N이어야 합니다.")
        if config.normalize not in valid_values:
            raise HTTPException(status_code=400, detail="normalize는 Y 또는 N이어야 합니다.")
        
        new_preprocessing = create_preprocessing(
            db,
            product_id=config.product_id,
            grayscale=config.grayscale,
            histogram=config.histogram,
            contrast=config.contrast,
            smoothing=config.smoothing,
            normalize=config.normalize
        )
        
        return {
            "preprocessing_id": new_preprocessing.preprocessing_id,
            "product_id": new_preprocessing.product_id,
            "grayscale": new_preprocessing.grayscale,
            "histogram": new_preprocessing.histogram,
            "contrast": new_preprocessing.contrast,
            "smoothing": new_preprocessing.smoothing,
            "normalize": new_preprocessing.normalize,
            "is_active": bool(new_preprocessing.is_active),
            "created_at": new_preprocessing.created_at.isoformat(),
            "updated_at": new_preprocessing.updated_at.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{preprocessing_id}", response_model=PreprocessingConfigResponse)
async def update_config(
    preprocessing_id: int,
    config: PreprocessingConfigUpdate,
    db: Session = Depends(get_db)
):
    """전처리 설정 수정"""
    try:
        # 존재 여부 확인
        existing = get_preprocessing_by_id(db, preprocessing_id)
        if not existing:
            raise HTTPException(status_code=404, detail="설정을 찾을 수 없습니다.")
        
        # 유효성 검증
        valid_values = ['Y', 'N']
        update_data = config.dict(exclude_unset=True)
        
        for key, value in update_data.items():
            if value and key in ['grayscale', 'histogram', 'contrast', 'smoothing', 'normalize']:
                if value not in valid_values:
                    raise HTTPException(status_code=400, detail=f"{key}는 Y 또는 N이어야 합니다.")
        
        # 업데이트 실행
        updated = update_preprocessing(db, existing.product_id, **update_data)
        
        return {
            "preprocessing_id": updated.preprocessing_id,
            "product_id": updated.product_id,
            "grayscale": updated.grayscale,
            "histogram": updated.histogram,
            "contrast": updated.contrast,
            "smoothing": updated.smoothing,
            "normalize": updated.normalize,
            "is_active": bool(updated.is_active),
            "created_at": updated.created_at.isoformat(),
            "updated_at": updated.updated_at.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{preprocessing_id}")
async def delete_config(preprocessing_id: int, db: Session = Depends(get_db)):
    """전처리 설정 삭제"""
    try:
        # 존재 여부 확인
        existing = get_preprocessing_by_id(db, preprocessing_id)
        if not existing:
            raise HTTPException(status_code=404, detail="설정을 찾을 수 없습니다.")
        
        # 활성화된 설정은 삭제 불가
        if existing.is_active:
            raise HTTPException(status_code=400, detail="활성화된 설정은 삭제할 수 없습니다.")
        
        success = delete_preprocessing(db, preprocessing_id)
        if not success:
            raise HTTPException(status_code=500, detail="삭제 실패")
        
        return {"message": "설정이 삭제되었습니다."}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{preprocessing_id}/activate")
async def activate_config(preprocessing_id: int, db: Session = Depends(get_db)):
    """전처리 설정 활성화"""
    try:
        # 존재 여부 확인
        existing = get_preprocessing_by_id(db, preprocessing_id)
        if not existing:
            raise HTTPException(status_code=404, detail="설정을 찾을 수 없습니다.")
        
        success = set_active_preprocessing(db, preprocessing_id)
        if not success:
            raise HTTPException(status_code=500, detail="활성화 실패")
        
        return {"message": "설정이 활성화되었습니다."}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/presets/default")
async def get_default_presets():
    """기본 프리셋 목록 조회"""
    presets = [
        {
            "name": "기본 (정규화만)",
            "grayscale": "N",
            "histogram": "N",
            "contrast": "N",
            "smoothing": "N",
            "normalize": "Y"
        },
        {
            "name": "그레이스케일 + 정규화",
            "grayscale": "Y",
            "histogram": "N",
            "contrast": "N",
            "smoothing": "N",
            "normalize": "Y"
        },
        {
            "name": "전체 전처리",
            "grayscale": "Y",
            "histogram": "Y",
            "contrast": "Y",
            "smoothing": "Y",
            "normalize": "Y"
        },
        {
            "name": "히스토그램 평활화",
            "grayscale": "N",
            "histogram": "Y",
            "contrast": "N",
            "smoothing": "N",
            "normalize": "Y"
        }
    ]
    return {"presets": presets}
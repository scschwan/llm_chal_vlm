"""
관리자 페이지 - 모델 선택 API
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List, Dict
from sqlalchemy.orm import Session

from web.database.connection import get_db
from web.database.crud import (
    get_model_params,
    create_model_param,
    update_model_param,
    get_active_model_param,
    set_active_model_param,
    get_products
)

router = APIRouter(prefix="/api/admin/models", tags=["admin-models"])


class ModelParamCreate(BaseModel):
    product_id: int
    model_type: str
    params: Dict


class ModelParamUpdate(BaseModel):
    params: Optional[Dict] = None
    is_active: Optional[bool] = None


class ModelParamResponse(BaseModel):
    param_id: int
    product_id: int
    product_name: Optional[str] = None
    model_type: str
    params: Dict
    is_active: bool
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True


# 사용 가능한 모델 목록
AVAILABLE_MODELS = {
    "clip": [
        {
            "model_id": "ViT-B-32",
            "name": "ViT-B/32",
            "description": "기본 CLIP 모델 (빠른 속도, 균형잡힌 성능)",
            "parameters": 151000000,
            "input_size": 224,
            "performance": "⭐⭐⭐",
            "speed": "⭐⭐⭐⭐⭐"
        },
        {
            "model_id": "ViT-B-16",
            "name": "ViT-B/16",
            "description": "고해상도 CLIP 모델 (높은 정확도)",
            "parameters": 149000000,
            "input_size": 224,
            "performance": "⭐⭐⭐⭐",
            "speed": "⭐⭐⭐"
        },
        {
            "model_id": "ViT-L-14",
            "name": "ViT-L/14",
            "description": "대형 CLIP 모델 (최고 성능)",
            "parameters": 427000000,
            "input_size": 224,
            "performance": "⭐⭐⭐⭐⭐",
            "speed": "⭐⭐"
        }
    ],
    "patchcore": [
        {
            "model_id": "wide_resnet50_2",
            "name": "WideResNet50",
            "description": "기본 PatchCore 백본 (권장)",
            "parameters": 68900000,
            "input_size": 224,
            "performance": "⭐⭐⭐⭐",
            "speed": "⭐⭐⭐⭐"
        },
        {
            "model_id": "resnet18",
            "name": "ResNet18",
            "description": "경량 백본 (빠른 속도)",
            "parameters": 11700000,
            "input_size": 224,
            "performance": "⭐⭐⭐",
            "speed": "⭐⭐⭐⭐⭐"
        }
    ]
}


@router.get("/available")
async def get_available_models():
    """사용 가능한 모델 목록 조회"""
    return {
        "models": AVAILABLE_MODELS
    }


@router.get("/", response_model=List[ModelParamResponse])
async def get_all_model_params(db: Session = Depends(get_db)):
    """전체 모델 설정 조회"""
    try:
        from sqlalchemy import text
        
        query = text("""
            SELECT m.*, p.product_name
            FROM model_params m
            LEFT JOIN products p ON m.product_id = p.product_id
            ORDER BY m.created_at DESC
        """)
        
        result = db.execute(query)
        rows = result.fetchall()
        
        configs = []
        for row in rows:
            import json
            config = {
                'param_id': row[0],
                'product_id': row[1],
                'model_type': row[2],
                'params': json.loads(row[3]) if isinstance(row[3], str) else row[3],
                'is_active': bool(row[4]),
                'created_at': row[5].isoformat() if row[5] else None,
                'updated_at': row[6].isoformat() if row[6] else None,
                'product_name': row[7] if len(row) > 7 else None
            }
            configs.append(config)
        
        return configs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/current")
async def get_current_models(db: Session = Depends(get_db)):
    """현재 선택된 모델 조회"""
    try:
        from sqlalchemy import text
        
        # CLIP 모델
        clip_query = text("""
            SELECT m.*, p.product_name
            FROM model_params m
            LEFT JOIN products p ON m.product_id = p.product_id
            WHERE m.model_type = 'clip' AND m.is_active = 1
            LIMIT 1
        """)
        
        clip_result = db.execute(clip_query).fetchone()
        
        # PatchCore 모델
        patchcore_query = text("""
            SELECT m.*, p.product_name
            FROM model_params m
            LEFT JOIN products p ON m.product_id = p.product_id
            WHERE m.model_type = 'patchcore' AND m.is_active = 1
            LIMIT 1
        """)
        
        patchcore_result = db.execute(patchcore_query).fetchone()
        
        import json
        
        current_models = {
            'clip': None,
            'patchcore': None
        }
        
        if clip_result:
            current_models['clip'] = {
                'param_id': clip_result[0],
                'product_id': clip_result[1],
                'model_type': clip_result[2],
                'params': json.loads(clip_result[3]) if isinstance(clip_result[3], str) else clip_result[3],
                'is_active': bool(clip_result[4]),
                'product_name': clip_result[7] if len(clip_result) > 7 else None
            }
        
        if patchcore_result:
            current_models['patchcore'] = {
                'param_id': patchcore_result[0],
                'product_id': patchcore_result[1],
                'model_type': patchcore_result[2],
                'params': json.loads(patchcore_result[3]) if isinstance(patchcore_result[3], str) else patchcore_result[3],
                'is_active': bool(patchcore_result[4]),
                'product_name': patchcore_result[7] if len(patchcore_result) > 7 else None
            }
        
        return current_models
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{model_id}/info")
async def get_model_info(model_id: str):
    """모델 상세 정보 조회"""
    try:
        # CLIP 모델 검색
        for model in AVAILABLE_MODELS['clip']:
            if model['model_id'] == model_id:
                return {
                    'model_type': 'clip',
                    'model_info': model
                }
        
        # PatchCore 모델 검색
        for model in AVAILABLE_MODELS['patchcore']:
            if model['model_id'] == model_id:
                return {
                    'model_type': 'patchcore',
                    'model_info': model
                }
        
        raise HTTPException(status_code=404, detail="모델을 찾을 수 없습니다.")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/select", response_model=ModelParamResponse)
async def select_model(config: ModelParamCreate, db: Session = Depends(get_db)):
    """모델 선택 (설정 생성)"""
    try:
        # 모델 타입 검증
        if config.model_type not in ['clip', 'patchcore']:
            raise HTTPException(status_code=400, detail="model_type은 'clip' 또는 'patchcore'여야 합니다.")
        
        # 모델 ID 검증
        model_id = config.params.get('model_id')
        if not model_id:
            raise HTTPException(status_code=400, detail="params에 model_id가 필요합니다.")
        
        # 유효한 모델인지 확인
        valid_models = [m['model_id'] for m in AVAILABLE_MODELS.get(config.model_type, [])]
        if model_id not in valid_models:
            raise HTTPException(status_code=400, detail=f"유효하지 않은 모델 ID: {model_id}")
        
        # 모델 설정 생성
        new_param = create_model_param(
            db,
            product_id=config.product_id,
            model_type=config.model_type,
            params=config.params
        )
        
        return {
            'param_id': new_param.param_id,
            'product_id': new_param.product_id,
            'model_type': new_param.model_type,
            'params': new_param.params,
            'is_active': bool(new_param.is_active),
            'created_at': new_param.created_at.isoformat(),
            'updated_at': new_param.updated_at.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{param_id}/activate")
async def activate_model(param_id: int, db: Session = Depends(get_db)):
    """모델 설정 활성화"""
    try:
        from web.database.crud import get_model_param_by_id, set_active_model_param
        
        # 존재 여부 확인
        existing = get_model_param_by_id(db, param_id)
        if not existing:
            raise HTTPException(status_code=404, detail="설정을 찾을 수 없습니다.")
        
        success = set_active_model_param(db, param_id)
        if not success:
            raise HTTPException(status_code=500, detail="활성화 실패")
        
        return {"message": f"{existing.model_type} 모델이 활성화되었습니다."}
        
    except HTTPException:
        raise
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
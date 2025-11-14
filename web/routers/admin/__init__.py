"""
관리자 라우터 모듈
"""

from .product import router as product_router
from .manual import router as manual_router
from .defect_type import router as defect_type_router
from .image import router as image_router
from .deployment import router as deployment_router
from .preprocessing import router as preprocessing_router

__all__ = [
    'product_router',
    'manual_router', 
    'defect_type_router',
    'image_router',
    'deployment_router',
    'preprocessing_router'
]
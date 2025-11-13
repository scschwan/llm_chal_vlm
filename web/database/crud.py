"""
CRUD 함수 (Create, Read, Update, Delete)
"""

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
from typing import List, Optional, Dict, Any
from datetime import datetime

from .models import (
    Product, Manual, DefectType, Image, ImagePreprocessing,
    SearchHistory, ResponseHistory, ModelParams, DeploymentLog, SystemConfig
)


# ========================================
# Product CRUD
# ========================================

def create_product(db: Session, product_code: str, product_name: str, description: str = None) -> Product:
    """제품 생성"""
    product = Product(
        product_code=product_code,
        product_name=product_name,
        description=description
    )
    db.add(product)
    db.commit()
    db.refresh(product)
    return product


def get_product(db: Session, product_id: int) -> Optional[Product]:
    """제품 조회 (ID)"""
    return db.query(Product).filter(Product.product_id == product_id).first()


def get_product_by_code(db: Session, product_code: str) -> Optional[Product]:
    """제품 조회 (코드)"""
    return db.query(Product).filter(Product.product_code == product_code).first()


def get_products(db: Session, skip: int = 0, limit: int = 100, is_active: bool = True) -> List[Product]:
    """제품 목록 조회"""
    query = db.query(Product)
    if is_active is not None:
        query = query.filter(Product.is_active == (1 if is_active else 0))
    return query.offset(skip).limit(limit).all()


def update_product(db: Session, product_id: int, **kwargs) -> Optional[Product]:
    """제품 수정"""
    product = get_product(db, product_id)
    if product:
        for key, value in kwargs.items():
            if hasattr(product, key):
                setattr(product, key, value)
        db.commit()
        db.refresh(product)
    return product


def delete_product(db: Session, product_id: int) -> bool:
    """제품 삭제 (비활성화)"""
    product = get_product(db, product_id)
    if product:
        product.is_active = 0
        db.commit()
        return True
    return False


# ========================================
# Manual CRUD
# ========================================

def create_manual(db: Session, product_id: int, file_name: str, file_path: str, 
                  file_size: int = None) -> Manual:
    """매뉴얼 생성"""
    manual = Manual(
        product_id=product_id,
        file_name=file_name,
        file_path=file_path,
        file_size=file_size
    )
    db.add(manual)
    db.commit()
    db.refresh(manual)
    return manual


def get_manual(db: Session, manual_id: int) -> Optional[Manual]:
    """매뉴얼 조회"""
    return db.query(Manual).filter(Manual.manual_id == manual_id).first()


def get_manuals_by_product(db: Session, product_id: int) -> List[Manual]:
    """제품별 매뉴얼 목록"""
    return db.query(Manual).filter(Manual.product_id == product_id).all()


def get_all_manuals(db: Session) -> List[Manual]:
    """전체 매뉴얼 목록"""
    return db.query(Manual).all()


def update_manual_index_status(db: Session, manual_id: int, indexed: bool = True) -> Optional[Manual]:
    """매뉴얼 인덱싱 상태 업데이트"""
    manual = get_manual(db, manual_id)
    if manual:
        manual.vector_indexed = 1 if indexed else 0
        manual.indexed_at = datetime.now() if indexed else None
        db.commit()
        db.refresh(manual)
    return manual


def delete_manual(db: Session, manual_id: int) -> bool:
    """매뉴얼 삭제"""
    manual = get_manual(db, manual_id)
    if manual:
        db.delete(manual)
        db.commit()
        return True
    return False


# ========================================
# DefectType CRUD
# ========================================

def create_defect_type(db: Session, product_id: int, defect_code: str, 
                       defect_name_ko: str, defect_name_en: str = None,
                       full_name_ko: str = None) -> DefectType:
    """불량 유형 생성"""
    defect_type = DefectType(
        product_id=product_id,
        defect_code=defect_code,
        defect_name_ko=defect_name_ko,
        defect_name_en=defect_name_en,
        full_name_ko=full_name_ko
    )
    db.add(defect_type)
    db.commit()
    db.refresh(defect_type)
    return defect_type


def get_defect_type(db: Session, defect_type_id: int) -> Optional[DefectType]:
    """불량 유형 조회"""
    return db.query(DefectType).filter(DefectType.defect_type_id == defect_type_id).first()


def get_defect_types_by_product(db: Session, product_id: int, is_active: bool = True) -> List[DefectType]:
    """제품별 불량 유형 목록"""
    query = db.query(DefectType).filter(DefectType.product_id == product_id)
    if is_active is not None:
        query = query.filter(DefectType.is_active == (1 if is_active else 0))
    return query.all()


def get_all_defect_types(db: Session, is_active: bool = True) -> List[DefectType]:
    """전체 불량 유형 목록"""
    query = db.query(DefectType)
    if is_active is not None:
        query = query.filter(DefectType.is_active == (1 if is_active else 0))
    return query.all()


def update_defect_type(db: Session, defect_type_id: int, **kwargs) -> Optional[DefectType]:
    """불량 유형 수정"""
    defect_type = get_defect_type(db, defect_type_id)
    if defect_type:
        for key, value in kwargs.items():
            if hasattr(defect_type, key):
                setattr(defect_type, key, value)
        db.commit()
        db.refresh(defect_type)
    return defect_type


def delete_defect_type(db: Session, defect_type_id: int) -> bool:
    """불량 유형 삭제 (비활성화)"""
    defect_type = get_defect_type(db, defect_type_id)
    if defect_type:
        defect_type.is_active = 0
        db.commit()
        return True
    return False


# ========================================
# Image CRUD
# ========================================

def create_image(db: Session, product_id: int, image_type: str, file_name: str,
                 file_path: str, defect_type_id: int = None, file_size: int = None) -> Image:
    """이미지 생성"""
    image = Image(
        product_id=product_id,
        image_type=image_type,
        defect_type_id=defect_type_id,
        file_name=file_name,
        file_path=file_path,
        file_size=file_size
    )
    db.add(image)
    db.commit()
    db.refresh(image)
    return image


def get_image(db: Session, image_id: int) -> Optional[Image]:
    """이미지 조회"""
    return db.query(Image).filter(Image.image_id == image_id).first()


def get_images_by_product(db: Session, product_id: int, image_type: str = None) -> List[Image]:
    """제품별 이미지 목록"""
    query = db.query(Image).filter(Image.product_id == product_id)
    if image_type:
        query = query.filter(Image.image_type == image_type)
    return query.all()


def get_images_by_defect_type(db: Session, defect_type_id: int) -> List[Image]:
    """불량 유형별 이미지 목록"""
    return db.query(Image).filter(Image.defect_type_id == defect_type_id).all()


def delete_image(db: Session, image_id: int) -> bool:
    """이미지 삭제"""
    image = get_image(db, image_id)
    if image:
        db.delete(image)
        db.commit()
        return True
    return False


# ========================================
# ImagePreprocessing CRUD
# ========================================

def create_preprocessing(db: Session, product_id: int, **options) -> ImagePreprocessing:
    """전처리 설정 생성"""
    preprocessing = ImagePreprocessing(product_id=product_id, **options)
    db.add(preprocessing)
    db.commit()
    db.refresh(preprocessing)
    return preprocessing


def get_preprocessing_by_product(db: Session, product_id: int) -> Optional[ImagePreprocessing]:
    """제품별 전처리 설정 조회"""
    return db.query(ImagePreprocessing).filter(
        ImagePreprocessing.product_id == product_id,
        ImagePreprocessing.is_active == 1
    ).first()


def update_preprocessing(db: Session, product_id: int, **options) -> Optional[ImagePreprocessing]:
    """전처리 설정 수정"""
    preprocessing = get_preprocessing_by_product(db, product_id)
    if preprocessing:
        for key, value in options.items():
            if hasattr(preprocessing, key):
                setattr(preprocessing, key, value)
        db.commit()
        db.refresh(preprocessing)
    return preprocessing
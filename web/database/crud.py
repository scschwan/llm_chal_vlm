"""
CRUD 함수 (Create, Read, Update, Delete)
"""

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
from typing import List, Optional, Dict, Any
from datetime import datetime
import json
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


def get_all_preprocessing_configs(db: Session) -> List[ImagePreprocessing]:
    """전체 전처리 설정 조회 (제품 정보 포함)"""
    from sqlalchemy.orm import joinedload
    return db.query(ImagePreprocessing).all()


def get_preprocessing_by_id(db: Session, preprocessing_id: int) -> Optional[ImagePreprocessing]:
    """전처리 설정 ID로 조회"""
    return db.query(ImagePreprocessing).filter(
        ImagePreprocessing.preprocessing_id == preprocessing_id
    ).first()


def set_active_preprocessing(db: Session, preprocessing_id: int) -> bool:
    """전처리 설정 활성화 (같은 제품의 다른 설정은 비활성화)"""
    preprocessing = get_preprocessing_by_id(db, preprocessing_id)
    if not preprocessing:
        return False
    
    # 같은 제품의 모든 설정 비활성화
    db.query(ImagePreprocessing).filter(
        ImagePreprocessing.product_id == preprocessing.product_id
    ).update({ImagePreprocessing.is_active: 0})
    
    # 선택한 설정 활성화
    preprocessing.is_active = 1
    db.commit()
    return True


def delete_preprocessing(db: Session, preprocessing_id: int) -> bool:
    """전처리 설정 삭제"""
    preprocessing = get_preprocessing_by_id(db, preprocessing_id)
    if preprocessing:
        db.delete(preprocessing)
        db.commit()
        return True
    return False


def get_preprocessing_configs_with_product(db: Session) -> list:
    """전처리 설정 목록 조회 (제품명 포함)"""
    from sqlalchemy import text
    
    query = text("""
        SELECT p.*, pr.product_name
        FROM image_preprocessing p
        LEFT JOIN products pr ON p.product_id = pr.product_id
        ORDER BY p.created_at DESC
    """)
    
    result = db.execute(query)
    rows = result.fetchall()
    
    configs = []
    for row in rows:
        config = {
            'preprocessing_id': row[0],
            'product_id': row[1],
            'grayscale': row[2],
            'histogram': row[3],
            'contrast': row[4],
            'smoothing': row[5],
            'normalize': row[6],
            'is_active': bool(row[7]),
            'created_at': row[8].isoformat() if row[8] else None,
            'updated_at': row[9].isoformat() if row[9] else None,
            'product_name': row[10] if len(row) > 10 else None
        }
        configs.append(config)
    
    return configs

# ========================================
# Deploy CRUD
# ========================================


def create_deployment_log(
    deployment_type: str,
    target: str,
    status: str = 'pending',
    product_id: int = None
) -> int:
    """배포 로그 생성"""
    from web.database.connection import get_db_connection
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        query = """
            INSERT INTO deployment_logs 
            (deploy_type, product_id, status, started_at, deployed_by)
            VALUES (%s, %s, %s, %s, %s)
        """
        now = datetime.now()
        cursor.execute(query, (deployment_type, product_id, status, now, 'admin'))
        conn.commit()
        
        log_id = cursor.lastrowid
        return log_id
        
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cursor.close()
        conn.close()


def update_deployment_status(
    log_id: int,
    status: str,
    result_data: dict = None,
    error_message: str = None
):
    """배포 상태 업데이트"""
    from web.database.connection import get_db_connection
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        query = """
            UPDATE deployment_logs
            SET status = %s,
                completed_at = %s,
                result_data = %s,
                result_message = %s
            WHERE deploy_id = %s
        """
        now = datetime.now()
        result_json = json.dumps(result_data) if result_data else None
        
        cursor.execute(query, (
            status,
            now,
            result_json,
            error_message,
            log_id
        ))
        conn.commit()
        
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cursor.close()
        conn.close()


def get_deployment_logs(
    limit: int = 20,
    deployment_type: str = None
) -> list:
    """배포 이력 조회"""
    from web.database.connection import get_db_connection
    
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    try:
        if deployment_type:
            query = """
                SELECT *
                FROM deployment_logs
                WHERE deploy_type = %s
                ORDER BY started_at DESC
                LIMIT %s
            """
            cursor.execute(query, (deployment_type, limit))
        else:
            query = """
                SELECT *
                FROM deployment_logs
                ORDER BY started_at DESC
                LIMIT %s
            """
            cursor.execute(query, (limit,))
        
        logs = cursor.fetchall()
        
        # 날짜 형식 변환
        for log in logs:
            if log.get('started_at'):
                log['start_time'] = log['started_at'].isoformat()
            if log.get('completed_at'):
                log['end_time'] = log['completed_at'].isoformat()
            # API 호환성을 위한 필드명 매핑
            log['target'] = log.get('product_id', 'all')
            log['deployment_type'] = log.get('deploy_type')
            log['error_message'] = log.get('result_message')
        
        return logs
        
    except Exception as e:
        raise e
    finally:
        cursor.close()
        conn.close()



# ========================================
# Preprocessing CRUD
# ========================================


def create_preprocessing_config(
    name: str,
    resize_width: int = 224,
    resize_height: int = 224,
    normalize: bool = True,
    augmentation: dict = None
) -> int:
    """전처리 설정 생성"""
    from web.database.connection import get_db_connection
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        query = """
            INSERT INTO preprocessing_configs 
            (name, resize_width, resize_height, normalize, augmentation, created_at)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        now = datetime.now()
        augmentation_json = json.dumps(augmentation) if augmentation else None
        
        cursor.execute(query, (name, resize_width, resize_height, normalize, augmentation_json, now))
        conn.commit()
        
        config_id = cursor.lastrowid
        return config_id
        
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cursor.close()
        conn.close()


def get_preprocessing_configs() -> list:
    """전체 전처리 설정 조회"""
    from web.database.connection import get_db_connection
    
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    try:
        query = """
            SELECT *
            FROM preprocessing_configs
            ORDER BY created_at DESC
        """
        cursor.execute(query)
        configs = cursor.fetchall()
        
        # 날짜 및 JSON 형식 변환
        for config in configs:
            if config.get('created_at'):
                config['created_at'] = config['created_at'].isoformat()
            if config.get('updated_at'):
                config['updated_at'] = config['updated_at'].isoformat()
            if config.get('augmentation') and isinstance(config['augmentation'], str):
                config['augmentation'] = json.loads(config['augmentation'])
        
        return configs
        
    except Exception as e:
        raise e
    finally:
        cursor.close()
        conn.close()


def get_preprocessing_config_by_id(config_id: int) -> dict:
    """ID로 전처리 설정 조회"""
    from web.database.connection import get_db_connection
    
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    try:
        query = "SELECT * FROM preprocessing_configs WHERE id = %s"
        cursor.execute(query, (config_id,))
        config = cursor.fetchone()
        
        if config:
            if config.get('created_at'):
                config['created_at'] = config['created_at'].isoformat()
            if config.get('updated_at'):
                config['updated_at'] = config['updated_at'].isoformat()
            if config.get('augmentation') and isinstance(config['augmentation'], str):
                config['augmentation'] = json.loads(config['augmentation'])
        
        return config
        
    except Exception as e:
        raise e
    finally:
        cursor.close()
        conn.close()


def update_preprocessing_config(config_id: int, **kwargs):
    """전처리 설정 수정"""
    from web.database.connection import get_db_connection
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # 업데이트할 필드 구성
        update_fields = []
        values = []
        
        if 'name' in kwargs:
            update_fields.append("name = %s")
            values.append(kwargs['name'])
        if 'resize_width' in kwargs:
            update_fields.append("resize_width = %s")
            values.append(kwargs['resize_width'])
        if 'resize_height' in kwargs:
            update_fields.append("resize_height = %s")
            values.append(kwargs['resize_height'])
        if 'normalize' in kwargs:
            update_fields.append("normalize = %s")
            values.append(kwargs['normalize'])
        if 'augmentation' in kwargs:
            update_fields.append("augmentation = %s")
            values.append(json.dumps(kwargs['augmentation']) if kwargs['augmentation'] else None)
        
        update_fields.append("updated_at = %s")
        values.append(datetime.now())
        
        values.append(config_id)
        
        query = f"""
            UPDATE preprocessing_configs
            SET {', '.join(update_fields)}
            WHERE id = %s
        """
        
        cursor.execute(query, values)
        conn.commit()
        
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cursor.close()
        conn.close()


def delete_preprocessing_config(config_id: int):
    """전처리 설정 삭제"""
    from web.database.connection import get_db_connection
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        query = "DELETE FROM preprocessing_configs WHERE id = %s"
        cursor.execute(query, (config_id,))
        conn.commit()
        
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cursor.close()
        conn.close()


def get_active_preprocessing_config() -> dict:
    """활성화된 전처리 설정 조회"""
    from web.database.connection import get_db_connection
    
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    try:
        query = "SELECT * FROM preprocessing_configs WHERE is_active = TRUE LIMIT 1"
        cursor.execute(query)
        config = cursor.fetchone()
        
        if config:
            if config.get('created_at'):
                config['created_at'] = config['created_at'].isoformat()
            if config.get('updated_at'):
                config['updated_at'] = config['updated_at'].isoformat()
            if config.get('augmentation') and isinstance(config['augmentation'], str):
                config['augmentation'] = json.loads(config['augmentation'])
        
        return config
        
    except Exception as e:
        raise e
    finally:
        cursor.close()
        conn.close()


def set_active_preprocessing_config(config_id: int):
    """전처리 설정 활성화 (다른 설정은 비활성화)"""
    from web.database.connection import get_db_connection
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # 모든 설정 비활성화
        cursor.execute("UPDATE preprocessing_configs SET is_active = FALSE")
        
        # 선택한 설정 활성화
        cursor.execute("UPDATE preprocessing_configs SET is_active = TRUE WHERE id = %s", (config_id,))
        
        conn.commit()
        
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cursor.close()
        conn.close()
"""
기존 Object Storage 이미지 → DB 일괄 등록
실행: python sync_storage_to_db.py
"""

import os
import sys
from pathlib import Path
from typing import List, Dict
import re

# 프로젝트 루트 경로 추가
sys.path.append('/home/dmillion/llm_chal_vlm')

from connection import get_db_connection

# 환경변수 로드
NCP_STORAGE_BASE_URL = os.getenv('NCP_STORAGE_BASE_URL', 'https://kr.object.ncloudstorage.com')
NCP_BUCKET = os.getenv('NCP_BUCKET', 'dm-obs')


def parse_defect_filename(filename: str) -> Dict[str, str]:
    """
    불량 이미지 파일명 파싱
    예: carpet_thread_012.png → {product: carpet, defect: thread, seq: 012}
    """
    stem = Path(filename).stem
    parts = stem.split('_')
    
    if len(parts) >= 3:
        return {
            'product_code': parts[0],
            'defect_code': parts[1],
            'sequence': parts[2]
        }
    elif len(parts) == 2:
        return {
            'product_code': parts[0],
            'defect_code': parts[1],
            'sequence': '000'
        }
    else:
        return {
            'product_code': parts[0] if parts else 'unknown',
            'defect_code': 'unknown',
            'sequence': '000'
        }


def parse_normal_filename(filename: str) -> Dict[str, str]:
    """
    정상 이미지 파일명 파싱
    예: carpet_ok_027.png → {product: carpet, seq: 027}
    """
    stem = Path(filename).stem
    parts = stem.split('_')
    
    if len(parts) >= 3:
        return {
            'product_code': parts[0],
            'sequence': parts[2]
        }
    elif len(parts) >= 2:
        return {
            'product_code': parts[0],
            'sequence': parts[1] if parts[1] != 'ok' else '000'
        }
    else:
        return {
            'product_code': parts[0] if parts else 'unknown',
            'sequence': '000'
        }


def get_product_id(conn, product_code: str) -> int:
    """제품 코드로 product_id 조회"""
    cursor = conn.cursor()
    try:
        query = "SELECT product_id FROM products WHERE product_code = %s"
        cursor.execute(query, (product_code,))
        result = cursor.fetchone()
        
        if result:
            return result[0]
        else:
            print(f"⚠️  제품을 찾을 수 없음: {product_code}")
            return None
    finally:
        cursor.close()


def get_defect_type_id(conn, product_id: int, defect_code: str) -> int:
    """불량 코드로 defect_type_id 조회"""
    cursor = conn.cursor()
    try:
        query = """
            SELECT defect_type_id 
            FROM defect_types 
            WHERE product_id = %s AND defect_code = %s
        """
        cursor.execute(query, (product_id, defect_code))
        result = cursor.fetchone()
        
        if result:
            return result[0]
        else:
            print(f"⚠️  불량 유형을 찾을 수 없음: product_id={product_id}, defect_code={defect_code}")
            return None
    finally:
        cursor.close()


def insert_image(conn, image_data: Dict) -> bool:
    """이미지 DB 삽입"""
    cursor = conn.cursor()
    try:
        query = """
            INSERT INTO images 
            (product_id, image_type, defect_type_id, file_name, file_path, storage_url)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        
        cursor.execute(query, (
            image_data['product_id'],
            image_data['image_type'],
            image_data.get('defect_type_id'),
            image_data['file_name'],
            image_data['file_path'],
            image_data['storage_url']
        ))
        
        conn.commit()
        return True
        
    except Exception as e:
        print(f"❌ DB 삽입 실패: {image_data['file_name']} - {e}")
        conn.rollback()
        return False
    finally:
        cursor.close()


def scan_defect_images() -> List[Dict]:
    """불량 이미지 스캔"""
    defect_dir = Path('/home/dmillion/llm_chal_vlm/data/def_split')
    
    if not defect_dir.exists():
        print(f"⚠️  디렉토리 없음: {defect_dir}")
        return []
    
    images = []
    
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        for img_path in defect_dir.glob(ext):
            filename = img_path.name
            parsed = parse_defect_filename(filename)
            
            storage_url = f"{NCP_STORAGE_BASE_URL}/{NCP_BUCKET}/def_split/{filename}"
            
            images.append({
                'file_name': filename,
                'file_path': str(img_path),
                'storage_url': storage_url,
                'product_code': parsed['product_code'],
                'defect_code': parsed['defect_code'],
                'image_type': 'defect'
            })
    
    return images


def scan_normal_images() -> List[Dict]:
    """정상 이미지 스캔"""
    base_dir = Path('/home/dmillion/llm_chal_vlm/data/patchCore')
    
    if not base_dir.exists():
        print(f"⚠️  디렉토리 없음: {base_dir}")
        return []
    
    images = []
    
    # 각 제품 폴더 순회
    for product_dir in base_dir.iterdir():
        if not product_dir.is_dir():
            continue
        
        product_code = product_dir.name
        ok_dir = product_dir / 'ok'
        
        if not ok_dir.exists():
            continue
        
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            for img_path in ok_dir.glob(ext):
                filename = img_path.name
                parsed = parse_normal_filename(filename)
                
                # Object Storage 경로: ok_image/{product}/{filename}
                storage_url = f"{NCP_STORAGE_BASE_URL}/{NCP_BUCKET}/ok_image/{product_code}/{filename}"
                
                images.append({
                    'file_name': filename,
                    'file_path': str(img_path),
                    'storage_url': storage_url,
                    'product_code': parsed['product_code'],
                    'image_type': 'normal'
                })
    
    return images


def sync_images_to_db():
    """이미지 → DB 동기화 메인 함수"""
    
    print("=" * 60)
    print("Object Storage 이미지 → DB 동기화 시작")
    print("=" * 60)
    
    # DB 연결
    conn = get_db_connection()
    
    try:
        # 1. 불량 이미지 스캔
        print("\n[1/4] 불량 이미지 스캔 중...")
        defect_images = scan_defect_images()
        print(f"✅ 불량 이미지 {len(defect_images)}개 발견")
        
        # 2. 정상 이미지 스캔
        print("\n[2/4] 정상 이미지 스캔 중...")
        normal_images = scan_normal_images()
        print(f"✅ 정상 이미지 {len(normal_images)}개 발견")
        
        # 3. 불량 이미지 DB 삽입
        print("\n[3/4] 불량 이미지 DB 삽입 중...")
        defect_success = 0
        defect_fail = 0
        
        for img in defect_images:
            product_id = get_product_id(conn, img['product_code'])
            if not product_id:
                defect_fail += 1
                continue
            
            defect_type_id = get_defect_type_id(conn, product_id, img['defect_code'])
            
            image_data = {
                'product_id': product_id,
                'image_type': img['image_type'],
                'defect_type_id': defect_type_id,
                'file_name': img['file_name'],
                'file_path': img['file_path'],
                'storage_url': img['storage_url']
            }
            
            if insert_image(conn, image_data):
                defect_success += 1
            else:
                defect_fail += 1
        
        print(f"✅ 불량 이미지 삽입 완료: {defect_success}개 성공, {defect_fail}개 실패")
        
        # 4. 정상 이미지 DB 삽입
        print("\n[4/4] 정상 이미지 DB 삽입 중...")
        normal_success = 0
        normal_fail = 0
        
        for img in normal_images:
            product_id = get_product_id(conn, img['product_code'])
            if not product_id:
                normal_fail += 1
                continue
            
            image_data = {
                'product_id': product_id,
                'image_type': img['image_type'],
                'defect_type_id': None,
                'file_name': img['file_name'],
                'file_path': img['file_path'],
                'storage_url': img['storage_url']
            }
            
            if insert_image(conn, image_data):
                normal_success += 1
            else:
                normal_fail += 1
        
        print(f"✅ 정상 이미지 삽입 완료: {normal_success}개 성공, {normal_fail}개 실패")
        
        # 최종 결과
        print("\n" + "=" * 60)
        print("동기화 완료")
        print("=" * 60)
        print(f"불량 이미지: {defect_success}/{len(defect_images)}")
        print(f"정상 이미지: {normal_success}/{len(normal_images)}")
        print(f"총 성공: {defect_success + normal_success}")
        print(f"총 실패: {defect_fail + normal_fail}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        conn.close()


if __name__ == "__main__":
    sync_images_to_db()
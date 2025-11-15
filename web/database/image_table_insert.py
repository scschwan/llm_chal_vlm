"""
기존 Object Storage 이미지 → DB 일괄 등록
실행: python sync_storage_to_db.py
"""

import os
import sys
from pathlib import Path
from typing import List, Dict

# 프로젝트 루트 경로 추가
sys.path.append('/home/dmillion/llm_chal_vlm')


# ✅ 환경변수 확인 및 출력
print(f"DB_HOST: {os.getenv('DB_HOST', 'NOT_SET')}")
print(f"DB_USER: {os.getenv('DB_USER', 'NOT_SET')}")
print(f"DB_NAME: {os.getenv('DB_NAME', 'NOT_SET')}")

os.environ['DB_HOST'] = 'localhost'
os.environ['DB_USER'] = 'dmillion'
os.environ['DB_PASSWORD'] = 'dm250120@'
os.environ['DB_NAME'] = 'defect_detection_db'


# ✅ 기존 connection.py 사용
from web.database.connection import SessionLocal
from web.database.models import Product, DefectType, Image

# 환경변수 로드
NCP_STORAGE_BASE_URL = os.getenv('NCP_STORAGE_BASE_URL', 'https://kr.object.ncloudstorage.com')
NCP_BUCKET = os.getenv('NCP_BUCKET', 'dm-obs')


def get_db_session():
    """DB 세션 생성 (기존 connection.py 활용)"""
    return SessionLocal()


def parse_defect_filename(filename: str) -> Dict[str, str]:
    """불량 이미지 파일명 파싱"""
    stem = Path(filename).stem
    parts = stem.split('_')
    
    if len(parts) < 2:
        return {
            'product_code': parts[0] if parts else 'unknown',
            'defect_code': 'unknown',
            'sequence': '000'
        }
    
    product_code = parts[0]
    
    seq_index = -1
    for i in range(len(parts) - 1, 0, -1):
        if parts[i].isdigit() or parts[i].startswith('r') or parts[i].startswith('fh'):
            seq_index = i
        else:
            break
    
    if seq_index > 1:
        defect_parts = parts[1:seq_index]
        defect_code = '_'.join(defect_parts)
        sequence = parts[seq_index] if seq_index < len(parts) else '000'
    else:
        defect_code = parts[1]
        sequence = parts[2] if len(parts) > 2 else '000'
    
    if defect_code == 'ok':
        return None
    
    return {
        'product_code': product_code,
        'defect_code': defect_code,
        'sequence': sequence
    }


def parse_normal_filename(filename: str) -> Dict[str, str]:
    """정상 이미지 파일명 파싱"""
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


def get_product_id(db, product_code: str) -> int:
    """제품 코드로 product_id 조회"""
    product = db.query(Product).filter(
        Product.product_code == product_code
    ).first()
    
    return product.product_id if product else None


def get_defect_type_id(db, product_id: int, defect_code: str) -> int:
    """불량 코드로 defect_type_id 조회"""
    # 1차: 정확한 매칭
    defect_type = db.query(DefectType).filter(
        DefectType.product_id == product_id,
        DefectType.defect_code == defect_code
    ).first()
    
    if defect_type:
        return defect_type.defect_type_id
    
    # 2차: 첫 단어로 매칭
    if '_' in defect_code:
        first_word = defect_code.split('_')[0]
        defect_type = db.query(DefectType).filter(
            DefectType.product_id == product_id,
            DefectType.defect_code == first_word
        ).first()
        
        if defect_type:
            return defect_type.defect_type_id
    
    return None


def insert_image(db, image_data: Dict) -> bool:
    """이미지 DB 삽입"""
    try:
        image = Image(
            product_id=image_data['product_id'],
            image_type=image_data['image_type'],
            defect_type_id=image_data.get('defect_type_id'),
            file_name=image_data['file_name'],
            file_path=image_data['file_path'],
            storage_url=image_data['storage_url']
        )
        
        db.add(image)
        db.commit()
        return True
        
    except Exception as e:
        print(f"❌ DB 삽입 실패: {image_data['file_name']} - {e}")
        db.rollback()
        return False


def scan_defect_images() -> List[Dict]:
    """불량 이미지 스캔"""
    defect_dir = Path('/home/dmillion/llm_chal_vlm/data/def_split')
    
    if not defect_dir.exists():
        print(f"⚠️  디렉토리 없음: {defect_dir}")
        return []
    
    images = []
    skipped = 0
    
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        for img_path in defect_dir.glob(ext):
            filename = img_path.name
            parsed = parse_defect_filename(filename)
            
            if parsed is None:
                skipped += 1
                continue
            
            storage_url = f"{NCP_STORAGE_BASE_URL}/{NCP_BUCKET}/def_split/{filename}"
            
            images.append({
                'file_name': filename,
                'file_path': str(img_path),
                'storage_url': storage_url,
                'product_code': parsed['product_code'],
                'defect_code': parsed['defect_code'],
                'image_type': 'defect'
            })
    
    if skipped > 0:
        print(f"ℹ️  정상 이미지 {skipped}개 건너뜀")
    
    return images


def scan_normal_images() -> List[Dict]:
    """정상 이미지 스캔"""
    base_dir = Path('/home/dmillion/llm_chal_vlm/data/patchCore')
    
    if not base_dir.exists():
        print(f"⚠️  디렉토리 없음: {base_dir}")
        return []
    
    images = []
    
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
    
    db = get_db_session()
    skipped_products = set()
    
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
        defect_skip = 0
        
        for img in defect_images:
            product_id = get_product_id(db, img['product_code'])
            if not product_id:
                skipped_products.add(img['product_code'])
                defect_skip += 1
                continue
            
            defect_type_id = get_defect_type_id(db, product_id, img['defect_code'])
            
            image_data = {
                'product_id': product_id,
                'image_type': img['image_type'],
                'defect_type_id': defect_type_id,
                'file_name': img['file_name'],
                'file_path': img['file_path'],
                'storage_url': img['storage_url']
            }
            
            if insert_image(db, image_data):
                defect_success += 1
            else:
                defect_fail += 1
        
        print(f"✅ 불량 이미지 삽입 완료: {defect_success}개 성공, {defect_fail}개 실패, {defect_skip}개 스킵")
        
        # 4. 정상 이미지 DB 삽입
        print("\n[4/4] 정상 이미지 DB 삽입 중...")
        normal_success = 0
        normal_fail = 0
        normal_skip = 0
        
        for img in normal_images:
            product_id = get_product_id(db, img['product_code'])
            if not product_id:
                skipped_products.add(img['product_code'])
                normal_skip += 1
                continue
            
            image_data = {
                'product_id': product_id,
                'image_type': img['image_type'],
                'defect_type_id': None,
                'file_name': img['file_name'],
                'file_path': img['file_path'],
                'storage_url': img['storage_url']
            }
            
            if insert_image(db, image_data):
                normal_success += 1
            else:
                normal_fail += 1
        
        print(f"✅ 정상 이미지 삽입 완료: {normal_success}개 성공, {normal_fail}개 실패, {normal_skip}개 스킵")
        
        # 최종 결과
        print("\n" + "=" * 60)
        print("동기화 완료")
        print("=" * 60)
        print(f"불량 이미지: {defect_success}/{len(defect_images)} (스킵: {defect_skip})")
        print(f"정상 이미지: {normal_success}/{len(normal_images)} (스킵: {normal_skip})")
        print(f"총 성공: {defect_success + normal_success}")
        print(f"총 실패: {defect_fail + normal_fail}")
        print(f"총 스킵: {defect_skip + normal_skip}")
        if skipped_products:
            print(f"스킵된 제품: {', '.join(sorted(skipped_products))}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        db.close()


if __name__ == "__main__":
    sync_images_to_db()
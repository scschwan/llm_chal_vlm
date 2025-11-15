"""
기존 Object Storage 이미지 → DB 일괄 등록
실행: python sync_storage_to_db.py
"""

import os
import sys
from pathlib import Path
from typing import List, Dict
import pymysql

# 프로젝트 루트 경로 추가
sys.path.append('/home/dmillion/llm_chal_vlm')

# 환경변수 로드
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER", "dmillion")
DB_PASSWORD = os.getenv("DB_PASSWORD", "dm250120@")
DB_NAME = os.getenv("DB_NAME", "defect_detection_db")

NCP_STORAGE_BASE_URL = os.getenv('NCP_STORAGE_BASE_URL', 'https://kr.object.ncloudstorage.com')
NCP_BUCKET = os.getenv('NCP_BUCKET', 'dm-obs')


def get_db_connection():
    """DB 연결 생성 (pymysql 직접 사용)"""
    return pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )


def parse_defect_filename(filename: str) -> Dict[str, str]:
    """
    불량 이미지 파일명 파싱
    
    패턴:
    - {product}_{defect}_xxx.png
    - {product}_{defect1}_{defect2}_xxx.png (metal_contamination 등)
    
    마지막 숫자 부분을 찾아서 그 앞까지를 defect_code로 간주
    """
    stem = Path(filename).stem
    parts = stem.split('_')
    
    if len(parts) < 2:
        return {
            'product_code': parts[0] if parts else 'unknown',
            'defect_code': 'unknown',
            'sequence': '000'
        }
    
    product_code = parts[0]
    
    # ✅ 마지막부터 역순으로 숫자 패턴 찾기
    # 예: grid_metal_contamination_008_r180 → sequence는 008 또는 r180
    # 첫 번째 숫자 패턴(또는 회전 패턴)을 찾을 때까지 역순 탐색
    
    seq_index = -1
    for i in range(len(parts) - 1, 0, -1):
        # 숫자로 시작하거나 r로 시작(회전)하면 sequence 부분
        if parts[i].isdigit() or parts[i].startswith('r') or parts[i].startswith('fh'):
            seq_index = i
        else:
            # 숫자 아닌 부분을 만나면 중단
            break
    
    # defect_code는 product 다음부터 sequence 앞까지
    if seq_index > 1:
        defect_parts = parts[1:seq_index]
        defect_code = '_'.join(defect_parts)
        sequence = parts[seq_index] if seq_index < len(parts) else '000'
    else:
        # 패턴을 못 찾으면 기본 로직
        defect_code = parts[1]
        sequence = parts[2] if len(parts) > 2 else '000'
    
    # ✅ 'ok'는 정상 이미지 → None 반환
    if defect_code == 'ok':
        return None
    
    return {
        'product_code': product_code,
        'defect_code': defect_code,
        'sequence': sequence
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


def get_product_id(cursor, product_code: str) -> int:
    """제품 코드로 product_id 조회"""
    query = "SELECT product_id FROM products WHERE product_code = %s"
    cursor.execute(query, (product_code,))
    result = cursor.fetchone()
    
    if result:
        return result['product_id']
    else:
        print(f"⚠️  제품을 찾을 수 없음: {product_code}")
        return None


def get_defect_type_id(db, product_id: int, defect_code: str) -> int:
    """
    불량 코드로 defect_type_id 조회
    
    매칭 전략:
    1. 정확한 매칭 시도 (metal_contamination)
    2. 실패 시 첫 단어로 매칭 시도 (metal)
    """
    from web.database.models import DefectType
    
    # ✅ 1차: 정확한 매칭
    defect_type = db.query(DefectType).filter(
        DefectType.product_id == product_id,
        DefectType.defect_code == defect_code
    ).first()
    
    if defect_type:
        return defect_type.defect_type_id
    
    # ✅ 2차: 첫 단어로 매칭 (metal_contamination → metal)
    if '_' in defect_code:
        first_word = defect_code.split('_')[0]
        defect_type = db.query(DefectType).filter(
            DefectType.product_id == product_id,
            DefectType.defect_code == first_word
        ).first()
        
        if defect_type:
            print(f"ℹ️  매칭: {defect_code} → {first_word}")
            return defect_type.defect_type_id
    
    print(f"⚠️  불량 유형을 찾을 수 없음: product_id={product_id}, defect_code={defect_code}")
    return None


def check_duplicate(cursor, file_name: str) -> bool:
    """중복 체크"""
    query = "SELECT COUNT(*) as cnt FROM images WHERE file_name = %s"
    cursor.execute(query, (file_name,))
    result = cursor.fetchone()
    return result['cnt'] > 0


def insert_image(cursor, conn, image_data: Dict) -> bool:
    """이미지 DB 삽입"""
    try:
        # 중복 체크
        if check_duplicate(cursor, image_data['file_name']):
            print(f"⏭️  이미 존재: {image_data['file_name']}")
            return False
        
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
            if parsed != None :
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
    cursor = conn.cursor()
    
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
        defect_skip = 0
        defect_fail = 0
        
        for img in defect_images:
            product_id = get_product_id(cursor, img['product_code'])
            if not product_id:
                defect_fail += 1
                continue
            
            defect_type_id = get_defect_type_id(cursor, product_id, img['defect_code'])
            
            image_data = {
                'product_id': product_id,
                'image_type': img['image_type'],
                'defect_type_id': defect_type_id,
                'file_name': img['file_name'],
                'file_path': img['file_path'],
                'storage_url': img['storage_url']
            }
            
            result = insert_image(cursor, conn, image_data)
            if result:
                defect_success += 1
            elif result is False:
                defect_skip += 1
            else:
                defect_fail += 1
        
        print(f"✅ 불량 이미지: {defect_success}개 삽입, {defect_skip}개 스킵, {defect_fail}개 실패")
        
        # 4. 정상 이미지 DB 삽입
        print("\n[4/4] 정상 이미지 DB 삽입 중...")
        normal_success = 0
        normal_skip = 0
        normal_fail = 0
        
        for img in normal_images:
            product_id = get_product_id(cursor, img['product_code'])
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
            
            result = insert_image(cursor, conn, image_data)
            if result:
                normal_success += 1
            elif result is False:
                normal_skip += 1
            else:
                normal_fail += 1
        
        print(f"✅ 정상 이미지: {normal_success}개 삽입, {normal_skip}개 스킵, {normal_fail}개 실패")
        
        # 최종 결과
        print("\n" + "=" * 60)
        print("동기화 완료")
        print("=" * 60)
        print(f"불량 이미지: {defect_success}개 삽입")
        print(f"정상 이미지: {normal_success}개 삽입")
        print(f"총 삽입: {defect_success + normal_success}개")
        print(f"총 스킵: {defect_skip + normal_skip}개")
        print(f"총 실패: {defect_fail + normal_fail}개")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    sync_images_to_db()
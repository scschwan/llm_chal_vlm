"""
이미지 관리 API 라우터
"""

from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Form
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, List
from pathlib import Path
import zipfile
import shutil
import tempfile
import boto3
import os

import sys
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from web.database.connection import get_db
from web.database import crud
from web.utils.object_storage import get_obs_manager

router = APIRouter(prefix="/api/admin/image", tags=["admin-image"])


# ========================================
# Response 모델
# ========================================

class ImageResponse(BaseModel):
    """이미지 응답"""
    image_id: int
    product_id: int
    product_code: Optional[str] = None
    product_name: Optional[str] = None
    image_type: str
    defect_type_id: Optional[int]
    defect_code: Optional[str] = None
    defect_name: Optional[str] = None
    file_name: str
    file_path: str
    file_size: Optional[int]
    uploaded_at: str
    display_url: Optional[str] = None
    
    class Config:
        from_attributes = True


class ImageStatsResponse(BaseModel):
    """이미지 통계 응답"""
    total_images: int
    normal_images: int
    defect_images: int
    by_product: dict


# ========================================
# 헬퍼 함수
# ========================================

def is_image_file(filename: str) -> bool:
    """이미지 파일 체크"""
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    return any(filename.lower().endswith(ext) for ext in valid_extensions)


def extract_zip(zip_path: Path, extract_to: Path) -> List[Path]:
    """ZIP 파일 압축 해제"""
    image_files = []
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    # 이미지 파일만 수집
    for file_path in extract_to.rglob('*'):
        if file_path.is_file() and is_image_file(file_path.name):
            image_files.append(file_path)
    
    return image_files


# ========================================
# API 엔드포인트 - 정상 이미지
# ========================================

@router.post("/normal")
async def upload_normal_images(
    product_id: int = Form(...),
    files: List[UploadFile] = File(...)
):
    """
    정상 이미지 업로드 (다중 파일 또는 ZIP)
    """
    db = next(get_db())
    
    try:
        # 제품 존재 확인
        product = crud.get_product(db, product_id)
        if not product:
            raise HTTPException(404, "제품을 찾을 수 없습니다")
        
        obs = get_obs_manager()
        uploaded_files = []
        failed_files = []
        
        # 임시 디렉토리
        temp_dir = Path(tempfile.mkdtemp())
        
        for file in files:
            try:
                # ZIP 파일 처리
                if file.filename.lower().endswith('.zip'):
                    zip_path = temp_dir / file.filename
                    
                    # ZIP 파일 저장
                    with zip_path.open('wb') as buffer:
                        shutil.copyfileobj(file.file, buffer)
                    
                    # 압축 해제
                    extract_dir = temp_dir / f"extracted_{file.filename}"
                    extract_dir.mkdir(exist_ok=True)
                    image_files = extract_zip(zip_path, extract_dir)
                    
                    # 각 이미지 업로드
                    for img_path in image_files:
                        s3_key = f"images/normal/{product_id}/{img_path.name}"
                        
                        # Object Storage 업로드
                        success = obs.upload_file(str(img_path), s3_key)
                        if success:
                            # DB 저장
                            file_size = img_path.stat().st_size
                            image = crud.create_image(
                                db,
                                product_id=product_id,
                                image_type='normal',
                                file_name=img_path.name,
                                file_path=s3_key,
                                file_size=file_size
                            )
                            uploaded_files.append(img_path.name)
                        else:
                            failed_files.append(img_path.name)
                
                # 단일 이미지 파일 처리
                elif is_image_file(file.filename):
                    s3_key = f"images/normal/{product_id}/{file.filename}"
                    
                    # Object Storage 업로드
                    file.file.seek(0)
                    success = obs.upload_fileobj(file.file, s3_key)
                    
                    if success:
                        # 파일 크기 조회
                        file_info = obs.get_file_info(s3_key)
                        file_size = file_info['size'] if file_info else None
                        
                        # DB 저장
                        image = crud.create_image(
                            db,
                            product_id=product_id,
                            image_type='normal',
                            file_name=file.filename,
                            file_path=s3_key,
                            file_size=file_size
                        )
                        uploaded_files.append(file.filename)
                    else:
                        failed_files.append(file.filename)
                
                else:
                    failed_files.append(file.filename)
            
            except Exception as e:
                print(f"파일 업로드 실패 {file.filename}: {e}")
                failed_files.append(file.filename)
        
        # 임시 디렉토리 삭제
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        return JSONResponse(content={
            "status": "success" if uploaded_files else "failed",
            "message": f"{len(uploaded_files)}개 파일 업로드 완료",
            "uploaded_count": len(uploaded_files),
            "failed_count": len(failed_files),
            "uploaded_files": uploaded_files,
            "failed_files": failed_files
        })
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"이미지 업로드 실패: {str(e)}")
    finally:
        db.close()


@router.get("/normal", response_model=List[ImageResponse])
def list_normal_images(product_id: Optional[int] = None, skip: int = 0, limit: int = 100,
                       db: Session = Depends(get_db)):
    """
    정상 이미지 목록 조회
    """
    try:
        obs = get_obs_manager()
        
        if product_id:
            images = crud.get_images_by_product(db, product_id, image_type='normal')
        else:
            # 전체 정상 이미지
            images = db.query(crud.Image).filter(crud.Image.image_type == 'normal').offset(skip).limit(limit).all()
        
        result = []
        for img in images:
            product = crud.get_product(db, img.product_id)
            result.append(ImageResponse(
                image_id=img.image_id,
                product_id=img.product_id,
                product_code=product.product_code if product else None,
                product_name=product.product_name if product else None,
                image_type=img.image_type,
                defect_type_id=img.defect_type_id,
                defect_code=None,
                defect_name=None,
                file_name=img.file_name,
                file_path=img.file_path,
                file_size=img.file_size,
                uploaded_at=str(img.uploaded_at),
                display_url=obs.get_url(img.file_path)
            ))
        
        return result
    
    except Exception as e:
        raise HTTPException(500, f"이미지 목록 조회 실패: {str(e)}")


# ========================================
# API 엔드포인트 - 불량 이미지
# ========================================

@router.post("/defect")
async def upload_defect_images(
    product_id: int = Form(...),
    defect_type_id: int = Form(...),
    files: List[UploadFile] = File(...)
):
    """
    불량 이미지 업로드 (다중 파일 또는 ZIP)
    """
    db = next(get_db())
    
    try:
        # 제품 존재 확인
        product = crud.get_product(db, product_id)
        if not product:
            raise HTTPException(404, "제품을 찾을 수 없습니다")
        
        # 불량 유형 확인
        defect_type = crud.get_defect_type(db, defect_type_id)
        if not defect_type:
            raise HTTPException(404, "불량 유형을 찾을 수 없습니다")
        
        obs = get_obs_manager()
        uploaded_files = []
        failed_files = []
        
        # 임시 디렉토리
        temp_dir = Path(tempfile.mkdtemp())
        
        # 파일명 규칙: {product_code}_{defect_code}_{seq}
        # 현재 최대 시퀀스 번호 확인
        existing_images = crud.get_images_by_defect_type(db, defect_type_id)
        max_seq = 0
        for img in existing_images:
            try:
                parts = Path(img.file_name).stem.split('_')
                if len(parts) >= 3:
                    seq = int(parts[-1])
                    max_seq = max(max_seq, seq)
            except:
                continue
        
        current_seq = max_seq + 1
        
        for file in files:
            try:
                # ZIP 파일 처리
                if file.filename.lower().endswith('.zip'):
                    zip_path = temp_dir / file.filename
                    
                    # ZIP 파일 저장
                    with zip_path.open('wb') as buffer:
                        shutil.copyfileobj(file.file, buffer)
                    
                    # 압축 해제
                    extract_dir = temp_dir / f"extracted_{file.filename}"
                    extract_dir.mkdir(exist_ok=True)
                    image_files = extract_zip(zip_path, extract_dir)
                    
                    # 각 이미지 업로드
                    for img_path in image_files:
                        # 새 파일명 생성
                        ext = img_path.suffix
                        new_filename = f"{product.product_code}_{defect_type.defect_code}_{current_seq:03d}{ext}"
                        s3_key = f"images/defect/{product_id}_{defect_type_id}/{new_filename}"
                        
                        # Object Storage 업로드
                        success = obs.upload_file(str(img_path), s3_key)
                        if success:
                            # DB 저장
                            file_size = img_path.stat().st_size
                            image = crud.create_image(
                                db,
                                product_id=product_id,
                                image_type='defect',
                                defect_type_id=defect_type_id,
                                file_name=new_filename,
                                file_path=s3_key,
                                file_size=file_size
                            )
                            uploaded_files.append(new_filename)
                            current_seq += 1
                        else:
                            failed_files.append(img_path.name)
                
                # 단일 이미지 파일 처리
                elif is_image_file(file.filename):
                    # 새 파일명 생성
                    ext = Path(file.filename).suffix
                    new_filename = f"{product.product_code}_{defect_type.defect_code}_{current_seq:03d}{ext}"
                    s3_key = f"images/defect/{product_id}_{defect_type_id}/{new_filename}"
                    
                    # Object Storage 업로드
                    file.file.seek(0)
                    success = obs.upload_fileobj(file.file, s3_key)
                    
                    if success:
                        # 파일 크기 조회
                        file_info = obs.get_file_info(s3_key)
                        file_size = file_info['size'] if file_info else None
                        
                        # DB 저장
                        image = crud.create_image(
                            db,
                            product_id=product_id,
                            image_type='defect',
                            defect_type_id=defect_type_id,
                            file_name=new_filename,
                            file_path=s3_key,
                            file_size=file_size
                        )
                        uploaded_files.append(new_filename)
                        current_seq += 1
                    else:
                        failed_files.append(file.filename)
                
                else:
                    failed_files.append(file.filename)
            
            except Exception as e:
                print(f"파일 업로드 실패 {file.filename}: {e}")
                failed_files.append(file.filename)
        
        # 임시 디렉토리 삭제
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        return JSONResponse(content={
            "status": "success" if uploaded_files else "failed",
            "message": f"{len(uploaded_files)}개 파일 업로드 완료",
            "uploaded_count": len(uploaded_files),
            "failed_count": len(failed_files),
            "uploaded_files": uploaded_files,
            "failed_files": failed_files
        })
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"이미지 업로드 실패: {str(e)}")
    finally:
        db.close()


@router.get("/defect", response_model=List[ImageResponse])
def list_defect_images(product_id: Optional[int] = None, defect_type_id: Optional[int] = None,
                       skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """
    불량 이미지 목록 조회
    """
    try:
        obs = get_obs_manager()
        
        if defect_type_id:
            images = crud.get_images_by_defect_type(db, defect_type_id)
        elif product_id:
            images = crud.get_images_by_product(db, product_id, image_type='defect')
        else:
            # 전체 불량 이미지
            images = db.query(crud.Image).filter(crud.Image.image_type == 'defect').offset(skip).limit(limit).all()
        
        result = []
        for img in images:
            product = crud.get_product(db, img.product_id)
            defect_type = crud.get_defect_type(db, img.defect_type_id) if img.defect_type_id else None
            
            result.append(ImageResponse(
                image_id=img.image_id,
                product_id=img.product_id,
                product_code=product.product_code if product else None,
                product_name=product.product_name if product else None,
                image_type=img.image_type,
                defect_type_id=img.defect_type_id,
                defect_code=defect_type.defect_code if defect_type else None,
                defect_name=defect_type.defect_name_ko if defect_type else None,
                file_name=img.file_name,
                file_path=img.file_path,
                file_size=img.file_size,
                uploaded_at=str(img.uploaded_at),
                display_url=obs.get_url(img.file_path)
            ))
        
        return result
    
    except Exception as e:
        raise HTTPException(500, f"이미지 목록 조회 실패: {str(e)}")


# ========================================
# 공통 엔드포인트
# ========================================

@router.delete("/{image_id}")
def delete_image(image_id: int, db: Session = Depends(get_db)):
    """
    이미지 삭제 (Object Storage + DB)
    """
    image = crud.get_image(db, image_id)
    if not image:
        raise HTTPException(404, "이미지를 찾을 수 없습니다")
    
    try:
        obs = get_obs_manager()
        
        # Object Storage에서 삭제
        obs.delete_file(image.file_path)
        
        # DB에서 삭제
        success = crud.delete_image(db, image_id)
        if success:
            return JSONResponse(content={"status": "success", "message": "이미지가 삭제되었습니다"})
        else:
            raise HTTPException(500, "이미지 삭제 실패")
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"이미지 삭제 실패: {str(e)}")


@router.get("/stats", response_model=ImageStatsResponse)
def get_image_stats(db: Session = Depends(get_db)):
    """
    이미지 통계 조회
    """
    try:
        # 전체 이미지
        total_images = db.query(crud.Image).count()
        
        # 정상 이미지
        normal_images = db.query(crud.Image).filter(crud.Image.image_type == 'normal').count()
        
        # 불량 이미지
        defect_images = db.query(crud.Image).filter(crud.Image.image_type == 'defect').count()
        
        # 제품별 통계
        products = crud.get_products(db)
        by_product = {}
        
        for product in products:
            normal_count = db.query(crud.Image).filter(
                crud.Image.product_id == product.product_id,
                crud.Image.image_type == 'normal'
            ).count()
            
            defect_count = db.query(crud.Image).filter(
                crud.Image.product_id == product.product_id,
                crud.Image.image_type == 'defect'
            ).count()
            
            by_product[product.product_code] = {
                "product_name": product.product_name,
                "normal": normal_count,
                "defect": defect_count,
                "total": normal_count + defect_count
            }
        
        return ImageStatsResponse(
            total_images=total_images,
            normal_images=normal_images,
            defect_images=defect_images,
            by_product=by_product
        )
    
    except Exception as e:
        raise HTTPException(500, f"통계 조회 실패: {str(e)}")
    

@router.post("/sync-normal")
async def sync_normal_images():
    """
    정상(normal) 이미지 Object Storage에서 다운로드
    
    1. DB에서 sync_yn=0, use_yn=1, image_type='normal' 조회
    2. Object Storage에서 파일 다운로드
    3. DEFECT_PATH에 원본 파일명으로 저장
    4. NORMAL_PATH/{prod_code}/ok/ 에도 저장
    5. DB에서 sync_yn=1로 업데이트
    """
    import boto3
    import os
    from pathlib import Path
    from sqlalchemy import text
    import shutil
    
    # 프로젝트 루트 경로
    project_root = Path(__file__).parent.parent.parent
    
    # Object Storage 설정
    endpoint_url = os.getenv('NCP_STORAGE_BASE_URL', 'https://kr.object.ncloudstorage.com')
    access_key = os.getenv('NCP_ACCESS_KEY', '')
    secret_key = os.getenv('NCP_SECRET_KEY', '')
    bucket_name = os.getenv('NCP_BUCKET', 'dm-obs')
    region_name = 'kr-standard'
    
    # 저장 경로
    defect_path = Path(os.getenv('DEFECT_PATH', '/home/dmillion/llm_chal_vlm/data/def_split'))
    normal_path = Path(os.getenv('NORMAL_PATH', '/home/dmillion/llm_chal_vlm/data/patchCore'))
    
    defect_path.mkdir(parents=True, exist_ok=True)
    normal_path.mkdir(parents=True, exist_ok=True)
    
    # boto3 S3 클라이언트 생성
    s3_client = boto3.client(
        's3',
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region_name
    )
    
    # DB 연결
    db = next(get_db())
    
    try:
        print("[SYNC-NORMAL] 정상 이미지 동기화 시작")
        
        # 1. 동기화 대상 조회
        query = text("""
            SELECT
                im.image_id,
                (SELECT p.product_code FROM products p WHERE p.product_id = im.product_id) AS prod_code,
                (SELECT dt.defect_code FROM defect_types dt WHERE dt.defect_type_id = im.defect_type_id) AS defect_code,
                im.image_type,
                im.file_name,
                im.storage_url
            FROM images im
            WHERE im.sync_yn = 0
              AND im.use_yn = 1
              AND im.image_type = 'normal'
        """)
        
        result = db.execute(query)
        normal_images = result.fetchall()
        
        if not normal_images:
            print("[SYNC-NORMAL] 동기화할 정상 이미지가 없습니다.")
            return {
                "status": "success",
                "message": "동기화할 정상 이미지가 없습니다.",
                "synced_count": 0
            }
        
        print(f"[SYNC-NORMAL] {len(normal_images)}개 정상 이미지 동기화 시작")
        
        synced_ids = []
        failed_count = 0
        
        # 2. Object Storage에서 다운로드
        for img in normal_images:
            image_id, prod_code, defect_code, image_type, file_name, storage_url = img
            
            # storage_url에서 object_key 추출
            if storage_url.startswith('http'):
                object_key = '/'.join(storage_url.split('/')[-2:])  # images/defect/xxx.jpg
            else:
                object_key = storage_url.lstrip('/')
            
            print(f"storage_url : {storage_url} -> object_key : {object_key} ")
            
            # 1) DEFECT_PATH에 저장
            defect_local_path = defect_path / file_name
            
            # 2) NORMAL_PATH/{prod_code}/ok/ 에 저장
            product_dir = normal_path / prod_code
            ok_dir = product_dir / "ok"
            ok_dir.mkdir(parents=True, exist_ok=True)
            normal_local_path = ok_dir / file_name
            
            print(f"[SYNC-NORMAL] 다운로드 중: {object_key}")
            
            try:
                # DEFECT_PATH에 다운로드
                s3_client.download_file(
                    Bucket=bucket_name,
                    Key=object_key,
                    Filename=str(defect_local_path)
                )
                
                print(f"[SYNC-NORMAL] DEFECT_PATH 저장: {defect_local_path}")
                
                # NORMAL_PATH/{prod_code}/ok/에 복사
                shutil.copy2(str(defect_local_path), str(normal_local_path))
                
                print(f"[SYNC-NORMAL] NORMAL_PATH 저장: {normal_local_path}")
                print(f"[SYNC-NORMAL] 다운로드 완료: {file_name}")
                
                synced_ids.append(image_id)
                
            except Exception as e:
                print(f"[SYNC-NORMAL] 다운로드 실패 ({file_name}): {str(e)}")
                import traceback
                traceback.print_exc()
                failed_count += 1
                continue
        
        # 3. DB 업데이트 - sync_yn = 1
        if synced_ids:
            update_query = text("""
                UPDATE images 
                SET sync_yn = 1 
                WHERE image_id IN :image_ids
            """)
            db.execute(update_query, {"image_ids": tuple(synced_ids)})
            db.commit()
            
            print(f"[SYNC-NORMAL] DB 업데이트 완료: {len(synced_ids)}개")
        
        return {
            "status": "success",
            "message": "정상 이미지 동기화 완료",
            "synced_count": len(synced_ids),
            "failed_count": failed_count,
            "total": len(normal_images)
        }
        
    except Exception as e:
        print(f"[SYNC-NORMAL] 동기화 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        db.rollback()
        
        return {
            "status": "error",
            "message": f"동기화 실패: {str(e)}"
        }
        
    finally:
        db.close()

@router.post("/sync-defect")
async def sync_defect_images():
    """
    불량(defect) 이미지 Object Storage에서 다운로드
    
    1. DB에서 sync_yn=0, use_yn=1, image_type='defect' 조회
    2. Object Storage에서 파일 다운로드
    3. DEFECT_PATH에 원본 파일명으로 저장
    4. DB에서 sync_yn=1로 업데이트
    """
    
    
    from sqlalchemy import text
    
    # 프로젝트 루트 경로
    project_root = Path(__file__).parent.parent.parent
    
    # Object Storage 설정
    endpoint_url = os.getenv('NCP_STORAGE_BASE_URL', 'https://kr.object.ncloudstorage.com')
    access_key = os.getenv('NCP_ACCESS_KEY', '')
    secret_key = os.getenv('NCP_SECRET_KEY', '')
    bucket_name = os.getenv('NCP_BUCKET', 'dm-obs')
    region_name = 'kr-standard'
    
    # 저장 경로
    defect_path = Path(os.getenv('DEFECT_PATH', '/home/dmillion/llm_chal_vlm/data/def_split'))
    defect_path.mkdir(parents=True, exist_ok=True)
    
    # boto3 S3 클라이언트 생성
    s3_client = boto3.client(
        's3',
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region_name
    )
    
    # DB 연결
    db = next(get_db())
    
    try:
        print("[SYNC-DEFECT] 불량 이미지 동기화 시작")
        
        # 1. 동기화 대상 조회
        query = text("""
            SELECT
                im.image_id,
                (SELECT p.product_code FROM products p WHERE p.product_id = im.product_id) AS prod_code,
                (SELECT dt.defect_code FROM defect_types dt WHERE dt.defect_type_id = im.defect_type_id) AS defect_code,
                im.image_type,
                im.file_name,
                im.storage_url
            FROM images im
            WHERE im.sync_yn = 0
              AND im.use_yn = 1
              AND im.image_type = 'defect'
        """)
        
        result = db.execute(query)
        defect_images = result.fetchall()
        
        if not defect_images:
            print("[SYNC-DEFECT] 동기화할 불량 이미지가 없습니다.")
            return {
                "status": "success",
                "message": "동기화할 불량 이미지가 없습니다.",
                "synced_count": 0
            }
        
        print(f"[SYNC-DEFECT] {len(defect_images)}개 불량 이미지 동기화 시작")
        
        synced_ids = []
        failed_count = 0
        
        # 2. Object Storage에서 다운로드
        for img in defect_images:
            image_id, prod_code, defect_code, image_type, file_name, storage_url = img
            
            # 저장 경로 (원본 파일명 그대로)
            local_path = defect_path / file_name
            
            # storage_url에서 object_key 추출
            # storage_url 형식: https://kr.object.ncloudstorage.com/dm-obs/images/defect/xxx.jpg
            # 또는 images/defect/xxx.jpg
            
            
            if storage_url.startswith('http'):
                object_key = '/'.join(storage_url.split('/')[-2:])  # images/defect/xxx.jpg
            else:
                object_key = storage_url.lstrip('/')
            print(f"storage_url : {storage_url} -> object_key : {object_key} ")
            #object_key = storage_url
            print(f"[SYNC-DEFECT] 다운로드 중: {object_key} -> {local_path}")
            
            try:
                # boto3로 파일 다운로드
                s3_client.download_file(
                    Bucket=bucket_name,
                    Key=object_key,
                    Filename=str(local_path)
                )
                
                print(f"[SYNC-DEFECT] 다운로드 완료: {file_name}")
                synced_ids.append(image_id)
                
            except Exception as e:
                print(f"[SYNC-DEFECT] 다운로드 실패 ({file_name}): {str(e)}")
                failed_count += 1
                continue
        
        # 3. DB 업데이트 - sync_yn = 1
        if synced_ids:
            update_query = text("""
                UPDATE images 
                SET sync_yn = 1 
                WHERE image_id IN :image_ids
            """)
            db.execute(update_query, {"image_ids": tuple(synced_ids)})
            db.commit()
            
            print(f"[SYNC-DEFECT] DB 업데이트 완료: {len(synced_ids)}개")
        
        return {
            "status": "success",
            "message": "불량 이미지 동기화 완료",
            "synced_count": len(synced_ids),
            "failed_count": failed_count,
            "total": len(defect_images)
        }
        
    except Exception as e:
        print(f"[SYNC-DEFECT] 동기화 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        db.rollback()
        
        return {
            "status": "error",
            "message": f"동기화 실패: {str(e)}"
        }
        
    finally:
        db.close()
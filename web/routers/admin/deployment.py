"""
관리자 페이지 - 서버 배포 API
CLIP 임베딩 재구축 및 PatchCore 메모리뱅크 생성
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List, Dict
import asyncio
import os
import sys
from datetime import datetime
import traceback
import subprocess
import re
from pathlib import Path

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from modules.similarity_matcher import TopKSimilarityMatcher
from web.utils.object_storage import ObjectStorageManager

router = APIRouter(prefix="/api/admin/deployment", tags=["admin-deployment"])

# 전역 변수 - 진행 상태 추적
deployment_tasks = {}


class DeploymentResponse(BaseModel):
    task_id: str
    status: str
    message: str


class DeploymentStatus(BaseModel):
    task_id: str
    status: str  # pending, running, success, failed
    progress: int  # 0-100
    message: str
    start_time: Optional[str]
    end_time: Optional[str]
    error: Optional[str]


# ========================================
# 배포 로그 함수 (기존 스키마 사용)
# ========================================

def create_deployment_log_record(
    deploy_type: str,
    product_id: int = None,
    status: str = 'pending'
) -> int:
    """배포 로그 생성"""
    from web.database.connection import get_db
    from sqlalchemy import text
    from datetime import datetime
    
    db = next(get_db())
    
    try:
        query = text("""
            INSERT INTO deployment_logs 
            (deploy_type, product_id, status, started_at, deployed_by)
            VALUES (:deploy_type, :product_id, :status, :started_at, :deployed_by)
        """)
        
        now = datetime.now()
        db.execute(query, {
            "deploy_type": deploy_type,
            "product_id": product_id,
            "status": status,
            "started_at": now,
            "deployed_by": 'admin'
        })
        db.commit()
        
        # last insert id 조회
        result = db.execute(text("SELECT LAST_INSERT_ID() as id"))
        log_id = result.first()[0]
        
        return log_id
        
    except Exception as e:
        db.rollback()
        print(f"배포 로그 생성 오류: {e}")
        raise e
    finally:
        db.close()


def update_deployment_log_status(
    deploy_id: int,
    status: str,
    result_data: dict = None,
    error_message: str = None
):
    """배포 상태 업데이트"""
    from web.database.connection import get_db
    from sqlalchemy import text
    from datetime import datetime
    import json
    
    db = next(get_db())
    
    try:
        query = text("""
            UPDATE deployment_logs
            SET status = :status,
                completed_at = :completed_at,
                result_data = :result_data,
                result_message = :result_message
            WHERE deploy_id = :deploy_id
        """)
        
        now = datetime.now()
        result_json = json.dumps(result_data, ensure_ascii=False) if result_data else None
        
        db.execute(query, {
            "status": status,
            "completed_at": now,
            "result_data": result_json,
            "result_message": error_message,
            "deploy_id": deploy_id
        })
        db.commit()
        
    except Exception as e:
        db.rollback()
        print(f"배포 상태 업데이트 오류: {e}")
        raise e
    finally:
        db.close()

def get_deployment_log_list(
    limit: int = 20,
    deploy_type: str = None
) -> list:
    """배포 이력 조회"""
    from web.database.connection import get_db
    from sqlalchemy import text
    
    db = next(get_db())
    
    try:
        if deploy_type:
            query = text("""
                SELECT *
                FROM deployment_logs
                WHERE deploy_type = :deploy_type
                ORDER BY started_at DESC
                LIMIT :limit
            """)
            result = db.execute(query, {"deploy_type": deploy_type, "limit": limit})
        else:
            query = text("""
                SELECT *
                FROM deployment_logs
                ORDER BY started_at DESC
                LIMIT :limit
            """)
            result = db.execute(query, {"limit": limit})
        
        logs = []
        for row in result:
            log = dict(row._mapping)
            
            # 날짜 형식 변환
            if log.get('started_at'):
                log['start_time'] = log['started_at'].isoformat() if log['started_at'] else None
            if log.get('completed_at'):
                log['end_time'] = log['completed_at'].isoformat() if log['completed_at'] else None
            
            # API 호환성을 위한 필드명 매핑
            log['target'] = str(log.get('product_id', 'all'))
            log['deployment_type'] = log.get('deploy_type')
            log['error_message'] = log.get('result_message')
            
            logs.append(log)
        
        return logs
        
    except Exception as e:
        print(f"배포 이력 조회 오류: {e}")
        import traceback
        traceback.print_exc()
        raise e
    finally:
        db.close()


# ========================================
# CLIP 재구축 함수들
# ========================================

async def download_images_from_storage(
    storage_manager: ObjectStorageManager,
    prefix: str,
    local_dir: str,
    task_id: str
) -> List[str]:
    """Object Storage에서 이미지를 로컬로 다운로드"""
    try:
        # 디렉토리 생성
        os.makedirs(local_dir, exist_ok=True)
        
        # Object Storage 파일 목록 조회
        objects = storage_manager.list_objects(prefix)
        image_files = [obj['Key'] for obj in objects if obj['Key'].lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        total_files = len(image_files)
        downloaded_files = []
        
        # 진행 상태 업데이트
        deployment_tasks[task_id]['total'] = total_files
        deployment_tasks[task_id]['current'] = 0
        
        # 병렬 다운로드 (최대 10개 동시)
        semaphore = asyncio.Semaphore(10)
        
        async def download_file(key):
            async with semaphore:
                try:
                    filename = os.path.basename(key)
                    local_path = os.path.join(local_dir, filename)
                    
                    # 동기 함수를 비동기로 실행
                    await asyncio.get_event_loop().run_in_executor(
                        None,
                        storage_manager.download_file,
                        key,
                        local_path
                    )
                    
                    deployment_tasks[task_id]['current'] += 1
                    progress = int((deployment_tasks[task_id]['current'] / total_files) * 50)
                    deployment_tasks[task_id]['progress'] = progress
                    deployment_tasks[task_id]['message'] = f"다운로드 중... ({deployment_tasks[task_id]['current']}/{total_files})"
                    
                    return local_path
                except Exception as e:
                    print(f"Failed to download {key}: {e}")
                    return None
        
        # 모든 파일 다운로드
        tasks = [download_file(key) for key in image_files]
        results = await asyncio.gather(*tasks)
        downloaded_files = [path for path in results if path]
        
        return downloaded_files
        
    except Exception as e:
        raise Exception(f"이미지 다운로드 실패: {str(e)}")


async def build_clip_index(
    image_dir: str,
    index_type: str,
    task_id: str
) -> dict:
    """CLIP 임베딩 생성 및 FAISS 인덱스 저장"""
    try:
        deployment_tasks[task_id]['progress'] = 50
        deployment_tasks[task_id]['message'] = "CLIP 임베딩 생성 중..."
        
        # TopKSimilarityMatcher 초기화
        matcher = TopKSimilarityMatcher(
            model_id="ViT-B-32",
            pretrained="openai",
            device="auto",
            use_fp16=False,
            batch_size=32,
            num_workers=4,
            verbose=True
        )
        
        # 인덱스 구축
        deployment_tasks[task_id]['progress'] = 60
        deployment_tasks[task_id]['message'] = f"인덱스 구축 중..."
        
        build_info = await asyncio.get_event_loop().run_in_executor(
            None,
            matcher.build_index,
            image_dir
        )
        
        # 인덱스 저장
        deployment_tasks[task_id]['progress'] = 90
        deployment_tasks[task_id]['message'] = "인덱스 저장 중..."
        
        # 저장 경로 설정
        cache_dir = Path("/home/dmillion/llm_chal_vlm/web/index_cache")
        save_dir = cache_dir / index_type
        
        await asyncio.get_event_loop().run_in_executor(
            None,
            matcher.save_index,
            save_dir
        )
        
        deployment_tasks[task_id]['progress'] = 100
        deployment_tasks[task_id]['message'] = "완료"
        
        return {
            'total_images': build_info['num_images'],
            'index_type': index_type,
            'index_path': str(save_dir),
            'embedding_dim': build_info['embedding_dim']
        }
        
    except Exception as e:
        raise Exception(f"CLIP 인덱스 생성 실패: {str(e)}")


async def clip_rebuild_task(
    index_type: str,
    task_id: str,
    storage_prefix: str
):
    """CLIP 재구축 백그라운드 작업"""
    deploy_id = None
    try:
        deployment_tasks[task_id] = {
            'status': 'running',
            'progress': 0,
            'message': '시작',
            'start_time': datetime.now().isoformat(),
            'current': 0,
            'total': 0
        }
        
        # DB 로그 생성
        deploy_id = create_deployment_log_record(
            deploy_type='clip_rebuild',
            product_id=None,
            status='running'
        )
        deployment_tasks[task_id]['log_id'] = deploy_id
        
        # 1. Object Storage에서 다운로드
        storage_manager = ObjectStorageManager()
        local_dir = f"/home/dmillion/llm_chal_vlm/temp_deploy/{index_type}"
        
        downloaded_files = await download_images_from_storage(
            storage_manager,
            storage_prefix,
            local_dir,
            task_id
        )
        
        if not downloaded_files:
            raise Exception("다운로드된 이미지가 없습니다.")
        
        # 2. CLIP 인덱스 생성
        result = await build_clip_index(local_dir, index_type, task_id)
        
        # 3. 완료 처리
        deployment_tasks[task_id]['status'] = 'success'
        deployment_tasks[task_id]['end_time'] = datetime.now().isoformat()
        deployment_tasks[task_id]['result'] = result
        
        # DB 업데이트
        update_deployment_log_status(
            deploy_id,
            status='completed',
            result_data=result
        )
        
        # 임시 파일 정리
        import shutil
        shutil.rmtree(local_dir, ignore_errors=True)
        
    except Exception as e:
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        deployment_tasks[task_id]['status'] = 'failed'
        deployment_tasks[task_id]['error'] = error_msg
        deployment_tasks[task_id]['end_time'] = datetime.now().isoformat()
        
        if deploy_id:
            update_deployment_log_status(
                deploy_id,
                status='failed',
                error_message=error_msg
            )


# ========================================
# PatchCore 함수들
# ========================================

async def patchcore_build_task(task_id: str):
    """PatchCore 메모리뱅크 생성 백그라운드 작업"""
    deploy_id = None
    try:
        deployment_tasks[task_id] = {
            'status': 'running',
            'progress': 0,
            'message': '시작',
            'start_time': datetime.now().isoformat(),
            'logs': [],
            'products': {}
        }
        
        # 제품 목록 초기화
        products = ['prod1', 'prod2', 'prod3', 'leather', 'grid', 'carpet']
        for product in products:
            deployment_tasks[task_id]['products'][product] = 'pending'
        
        # DB 로그 생성
        deploy_id = create_deployment_log_record(
            deploy_type='patchcore_create',
            product_id=None,
            status='running'
        )
        deployment_tasks[task_id]['log_id'] = deploy_id
        
        # 스크립트 경로
        script_path = '/home/dmillion/llm_chal_vlm/build_patchcore.sh'
        
        # 스크립트 실행 권한 확인
        if not os.path.exists(script_path):
            raise Exception(f"스크립트를 찾을 수 없습니다: {script_path}")
        
        deployment_tasks[task_id]['message'] = 'PatchCore 메모리뱅크 생성 스크립트 실행 중...'
        deployment_tasks[task_id]['logs'].append('스크립트 실행 시작: ' + script_path)
        
        # 프로세스 실행
        process = await asyncio.create_subprocess_exec(
            'bash',
            script_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd='/home/dmillion/llm_chal_vlm'
        )
        
        # 실시간 로그 파싱
        total_products = len(products)
        completed_products = 0
        current_product = None
        
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            
            log_line = line.decode('utf-8').strip()
            if log_line:
                deployment_tasks[task_id]['logs'].append(log_line)
                
                # 제품별 진행 상태 파싱
                product_match = re.search(r'Building.*for\s+(\w+)', log_line, re.IGNORECASE)
                if product_match:
                    current_product = product_match.group(1)
                    if current_product in deployment_tasks[task_id]['products']:
                        deployment_tasks[task_id]['products'][current_product] = 'running'
                        deployment_tasks[task_id]['message'] = f'{current_product} 메모리뱅크 생성 중...'
                
                # 완료 메시지 파싱
                success_match = re.search(r'Successfully.*for\s+(\w+)', log_line, re.IGNORECASE)
                if success_match:
                    product = success_match.group(1)
                    if product in deployment_tasks[task_id]['products']:
                        deployment_tasks[task_id]['products'][product] = 'success'
                        completed_products += 1
                        progress = int((completed_products / total_products) * 100)
                        deployment_tasks[task_id]['progress'] = progress
                
                # 에러 메시지 파싱
                if 'error' in log_line.lower() or 'failed' in log_line.lower():
                    if current_product and current_product in deployment_tasks[task_id]['products']:
                        deployment_tasks[task_id]['products'][current_product] = 'failed'
        
        # 프로세스 완료 대기
        await process.wait()
        
        if process.returncode == 0:
            # 성공
            deployment_tasks[task_id]['status'] = 'success'
            deployment_tasks[task_id]['progress'] = 100
            deployment_tasks[task_id]['message'] = '모든 메모리뱅크 생성 완료'
            deployment_tasks[task_id]['end_time'] = datetime.now().isoformat()
            
            result_data = {
                'products': [p for p in products if deployment_tasks[task_id]['products'].get(p) == 'success'],
                'total': total_products,
                'success': sum(1 for s in deployment_tasks[task_id]['products'].values() if s == 'success'),
                'failed': sum(1 for s in deployment_tasks[task_id]['products'].values() if s == 'failed')
            }
            deployment_tasks[task_id]['result'] = result_data
            
            # DB 업데이트
            update_deployment_log_status(
                deploy_id,
                status='completed',
                result_data=result_data
            )
        else:
            # 실패
            raise Exception(f"스크립트 실행 실패 (exit code: {process.returncode})")
        
    except Exception as e:
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        deployment_tasks[task_id]['status'] = 'failed'
        deployment_tasks[task_id]['error'] = error_msg
        deployment_tasks[task_id]['end_time'] = datetime.now().isoformat()
        deployment_tasks[task_id]['logs'].append(f'ERROR: {error_msg}')
        
        if deploy_id:
            update_deployment_log_status(
                deploy_id,
                status='failed',
                error_message=error_msg
            )


# ========================================
# API 엔드포인트
# ========================================

@router.post("/clip/normal", response_model=DeploymentResponse)
async def rebuild_clip_normal(background_tasks: BackgroundTasks):
    """정상 이미지 CLIP 인덱스 재구축"""
    try:
        import uuid
        task_id = str(uuid.uuid4())
        
        # 백그라운드 작업 시작
        background_tasks.add_task(
            clip_rebuild_task,
            index_type='normal',
            task_id=task_id,
            storage_prefix='images/normal/'
        )
        
        return DeploymentResponse(
            task_id=task_id,
            status='started',
            message='정상 이미지 인덱스 재구축을 시작했습니다.'
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clip/defect", response_model=DeploymentResponse)
async def rebuild_clip_defect(background_tasks: BackgroundTasks):
    """불량 이미지 CLIP 인덱스 재구축"""
    try:
        import uuid
        task_id = str(uuid.uuid4())
        
        # 백그라운드 작업 시작
        background_tasks.add_task(
            clip_rebuild_task,
            index_type='defect',
            task_id=task_id,
            storage_prefix='images/defect/'
        )
        
        return DeploymentResponse(
            task_id=task_id,
            status='started',
            message='불량 이미지 인덱스 재구축을 시작했습니다.'
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{task_id}", response_model=DeploymentStatus)
async def get_deployment_status(task_id: str):
    """배포 진행 상태 조회"""
    if task_id not in deployment_tasks:
        raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다.")
    
    task = deployment_tasks[task_id]
    
    return DeploymentStatus(
        task_id=task_id,
        status=task.get('status', 'unknown'),
        progress=task.get('progress', 0),
        message=task.get('message', ''),
        start_time=task.get('start_time'),
        end_time=task.get('end_time'),
        error=task.get('error')
    )


@router.get("/logs")
async def get_deployment_history(
    limit: int = 20,
    deployment_type: str = None
):
    """배포 이력 조회"""
    try:
        logs = get_deployment_log_list(limit=limit, deploy_type=deployment_type)
        return {'logs': logs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/patchcore", response_model=DeploymentResponse)
async def build_patchcore_memory_bank(background_tasks: BackgroundTasks):
    """전체 PatchCore 메모리뱅크 생성"""
    try:
        import uuid
        task_id = str(uuid.uuid4())
        
        # 백그라운드 작업 시작
        background_tasks.add_task(
            patchcore_build_task,
            task_id=task_id
        )
        
        return DeploymentResponse(
            task_id=task_id,
            status='started',
            message='PatchCore 메모리뱅크 생성을 시작했습니다.'
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

# ========================================
# V2: DB 기반 인덱스 재구축
# ========================================

async def clip_rebuild_v2_task(
    index_type: str,
    task_id: str
):
    """V2 CLIP 재구축 백그라운드 작업 (DB 기반)"""
    deploy_id = None
    try:
        deployment_tasks[task_id] = {
            'status': 'running',
            'progress': 0,
            'message': '시작',
            'start_time': datetime.now().isoformat(),
        }
        
        # DB 로그 생성
        deploy_id = create_deployment_log_record(
            deploy_type=f'clip_v2_{index_type}',
            product_id=None,
            status='running'
        )
        deployment_tasks[task_id]['log_id'] = deploy_id
        
        # DB 세션 가져오기
        from web.database.connection import get_db
        from modules.similarity_matcher_v2 import create_matcher_v2
        
        db = next(get_db())
        
        deployment_tasks[task_id]['progress'] = 10
        deployment_tasks[task_id]['message'] = 'DB에서 메타데이터 조회 중...'
        
        # V2 매처 생성
        matcher = create_matcher_v2(
            model_id="ViT-B-32/openai",
            device="auto",
            use_fp16=False,
            batch_size=32,
            num_workers=4,
            verbose=True
        )
        
        deployment_tasks[task_id]['progress'] = 30
        deployment_tasks[task_id]['message'] = f'{index_type} 인덱스 구축 중...'
        
        # 인덱스 구축 (비동기 실행)
        info = await asyncio.get_event_loop().run_in_executor(
            None,
            matcher.build_index_from_db,
            db,
            index_type
        )
        
        deployment_tasks[task_id]['progress'] = 80
        deployment_tasks[task_id]['message'] = '인덱스 저장 중...'
        
        # 인덱스 저장
        save_dir = Path(f"/home/dmillion/llm_chal_vlm/web/index_cache_v2/{index_type}")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        await asyncio.get_event_loop().run_in_executor(
            None,
            matcher.save_index,
            str(save_dir)
        )
        
        deployment_tasks[task_id]['progress'] = 100
        deployment_tasks[task_id]['status'] = 'success'
        deployment_tasks[task_id]['message'] = '완료'
        deployment_tasks[task_id]['end_time'] = datetime.now().isoformat()
        
        result_data = {
            'total_images': info['num_images'],
            'index_type': index_type,
            'index_path': str(save_dir),
            'embedding_dim': info['embedding_dim'],
            'version': 'v2'
        }
        deployment_tasks[task_id]['result'] = result_data
        
        # DB 업데이트
        update_deployment_log_status(
            deploy_id,
            status='completed',
            result_data=result_data
        )
        
    except Exception as e:
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        deployment_tasks[task_id]['status'] = 'failed'
        deployment_tasks[task_id]['error'] = error_msg
        deployment_tasks[task_id]['end_time'] = datetime.now().isoformat()
        
        if deploy_id:
            update_deployment_log_status(
                deploy_id,
                status='failed',
                error_message=error_msg
            )


@router.post("/v2/clip/normal", response_model=DeploymentResponse)
async def rebuild_clip_v2_normal(background_tasks: BackgroundTasks):
    """V2 정상 이미지 CLIP 인덱스 재구축 (DB 기반)"""
    try:
        import uuid
        task_id = str(uuid.uuid4())
        
        # 백그라운드 작업 시작
        background_tasks.add_task(
            clip_rebuild_v2_task,
            index_type='normal',
            task_id=task_id
        )
        
        return DeploymentResponse(
            task_id=task_id,
            status='started',
            message='V2 정상 이미지 인덱스 재구축을 시작했습니다.'
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v2/clip/defect", response_model=DeploymentResponse)
async def rebuild_clip_v2_defect(background_tasks: BackgroundTasks):
    """V2 불량 이미지 CLIP 인덱스 재구축 (DB 기반)"""
    try:
        import uuid
        task_id = str(uuid.uuid4())
        
        # 백그라운드 작업 시작
        background_tasks.add_task(
            clip_rebuild_v2_task,
            index_type='defect',
            task_id=task_id
        )
        
        return DeploymentResponse(
            task_id=task_id,
            status='started',
            message='V2 불량 이미지 인덱스 재구축을 시작했습니다.'
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v2/clip/all", response_model=DeploymentResponse)
async def rebuild_clip_v2_all(background_tasks: BackgroundTasks):
    """V2 전체 CLIP 인덱스 재구축 (정상 + 불량)"""
    try:
        import uuid
        task_id = str(uuid.uuid4())
        
        # 정상 이미지 작업
        normal_task_id = str(uuid.uuid4())
        background_tasks.add_task(
            clip_rebuild_v2_task,
            index_type='normal',
            task_id=normal_task_id
        )
        
        # 불량 이미지 작업
        defect_task_id = str(uuid.uuid4())
        background_tasks.add_task(
            clip_rebuild_v2_task,
            index_type='defect',
            task_id=defect_task_id
        )
        
        # 메타 태스크 생성 (진행 상황 추적용)
        deployment_tasks[task_id] = {
            'status': 'running',
            'progress': 0,
            'message': '정상/불량 인덱스 재구축 시작',
            'start_time': datetime.now().isoformat(),
            'sub_tasks': {
                'normal': normal_task_id,
                'defect': defect_task_id
            }
        }
        
        return DeploymentResponse(
            task_id=task_id,
            status='started',
            message='V2 전체 인덱스 재구축을 시작했습니다. (정상 + 불량)'
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
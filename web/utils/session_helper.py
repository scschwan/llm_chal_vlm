"""
세션 관리 헬퍼 - OBS URL 기반 세션 디렉토리 관리
"""
import os
import shutil
from pathlib import Path
from urllib.parse import urlparse
from datetime import datetime
import uuid

# 기존 ObsManager 활용
from web.utils.object_storage import get_obs_manager


def generate_session_id() -> str:
    """유니크한 세션 ID 생성"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_id = uuid.uuid4().hex[:8]
    return f"{timestamp}_{unique_id}"


def create_obs_session_folder(session_id: str):
    """
    OBS에 세션 폴더 생성
    
    Args:
        session_id: 세션 ID
    """
    obs = get_obs_manager()
    folder_key = f"uploads/{session_id}/"
    obs.s3.put_object(Bucket=obs.bucket, Key=folder_key)


def upload_origin_to_obs(local_file_path: str, session_id: str) -> str:
    """
    원본 이미지를 OBS에 업로드 (origin.{확장자} 형식)
    
    Args:
        local_file_path: 로컬 파일 경로
        session_id: 세션 ID
        
    Returns:
        str: OBS URL
    """
    obs = get_obs_manager()
    
    # 파일 확장자 추출
    file_ext = Path(local_file_path).suffix
    
    # origin.{확장자} 형식으로 업로드
    origin_filename = f"origin{file_ext}"
    s3_key = f"uploads/{session_id}/{origin_filename}"
    
    if obs.upload_file(local_file_path, s3_key):
        return obs.get_url(s3_key)
    else:
        raise Exception(f"OBS 업로드 실패: {s3_key}")


def get_session_dir_from_url(obs_url: str, base_dir: str = "/home/dmillion/llm_chal_vlm/uploads") -> tuple[Path, str]:
    """
    OBS URL에서 세션 디렉토리 추출/생성 및 원본 이미지 다운로드
    
    Args:
        obs_url: https://kr.object.ncloudstorage.com/dm-obs/uploads/20250116_123456/origin.jpg
        base_dir: 로컬 베이스 디렉토리
        
    Returns:
        tuple: (세션 디렉토리 Path, 원본 이미지 경로)
    """
    obs = get_obs_manager()
    
    # URL 파싱하여 세션 ID 추출
    parsed = urlparse(obs_url)
    path_parts = [p for p in parsed.path.split('/') if p]
    
    # uploads/ 이후 첫 번째 폴더명 = 세션 ID
    try:
        # dm-obs/uploads/20250116_123456/origin.jpg
        uploads_idx = path_parts.index('uploads')
        session_id = path_parts[uploads_idx + 1]
        original_filename = path_parts[-1]  # origin.jpg
    except (ValueError, IndexError):
        raise ValueError(f"Invalid OBS URL format: {obs_url}")
    
    # 로컬 세션 디렉토리 생성
    session_dir = Path(base_dir) / session_id
    
    # 기존 폴더 있으면 삭제 후 재생성
    if session_dir.exists():
        shutil.rmtree(session_dir)
    
    session_dir.mkdir(parents=True, exist_ok=True)
    
    # 원본 파일 확장자 추출
    file_ext = Path(original_filename).suffix
    
    # origin.{확장자} 형식으로 로컬 저장
    origin_filename = f"origin{file_ext}"
    origin_path = session_dir / origin_filename
    
    # OBS에서 원본 파일 다운로드
    s3_key = f"uploads/{session_id}/{origin_filename}"
    
    if obs.download_file(s3_key, str(origin_path)):
        print(f"[SESSION] 원본 이미지 다운로드: {origin_path}")
    else:
        print(f"[SESSION] 원본 이미지 다운로드 실패: {s3_key}")
        # 첫 호출(유사이미지 검색)에서는 파일이 없을 수 있음
    
    return session_dir, str(origin_path)


def upload_session_file(local_file_path: Path, obs_base_url: str, custom_filename: str = None) -> str:
    """
    세션 폴더에 파일 업로드
    
    Args:
        local_file_path: 로컬 파일 경로
        obs_base_url: 기준 OBS URL (세션 ID 포함)
        custom_filename: 커스텀 파일명 (기본값: 원본 파일명)
        
    Returns:
        str: 업로드된 파일의 OBS URL
    """
    obs = get_obs_manager()
    
    # URL에서 세션 ID 추출
    parsed = urlparse(obs_base_url)
    path_parts = [p for p in parsed.path.split('/') if p]
    
    try:
        uploads_idx = path_parts.index('uploads')
        session_id = path_parts[uploads_idx + 1]
    except (ValueError, IndexError):
        raise ValueError(f"Invalid OBS URL format: {obs_base_url}")
    
    # 파일명 결정
    filename = custom_filename if custom_filename else local_file_path.name
    
    # OBS 경로
    s3_key = f"uploads/{session_id}/{filename}"
    
    # 업로드
    if obs.upload_file(str(local_file_path), s3_key):
        return obs.get_url(s3_key)
    else:
        raise Exception(f"파일 업로드 실패: {s3_key}")


def get_origin_image_path(session_dir: Path) -> Path:
    """
    세션 폴더에서 origin 이미지 경로 찾기
    
    Args:
        session_dir: 세션 디렉토리
        
    Returns:
        Path: origin.{확장자} 파일 경로
    """
    # origin.* 패턴으로 검색
    origin_files = list(session_dir.glob("origin.*"))
    
    if not origin_files:
        raise FileNotFoundError(f"원본 이미지를 찾을 수 없습니다: {session_dir}")
    
    return origin_files[0]
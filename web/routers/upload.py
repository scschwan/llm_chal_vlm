"""
이미지 업로드 관련 API 라우터
"""
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil
from typing import Optional

router = APIRouter(prefix="/upload", tags=["upload"])

# 업로드 디렉토리 (api_server.py에서 전달받음)
UPLOAD_DIR: Optional[Path] = None

def init_upload_router(upload_dir: Path):
    """라우터 초기화"""
    global UPLOAD_DIR
    UPLOAD_DIR = upload_dir
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/image")
async def upload_image(file: UploadFile = File(...)):
    """
    이미지 업로드
    
    Args:
        file: 업로드할 이미지 파일
    
    Returns:
        업로드된 파일 정보
    """
    if UPLOAD_DIR is None:
        raise HTTPException(500, "업로드 디렉토리가 초기화되지 않았습니다")
    
    # 파일 확장자 검증
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            400, 
            f"지원하지 않는 파일 형식입니다. 허용: {', '.join(allowed_extensions)}"
        )
    
    try:
        # 파일 저장
        file_path = UPLOAD_DIR / file.filename
        
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 파일 정보 수집
        file_size = file_path.stat().st_size
        
        return JSONResponse(content={
            "status": "success",
            "filename": file.filename,
            "file_path": str(file_path),
            "file_size": file_size,
            "file_size_mb": round(file_size / (1024 * 1024), 2),
            "message": "파일 업로드 완료"
        })
    
    except Exception as e:
        raise HTTPException(500, f"파일 업로드 실패: {str(e)}")


@router.get("/list")
async def list_uploaded_files():
    """업로드된 파일 목록 조회"""
    if UPLOAD_DIR is None:
        raise HTTPException(500, "업로드 디렉토리가 초기화되지 않았습니다")
    
    try:
        files = []
        for file_path in UPLOAD_DIR.iterdir():
            if file_path.is_file():
                files.append({
                    "filename": file_path.name,
                    "file_path": str(file_path),
                    "file_size": file_path.stat().st_size,
                    "modified_at": file_path.stat().st_mtime
                })
        
        return JSONResponse(content={
            "status": "success",
            "files": sorted(files, key=lambda x: x["modified_at"], reverse=True),
            "total_count": len(files)
        })
    
    except Exception as e:
        raise HTTPException(500, f"파일 목록 조회 실패: {str(e)}")


@router.delete("/clean")
async def clean_uploads():
    """업로드 디렉토리 정리"""
    if UPLOAD_DIR is None:
        raise HTTPException(500, "업로드 디렉토리가 초기화되지 않았습니다")
    
    try:
        deleted_count = 0
        for file_path in UPLOAD_DIR.glob("*"):
            if file_path.is_file():
                file_path.unlink()
                deleted_count += 1
        
        return JSONResponse(content={
            "status": "success",
            "deleted_count": deleted_count,
            "message": f"{deleted_count}개 파일 삭제 완료"
        })
    
    except Exception as e:
        raise HTTPException(500, f"파일 정리 실패: {str(e)}")


@router.delete("/file/{filename}")
async def delete_file(filename: str):
    """특정 파일 삭제"""
    if UPLOAD_DIR is None:
        raise HTTPException(500, "업로드 디렉토리가 초기화되지 않았습니다")
    
    try:
        file_path = UPLOAD_DIR / filename
        
        if not file_path.exists():
            raise HTTPException(404, f"파일을 찾을 수 없습니다: {filename}")
        
        file_path.unlink()
        
        return JSONResponse(content={
            "status": "success",
            "filename": filename,
            "message": "파일 삭제 완료"
        })
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"파일 삭제 실패: {str(e)}")
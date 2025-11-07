#!/usr/bin/env python3
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import socket
import datetime
from pathlib import Path

app = FastAPI(title="유사이미지 매칭 시스템")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 디렉토리 설정
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# ========== 헬스 체크 엔드포인트 ==========
@app.get("/health")
async def health_check():
    """ALB 헬스 체크용"""
    return JSONResponse({
        "status": "ok",
        "host": socket.gethostname(),
        "time": datetime.datetime.utcnow().isoformat()
    })

@app.get("/api/health")
async def api_health_check():
    """API 헬스 체크"""
    return JSONResponse({
        "status": "healthy",
        "message": "서버가 정상 작동 중입니다.",
        "host": socket.gethostname(),
        "time": datetime.datetime.utcnow().isoformat()
    })

# ========== HTML 페이지 제공 ==========
@app.get("/", response_class=HTMLResponse)
async def read_root():
    """메인 페이지"""
    html_file = BASE_DIR / "matching.html"
    if html_file.exists():
        content = html_file.read_text(encoding='utf-8')
        return HTMLResponse(content=content, status_code=200)
    return HTMLResponse(
        content="<h1>matching.html 파일을 찾을 수 없습니다.</h1>",
        status_code=404
    )

@app.get("/matching.html", response_class=HTMLResponse)
async def matching():
    """유사이미지 매칭 화면"""
    html_file = BASE_DIR / "matching.html"
    if html_file.exists():
        content = html_file.read_text(encoding='utf-8')
        return HTMLResponse(content=content, status_code=200)
    return HTMLResponse(
        content="<h1>matching.html 파일을 찾을 수 없습니다.</h1>",
        status_code=404
    )

@app.get("/manual_mapping.html", response_class=HTMLResponse)
async def manual_mapping():
    """데이터셋 매핑 화면"""
    html_file = BASE_DIR / "manual_mapping.html"
    if html_file.exists():
        content = html_file.read_text(encoding='utf-8')
        return HTMLResponse(content=content, status_code=200)
    return HTMLResponse(
        content="<h1>manual_mapping.html 파일을 찾을 수 없습니다.</h1>",
        status_code=404
    )

@app.get("/defect_analysis.html", response_class=HTMLResponse)
async def defect_analysis():
    """불량 분석 (RAG) 화면"""
    html_file = BASE_DIR / "defect_analysis.html"
    if html_file.exists():
        content = html_file.read_text(encoding='utf-8')
        return HTMLResponse(content=content, status_code=200)
    return HTMLResponse(
        content="<h1>defect_analysis.html 파일을 찾을 수 없습니다.</h1>",
        status_code=404
    )

@app.get("/dashboard.html", response_class=HTMLResponse)
async def dashboard():
    """통계 대시보드 화면"""
    html_file = BASE_DIR / "dashboard.html"
    if html_file.exists():
        content = html_file.read_text(encoding='utf-8')
        return HTMLResponse(content=content, status_code=200)
    return HTMLResponse(
        content="<h1>dashboard.html 파일을 찾을 수 없습니다.</h1>",
        status_code=404
    )

# ========== API 엔드포인트 ==========
@app.post("/api/upload-image")
async def upload_image(file: UploadFile = File(...)):
    """이미지 업로드 API"""
    try:
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        return JSONResponse({
            "status": "success",
            "message": "파일 업로드 성공",
            "filename": file.filename,
            "size": len(content),
            "path": str(file_path)
        })
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@app.post("/api/search-similar")
async def search_similar(top_k: int = Form(5)):
    """유사 이미지 검색 API (데모)"""
    results = []
    for i in range(1, top_k + 1):
        results.append({
            "rank": i,
            "similarity": round(95 - i * 5, 2),
            "image_path": f"/uploads/similar_{i}.jpg"
        })
    
    return JSONResponse({
        "status": "success",
        "results": results,
        "total": len(results)
    })

@app.get("/uploads/{filename}")
async def get_uploaded_file(filename: str):
    """업로드된 파일 제공"""
    file_path = UPLOAD_DIR / filename
    if file_path.exists():
        return FileResponse(file_path)
    return JSONResponse(
        {"status": "error", "message": "파일을 찾을 수 없습니다."},
        status_code=404
    )

# ========== 디버깅용 엔드포인트 ==========
@app.get("/debug/info")
async def debug_info():
    """서버 정보 확인용"""
    return JSONResponse({
        "hostname": socket.gethostname(),
        "base_dir": str(BASE_DIR),
        "upload_dir": str(UPLOAD_DIR),
        "time": datetime.datetime.now().isoformat(),
        "files": {
            "matching.html": (BASE_DIR / "matching.html").exists(),
            "manual_mapping.html": (BASE_DIR / "manual_mapping.html").exists(),
            "defect_analysis.html": (BASE_DIR / "defect_analysis.html").exists(),
            "dashboard.html": (BASE_DIR / "dashboard.html").exists(),
        }
    })

if __name__ == "__main__":
    print("=" * 70)
    print("🚀 유사이미지 매칭 시스템 서버 시작")
    print("=" * 70)
    print(f"📂 작업 디렉토리: {BASE_DIR}")
    print(f"📂 업로드 디렉토리: {UPLOAD_DIR}")
    print(f"🖥️  호스트명: {socket.gethostname()}")
    print("=" * 70)
    print("📍 접속 URL:")
    print("   - http://0.0.0.0:8080/")
    print("   - http://0.0.0.0:8080/matching.html")
    print("   - http://0.0.0.0:8080/manual_mapping.html")
    print("   - http://0.0.0.0:8080/defect_analysis.html")
    print("   - http://0.0.0.0:8080/dashboard.html")
    print("=" * 70)
    print("🔍 헬스 체크:")
    print("   - http://0.0.0.0:8080/health")
    print("   - http://0.0.0.0:8080/api/health")
    print("=" * 70)
    print("🐛 디버그 정보:")
    print("   - http://0.0.0.0:8080/debug/info")
    print("=" * 70)
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info"
    )

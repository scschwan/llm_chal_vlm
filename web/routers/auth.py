"""
인증 라우터
"""
from fastapi import APIRouter, HTTPException, Response, Cookie, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from web.utils.auth import verify_password, create_session, validate_session, delete_session

router = APIRouter(prefix="/api/auth", tags=["auth"])

# ========================================
# Request/Response 모델
# ========================================
class LoginRequest(BaseModel):
    """로그인 요청"""
    username: str
    password: str

class LoginResponse(BaseModel):
    """로그인 응답"""
    status: str
    message: str
    user_type: str
    full_name: str

class SessionResponse(BaseModel):
    """세션 정보 응답"""
    username: str
    user_type: str
    full_name: str

# ========================================
# API 엔드포인트
# ========================================
@router.post("/login")
async def login(request: LoginRequest, response: Response):
    """
    로그인
    """
    # 사용자 인증
    if not verify_password(request.username, request.password):
        raise HTTPException(401, "아이디 또는 비밀번호가 일치하지 않습니다")
    
    # 세션 생성
    session_token = create_session(request.username)
    
    # ✅ 쿠키 설정 강화
    response.set_cookie(
        key="session_token",
        value=session_token,
        httponly=True,
        max_age=28800,  # 8시간
        path="/",       # ✅ 모든 경로에서 쿠키 전달
        samesite="lax",
        secure=False    # ✅ HTTP 환경 (HTTPS에서는 True)
    )
    
    # 사용자 정보 반환
    from web.utils.auth import USERS
    user = USERS[request.username]
    
    print(f"[LOGIN] 세션 생성: {session_token[:8]}... for {request.username}")  # ✅ 디버깅
    
    return JSONResponse(content={
        "status": "success",
        "message": "로그인 성공",
        "user_type": user["user_type"],
        "full_name": user["full_name"],
        "is_new_login": True
    })


@router.post("/logout")
async def logout(response: Response, session_token: Optional[str] = Cookie(None)):
    """
    로그아웃
    """
    if session_token:
        delete_session(session_token)
    
    # 쿠키 삭제
    response.delete_cookie("session_token", path="/")  # ✅ path 추가
    
    return JSONResponse(content={
        "status": "success",
        "message": "로그아웃 되었습니다"
    })


@router.get("/session", response_model=SessionResponse)
async def get_session(session_token: Optional[str] = Cookie(None)):
    """
    현재 세션 정보 조회
    """
    if not session_token:
        raise HTTPException(401, "로그인이 필요합니다")
    
    session = validate_session(session_token)
    if not session:
        raise HTTPException(401, "세션이 만료되었습니다")
    
    return SessionResponse(
        username=session["username"],
        user_type=session["user_type"],
        full_name=session["full_name"]
    )


@router.get("/check")
async def check_auth(request: Request, session_token: Optional[str] = Cookie(None)):
    """
    인증 상태 확인
    """
    # ✅ 디버깅: 쿠키 전체 확인
    print(f"[AUTH CHECK] 요청 쿠키: {request.cookies}")
    print(f"[AUTH CHECK] session_token 파라미터: {session_token}")
    
    if not session_token:
        print("[AUTH CHECK] 세션 토큰 없음")
        return JSONResponse(content={
            "authenticated": False
        })
    
    session = validate_session(session_token)
    print(f"[AUTH CHECK] 세션 검증 결과: {session}")
    
    if not session:
        print("[AUTH CHECK] 세션 검증 실패 또는 만료")
        return JSONResponse(content={
            "authenticated": False
        })
    
    return JSONResponse(content={
        "authenticated": True,
        "user_type": session["user_type"],
        "username": session["username"],
        "full_name": session.get("full_name", "작업자")
    })
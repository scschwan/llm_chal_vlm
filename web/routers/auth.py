"""
인증 라우터 - 단순화 버전
"""
from fastapi import APIRouter, HTTPException, Response, Cookie
from pydantic import BaseModel
from typing import Optional

from web.utils.auth import (
    verify_password, 
    create_session, 
    validate_session, 
    delete_session,
    get_user_info
)

router = APIRouter(prefix="/api/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str
    password: str


@router.post("/login")
async def login(request: LoginRequest, response: Response):
    """로그인"""
    print(f"\n{'='*50}")
    print(f"[LOGIN] 요청: username={request.username}")
    
    # 1. 비밀번호 검증
    if not verify_password(request.username, request.password):
        print(f"[LOGIN] 실패: 잘못된 인증 정보")
        raise HTTPException(status_code=401, detail="아이디 또는 비밀번호가 일치하지 않습니다")
    
    # 2. 세션 생성
    session_token = create_session(request.username)
    
    # 3. 사용자 정보
    user_info = get_user_info(request.username)
    
    # 4. 쿠키 설정
    response.set_cookie(
        key="session_token",
        value=session_token,
        httponly=True,
        max_age=28800,
        path="/",
        samesite="lax"
    )
    
    print(f"[LOGIN] 성공: 쿠키 설정 완료")
    print(f"{'='*50}\n")
    
    return {
        "status": "success",
        "message": "로그인 성공",
        "user_type": user_info["user_type"],
        "full_name": user_info["full_name"]
    }


@router.get("/check")
async def check_auth(session_token: Optional[str] = Cookie(None)):
    """인증 확인"""
    print(f"\n[AUTH] 체크 요청: token={'있음' if session_token else '없음'}")
    
    if not session_token:
        print(f"[AUTH] 결과: 미인증 (토큰 없음)")
        return {"authenticated": False}
    
    session = validate_session(session_token)
    
    if not session:
        print(f"[AUTH] 결과: 미인증 (세션 무효)")
        return {"authenticated": False}
    
    print(f"[AUTH] 결과: 인증됨 (user={session['username']})")
    
    return {
        "authenticated": True,
        "user_type": session["user_type"],
        "username": session["username"],
        "full_name": session["full_name"]
    }


@router.post("/logout")
async def logout(response: Response, session_token: Optional[str] = Cookie(None)):
    """로그아웃"""
    if session_token:
        delete_session(session_token)
    
    response.delete_cookie("session_token", path="/")
    
    return {"status": "success", "message": "로그아웃 되었습니다"}
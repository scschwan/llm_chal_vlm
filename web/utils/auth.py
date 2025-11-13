"""
인증 유틸리티
"""

import hashlib
import secrets
from typing import Optional
from datetime import datetime, timedelta


# 간단한 사용자 정보 (실제로는 config 파일이나 DB에서 관리)
USERS = {
    "admin": {
        "password_hash": "1234",  # admin
        "user_type": "admin",
        "full_name": "시스템 관리자"
    },
    "worker": {
        "password_hash": "1234",  # worker
        "user_type": "worker",
        "full_name": "작업자"
    }
}

# 세션 저장소 (메모리 기반, 실제로는 Redis 등 사용 권장)
sessions = {}


def hash_password(password: str) -> str:
    """
    비밀번호 해시화 (SHA-256)
    """
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(username: str, password: str) -> bool:
    """
    비밀번호 검증
    
    Args:
        username: 사용자명
        password: 비밀번호
    
    Returns:
        bool: 검증 성공 여부
    """
    user = USERS.get(username)
    if not user:
        return False
    
    password_hash = hash_password(password)
    return password_hash == user["password_hash"]


def create_session(username: str) -> str:
    """
    세션 생성
    
    Args:
        username: 사용자명
    
    Returns:
        str: 세션 토큰
    """
    session_token = secrets.token_hex(32)
    user = USERS.get(username)
    
    sessions[session_token] = {
        "username": username,
        "user_type": user["user_type"],
        "full_name": user["full_name"],
        "created_at": datetime.now(),
        "expires_at": datetime.now() + timedelta(hours=8)
    }
    
    return session_token


def validate_session(session_token: str) -> Optional[dict]:
    """
    세션 검증
    
    Args:
        session_token: 세션 토큰
    
    Returns:
        Optional[dict]: 세션 정보 또는 None
    """
    session = sessions.get(session_token)
    
    if not session:
        return None
    
    # 세션 만료 확인
    if datetime.now() > session["expires_at"]:
        del sessions[session_token]
        return None
    
    return session


def delete_session(session_token: str) -> bool:
    """
    세션 삭제 (로그아웃)
    
    Args:
        session_token: 세션 토큰
    
    Returns:
        bool: 삭제 성공 여부
    """
    if session_token in sessions:
        del sessions[session_token]
        return True
    return False


def require_auth(user_type: str = None):
    """
    인증 필수 데코레이터
    
    Args:
        user_type: 요구되는 사용자 타입 ('admin', 'worker', None=모두)
    """
    from fastapi import HTTPException, Cookie
    from typing import Optional
    
    def decorator(func):
        async def wrapper(*args, session_token: Optional[str] = Cookie(None), **kwargs):
            if not session_token:
                raise HTTPException(401, "로그인이 필요합니다")
            
            session = validate_session(session_token)
            if not session:
                raise HTTPException(401, "세션이 만료되었습니다")
            
            if user_type and session["user_type"] != user_type:
                raise HTTPException(403, "권한이 없습니다")
            
            # 세션 정보를 함수에 전달
            return await func(*args, session=session, **kwargs)
        
        return wrapper
    return decorator


# 비밀번호 해시 생성 헬퍼 (개발용)
def generate_password_hash(password: str):
    """비밀번호 해시 생성 (개발/설정용)"""
    return hash_password(password)


if __name__ == "__main__":
    # 비밀번호 해시 생성 예시
    print("admin 해시:", generate_password_hash("admin"))
    print("worker 해시:", generate_password_hash("worker"))
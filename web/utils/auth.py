"""
인증 유틸리티
"""

import hashlib
import secrets
from typing import Optional
from datetime import datetime, timedelta


# 간단한 사용자 정보
USERS = {
    "admin": {
        "password": "admin",  # 실제 비밀번호 (개발용)
        "user_type": "admin",
        "full_name": "시스템 관리자"
    },
    "worker": {
        "password": "worker",  # 실제 비밀번호 (개발용)
        "user_type": "worker",
        "full_name": "작업자"
    }
}

# 세션 저장소
sessions = {}


def verify_password(username: str, password: str) -> bool:
    """
    비밀번호 검증
    """
    user = USERS.get(username)
    if not user:
        return False
    
    # 개발용: 평문 비교
    return password == user["password"]


def create_session(username: str) -> str:
    """세션 생성"""
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
    """세션 검증"""
    session = sessions.get(session_token)
    
    if not session:
        return None
    
    if datetime.now() > session["expires_at"]:
        del sessions[session_token]
        return None
    
    return session


def delete_session(session_token: str) -> bool:
    """세션 삭제"""
    if session_token in sessions:
        del sessions[session_token]
        return True
    return False
"""
인증 유틸리티 - 단순화 버전
"""
import secrets
from typing import Optional
from datetime import datetime, timedelta

# 사용자 정보
USERS = {
    "admin": {
        "password": "admin",
        "user_type": "admin",
        "full_name": "시스템 관리자"
    },
    "worker": {
        "password": "worker",
        "user_type": "worker",
        "full_name": "작업자"
    }
}

# 세션 저장소 (메모리)
_sessions = {}


def verify_password(username: str, password: str) -> bool:
    """비밀번호 검증"""
    user = USERS.get(username)
    if not user:
        return False
    return password == user["password"]


def create_session(username: str) -> str:
    """세션 생성"""
    session_token = secrets.token_hex(32)
    user = USERS[username]
    
    _sessions[session_token] = {
        "username": username,
        "user_type": user["user_type"],
        "full_name": user["full_name"],
        "created_at": datetime.now(),
        "expires_at": datetime.now() + timedelta(hours=8)
    }
    
    print(f"[SESSION] 생성: {session_token[:16]}... (user={username})")
    print(f"[SESSION] 현재 활성 세션 수: {len(_sessions)}")
    
    return session_token


def validate_session(session_token: str) -> Optional[dict]:
    """세션 검증"""
    if not session_token:
        return None
    
    session = _sessions.get(session_token)
    
    if not session:
        print(f"[SESSION] 검증 실패: 세션 없음 ({session_token[:16]}...)")
        return None
    
    if datetime.now() > session["expires_at"]:
        print(f"[SESSION] 검증 실패: 만료됨 ({session_token[:16]}...)")
        del _sessions[session_token]
        return None
    
    print(f"[SESSION] 검증 성공: {session['username']}")
    return session


def delete_session(session_token: str) -> bool:
    """세션 삭제"""
    if session_token in _sessions:
        username = _sessions[session_token].get("username", "unknown")
        del _sessions[session_token]
        print(f"[SESSION] 삭제: {username}")
        return True
    return False


def get_user_info(username: str) -> Optional[dict]:
    """사용자 정보 조회"""
    return USERS.get(username)
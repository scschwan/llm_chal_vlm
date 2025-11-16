// 공통 함수 및 설정

// API 기본 URL
const API_BASE_URL = window.location.origin;

/**
 * 파일 크기 포맷팅
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

/**
 * 날짜 포맷팅
 */
function formatDateTime(timestamp) {
    const date = new Date(timestamp * 1000);
    return date.toLocaleString('ko-KR');
}

/**
 * 알림 메시지 표시
 */
function showNotification(message, type = 'info') {
    // 간단한 알림 (나중에 toast 라이브러리로 교체 가능)
    const colors = {
        'success': '#10b981',
        'error': '#ef4444',
        'warning': '#f59e0b',
        'info': '#3b82f6'
    };
    
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 15px 20px;
        background: ${colors[type]};
        color: white;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        z-index: 9999;
        animation: slideIn 0.3s ease;
    `;
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// 애니메이션 CSS 추가
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from { transform: translateX(400px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    @keyframes slideOut {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(400px); opacity: 0; }
    }
`;
document.head.appendChild(style);

/**
 * 세션 스토리지 헬퍼
 */
const SessionData = {
    set(key, value) {
        sessionStorage.setItem(key, JSON.stringify(value));
    },
    get(key) {
        const data = sessionStorage.getItem(key);
        return data ? JSON.parse(data) : null;
    },
    remove(key) {
        sessionStorage.removeItem(key);
    },
    // ✅ 추가: 모든 세션 데이터 삭제
    clear: () => {
        // isNewLogin 플래그는 유지 (삭제는 upload.js에서 처리)
        const isNewLogin = sessionStorage.getItem('isNewLogin');
        sessionStorage.clear();
        if (isNewLogin) {
            sessionStorage.setItem('isNewLogin', isNewLogin);
        }
    },
    // ✅ 새 워크플로우 시작 (완전 초기화)
    startNewWorkflow() {
        console.log('[SESSION] 새 워크플로우 시작 - 전체 초기화');
        sessionStorage.clear();
    }
};

/**
 * 인덱스 상태 확인
 */
async function checkIndexStatus() {
    try {
        const response = await fetch(`${API_BASE_URL}/index/status`);
        const data = await response.json();
        
        const statusEl = document.getElementById('indexStatus');
        if (!statusEl) return;
        
        if (data.status === 'success') {
            const typeText = data.current_index_type === 'defect' ? '불량 이미지' : '정상 이미지';
            statusEl.innerHTML = `
                <p class="status-ok">
                    ✅ ${typeText} 인덱스 활성<br>
                    <small>${data.gallery_count}개 이미지</small>
                </p>
            `;
        } else {
            statusEl.innerHTML = `
                <p class="status-error">❌ 인덱스 미구축</p>
            `;
        }
        
        return data;
    } catch (err) {
        console.error('[INDEX] 상태 확인 실패:', err);
        const statusEl = document.getElementById('indexStatus');
        if (statusEl) {
            statusEl.innerHTML = `
                <p class="status-error">❌ 상태 확인 실패</p>
            `;
        }
        return null;
    }
}
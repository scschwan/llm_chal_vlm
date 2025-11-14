// CLIP 재구축 JavaScript

let currentTasks = {
    normal: null,
    defect: null
};

let pollIntervals = {
    normal: null,
    defect: null
};

// 페이지 로드 시 이력 조회
document.addEventListener('DOMContentLoaded', function() {
    loadDeploymentLogs();
    // 5초마다 이력 자동 갱신
    setInterval(loadDeploymentLogs, 5000);
});

/**
 * 재구축 시작
 */
async function startRebuild(indexType) {
    const btnId = indexType === 'normal' ? 'btnNormal' : 'btnDefect';
    const btn = document.getElementById(btnId);
    
    // 버튼 비활성화
    btn.disabled = true;
    btn.textContent = '진행 중...';
    
    // 진행 상태 영역 표시
    showProgress(indexType);
    
    try {
        const response = await fetch(`/api/admin/deployment/clip/${indexType}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        if (!response.ok) {
            throw new Error('재구축 시작 실패');
        }
        
        const data = await response.json();
        currentTasks[indexType] = data.task_id;
        
        showAlert('success', `${indexType === 'normal' ? '정상' : '불량'} 이미지 인덱스 재구축을 시작했습니다.`);
        
        // 진행 상태 폴링 시작
        startPolling(indexType, data.task_id);
        
    } catch (error) {
        console.error('Error:', error);
        showAlert('error', '재구축 시작 중 오류가 발생했습니다: ' + error.message);
        hideProgress(indexType);
        btn.disabled = false;
        btn.textContent = '재구축 시작';
    }
}

/**
 * 진행 상태 폴링
 */
function startPolling(indexType, taskId) {
    // 기존 폴링 중지
    if (pollIntervals[indexType]) {
        clearInterval(pollIntervals[indexType]);
    }
    
    // 1초마다 상태 조회
    pollIntervals[indexType] = setInterval(async () => {
        try {
            const response = await fetch(`/api/admin/deployment/status/${taskId}`);
            
            if (!response.ok) {
                throw new Error('상태 조회 실패');
            }
            
            const status = await response.json();
            updateProgress(indexType, status);
            
            // 완료 또는 실패 시 폴링 중지
            if (status.status === 'success' || status.status === 'failed') {
                clearInterval(pollIntervals[indexType]);
                pollIntervals[indexType] = null;
                
                const btnId = indexType === 'normal' ? 'btnNormal' : 'btnDefect';
                const btn = document.getElementById(btnId);
                btn.disabled = false;
                btn.textContent = '재구축 시작';
                
                // 이력 갱신
                loadDeploymentLogs();
                
                if (status.status === 'success') {
                    showAlert('success', `${indexType === 'normal' ? '정상' : '불량'} 이미지 인덱스 재구축이 완료되었습니다.`);
                } else {
                    showAlert('error', `재구축 실패: ${status.error || '알 수 없는 오류'}`);
                }
            }
            
        } catch (error) {
            console.error('Polling error:', error);
        }
    }, 1000);
}

/**
 * 진행 상태 표시
 */
function showProgress(indexType) {
    const progressId = indexType === 'normal' ? 'progressNormal' : 'progressDefect';
    const progressSection = document.getElementById(progressId);
    progressSection.classList.add('active');
    
    // 초기화
    updateProgress(indexType, {
        progress: 0,
        message: '시작 중...'
    });
}

/**
 * 진행 상태 숨기기
 */
function hideProgress(indexType) {
    const progressId = indexType === 'normal' ? 'progressNormal' : 'progressDefect';
    const progressSection = document.getElementById(progressId);
    progressSection.classList.remove('active');
}

/**
 * 진행 상태 업데이트
 */
function updateProgress(indexType, status) {
    const progressBarId = indexType === 'normal' ? 'progressBarNormal' : 'progressBarDefect';
    const messageId = indexType === 'normal' ? 'messageNormal' : 'messageDefect';
    
    const progressBar = document.getElementById(progressBarId);
    const message = document.getElementById(messageId);
    
    const progress = status.progress || 0;
    progressBar.style.width = progress + '%';
    progressBar.textContent = progress + '%';
    
    message.textContent = status.message || '';
}

/**
 * 알림 표시
 */
function showAlert(type, message) {
    const alertContainer = document.getElementById('alertContainer');
    const alertClass = type === 'success' ? 'alert-success' : 'alert-error';
    
    const alertHtml = `
        <div class="alert ${alertClass}">
            ${message}
        </div>
    `;
    
    alertContainer.innerHTML = alertHtml;
    
    // 5초 후 자동 제거
    setTimeout(() => {
        alertContainer.innerHTML = '';
    }, 5000);
}

/**
 * 배포 이력 조회
 */
async function loadDeploymentLogs() {
    try {
        const response = await fetch('/api/admin/deployment/logs?deployment_type=clip&limit=20');
        
        if (!response.ok) {
            throw new Error('이력 조회 실패');
        }
        
        const data = await response.json();
        displayLogs(data.logs);
        
    } catch (error) {
        console.error('Error loading logs:', error);
    }
}

/**
 * 이력 테이블 표시
 */
function displayLogs(logs) {
    const tbody = document.getElementById('logTableBody');
    
    if (!logs || logs.length === 0) {
        tbody.innerHTML = '<tr><td colspan="6" class="empty-message">이력이 없습니다.</td></tr>';
        return;
    }
    
    const rows = logs.map(log => {
        const startTime = new Date(log.start_time).toLocaleString('ko-KR');
        const statusBadge = getStatusBadge(log.status);
        const duration = calculateDuration(log.start_time, log.end_time);
        const result = formatResult(log.result_data);
        
        return `
            <tr>
                <td>${startTime}</td>
                <td>CLIP</td>
                <td>${log.target === 'normal' ? '정상' : '불량'}</td>
                <td>${statusBadge}</td>
                <td>${duration}</td>
                <td>${result}</td>
            </tr>
        `;
    }).join('');
    
    tbody.innerHTML = rows;
}

/**
 * 상태 뱃지 생성
 */
function getStatusBadge(status) {
    const badges = {
        'running': '<span class="status-badge status-running">진행중</span>',
        'success': '<span class="status-badge status-success">완료</span>',
        'failed': '<span class="status-badge status-failed">실패</span>'
    };
    return badges[status] || status;
}

/**
 * 소요 시간 계산
 */
function calculateDuration(startTime, endTime) {
    if (!endTime) return '-';
    
    const start = new Date(startTime);
    const end = new Date(endTime);
    const diff = Math.floor((end - start) / 1000); // 초 단위
    
    if (diff < 60) {
        return `${diff}초`;
    } else if (diff < 3600) {
        const minutes = Math.floor(diff / 60);
        const seconds = diff % 60;
        return `${minutes}분 ${seconds}초`;
    } else {
        const hours = Math.floor(diff / 3600);
        const minutes = Math.floor((diff % 3600) / 60);
        return `${hours}시간 ${minutes}분`;
    }
}

/**
 * 결과 데이터 포맷팅
 */
function formatResult(resultData) {
    if (!resultData) return '-';
    
    try {
        const data = typeof resultData === 'string' ? JSON.parse(resultData) : resultData;
        return `${data.total_images || 0}개 이미지`;
    } catch (e) {
        return '-';
    }
}
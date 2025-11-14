// PatchCore 메모리뱅크 생성 JavaScript

let currentTaskId = null;
let pollInterval = null;
let logLines = [];

const PRODUCTS = ['prod1', 'prod2', 'prod3', 'leather', 'grid', 'carpet'];

// 페이지 로드 시 초기화
document.addEventListener('DOMContentLoaded', function() {
    initProductGrid();
    loadDeploymentLogs();
    // 10초마다 이력 자동 갱신
    setInterval(loadDeploymentLogs, 10000);
});

/**
 * 제품 그리드 초기화
 */
function initProductGrid() {
    const grid = document.getElementById('productGrid');
    
    const items = PRODUCTS.map(product => `
        <div class="product-item pending" id="product-${product}">
            <div class="product-name">${product}</div>
            <div class="product-status">대기 중</div>
        </div>
    `).join('');
    
    grid.innerHTML = items;
}

/**
 * 메모리뱅크 생성 시작
 */
async function startBuild() {
    const btn = document.getElementById('btnBuild');
    
    // 버튼 비활성화
    btn.disabled = true;
    btn.textContent = '생성 중...';
    
    // 진행 상태 영역 표시
    showProgress();
    
    // 제품 그리드 초기화
    initProductGrid();
    logLines = [];
    updateLogOutput('메모리뱅크 생성을 시작합니다...\n');
    
    try {
        const response = await fetch('/api/admin/deployment/patchcore', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        if (!response.ok) {
            throw new Error('메모리뱅크 생성 시작 실패');
        }
        
        const data = await response.json();
        currentTaskId = data.task_id;
        
        showAlert('success', 'PatchCore 메모리뱅크 생성을 시작했습니다.');
        
        // 진행 상태 폴링 시작
        startPolling(data.task_id);
        
    } catch (error) {
        console.error('Error:', error);
        showAlert('error', '메모리뱅크 생성 시작 중 오류가 발생했습니다: ' + error.message);
        hideProgress();
        btn.disabled = false;
        btn.textContent = '전체 메모리뱅크 생성 시작';
    }
}

/**
 * 진행 상태 폴링
 */
function startPolling(taskId) {
    // 기존 폴링 중지
    if (pollInterval) {
        clearInterval(pollInterval);
    }
    
    // 2초마다 상태 조회
    pollInterval = setInterval(async () => {
        try {
            const response = await fetch(`/api/admin/deployment/status/${taskId}`);
            
            if (!response.ok) {
                throw new Error('상태 조회 실패');
            }
            
            const status = await response.json();
            updateProgress(status);
            
            // 완료 또는 실패 시 폴링 중지
            if (status.status === 'success' || status.status === 'failed') {
                clearInterval(pollInterval);
                pollInterval = null;
                
                const btn = document.getElementById('btnBuild');
                btn.disabled = false;
                btn.textContent = '전체 메모리뱅크 생성 시작';
                
                // 이력 갱신
                loadDeploymentLogs();
                
                if (status.status === 'success') {
                    showAlert('success', 'PatchCore 메모리뱅크 생성이 완료되었습니다.');
                    document.getElementById('overallStatus').textContent = '완료';
                    document.getElementById('overallStatus').className = 'status-badge status-success';
                } else {
                    showAlert('error', `생성 실패: ${status.error || '알 수 없는 오류'}`);
                    document.getElementById('overallStatus').textContent = '실패';
                    document.getElementById('overallStatus').className = 'status-badge status-failed';
                }
            }
            
        } catch (error) {
            console.error('Polling error:', error);
        }
    }, 2000);
}

/**
 * 진행 상태 표시
 */
function showProgress() {
    const progressSection = document.getElementById('progressSection');
    progressSection.classList.add('active');
    
    // 초기화
    document.getElementById('progressBar').style.width = '0%';
    document.getElementById('progressBar').textContent = '0%';
    document.getElementById('currentProduct').textContent = '시작 중...';
    document.getElementById('overallStatus').textContent = '진행 중';
    document.getElementById('overallStatus').className = 'status-badge status-running';
}

/**
 * 진행 상태 숨기기
 */
function hideProgress() {
    const progressSection = document.getElementById('progressSection');
    progressSection.classList.remove('active');
}

/**
 * 진행 상태 업데이트
 */
function updateProgress(status) {
    // 전체 진행률
    const progress = status.progress || 0;
    const progressBar = document.getElementById('progressBar');
    progressBar.style.width = progress + '%';
    progressBar.textContent = progress + '%';
    
    // 현재 작업 중인 제품
    const currentProduct = document.getElementById('currentProduct');
    currentProduct.textContent = status.message || '';
    
    // 로그 업데이트
    if (status.logs && Array.isArray(status.logs)) {
        status.logs.forEach(log => {
            if (!logLines.includes(log)) {
                logLines.push(log);
                updateLogOutput(log + '\n');
            }
        });
    }
    
    // 제품별 상태 업데이트
    if (status.products) {
        Object.keys(status.products).forEach(product => {
            updateProductStatus(product, status.products[product]);
        });
    }
}

/**
 * 제품 상태 업데이트
 */
function updateProductStatus(product, status) {
    const productElement = document.getElementById(`product-${product}`);
    if (!productElement) return;
    
    // 기존 클래스 제거
    productElement.classList.remove('pending', 'running', 'success', 'failed');
    
    // 새 클래스 추가
    productElement.classList.add(status);
    
    // 상태 텍스트 업데이트
    const statusText = {
        'pending': '대기 중',
        'running': '생성 중',
        'success': '완료',
        'failed': '실패'
    };
    
    const statusElement = productElement.querySelector('.product-status');
    statusElement.textContent = statusText[status] || status;
}

/**
 * 로그 출력 업데이트
 */
function updateLogOutput(text) {
    const logOutput = document.getElementById('logOutput');
    
    if (logLines.length === 0) {
        logOutput.textContent = text;
    } else {
        logOutput.textContent += text;
    }
    
    // 자동 스크롤
    logOutput.scrollTop = logOutput.scrollHeight;
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
        const response = await fetch('/api/admin/deployment/logs?deployment_type=patchcore&limit=20');
        
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
        const productCount = getProductCount(log.result_data);
        const message = log.error_message || '정상 완료';
        
        return `
            <tr>
                <td>${startTime}</td>
                <td>PatchCore</td>
                <td>${statusBadge}</td>
                <td>${duration}</td>
                <td>${productCount}</td>
                <td>${message}</td>
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
    const diff = Math.floor((end - start) / 1000);
    
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
 * 제품 수 추출
 */
function getProductCount(resultData) {
    if (!resultData) return '-';
    
    try {
        const data = typeof resultData === 'string' ? JSON.parse(resultData) : resultData;
        return `${data.products?.length || 0}개`;
    } catch (e) {
        return '-';
    }
}
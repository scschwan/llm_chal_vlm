/**
 * 관리자 대시보드 스크립트
 */

// 페이지 로드 시 초기화
document.addEventListener('DOMContentLoaded', async () => {
    await checkAuth();
    await loadDashboardData();
    initEventListeners();
});

/**
 * 인증 확인
 */
async function checkAuth() {
    try {
        const response = await fetch('/api/auth/session');
        
        if (!response.ok) {
            window.location.href = '/login.html';
            return;
        }
        
        const session = await response.json();
        
        if (session.user_type !== 'admin') {
            window.location.href = '/upload.html';
        }
    } catch (error) {
        console.error('인증 확인 실패:', error);
        window.location.href = '/login.html';
    }
}

/**
 * 대시보드 데이터 로드
 */
async function loadDashboardData() {
    try {
        // 통계 데이터 로드
        await loadStats();
        
        // 최근 검사 내역 로드
        await loadRecentInspections();
        
        // 제품별 통계 로드
        await loadProductStats();
        
    } catch (error) {
        console.error('대시보드 데이터 로드 실패:', error);
    }
}

/**
 * 통계 데이터 로드
 */
async function loadStats() {
    try {
        const response = await fetch('/api/admin/dashboard/stats');
        const result = await response.json();
        
        if (result.status === 'success') {
            const data = result.data;
            document.getElementById('totalProducts').textContent = data.totalProducts;
            document.getElementById('totalNormalImages').textContent = data.totalNormalImages.toLocaleString();
            document.getElementById('totalDefectImages').textContent = data.totalDefectImages.toLocaleString();
            document.getElementById('todayInspections').textContent = data.todayInspections;
        }
    } catch (error) {
        console.error('통계 로드 실패:', error);
        document.getElementById('totalProducts').textContent = 'Error';
        document.getElementById('totalNormalImages').textContent = 'Error';
        document.getElementById('totalDefectImages').textContent = 'Error';
        document.getElementById('todayInspections').textContent = 'Error';
    }
}

/**
 * 최근 검사 내역 로드
 */
async function loadRecentInspections() {
    const container = document.getElementById('recentInspections');
    
    try {
        const response = await fetch('/api/admin/dashboard/inspections/recent?limit=20');
        const result = await response.json();
        
        if (result.status === 'success') {
            const inspections = result.data;
            
            if (inspections.length === 0) {
                container.innerHTML = `
                    <div class="empty-state">
                        <div class="empty-state-icon">📭</div>
                        <div class="empty-state-text">검사 내역이 없습니다</div>
                    </div>
                `;
                return;
            }
            
            container.innerHTML = `
                <table>
                    <thead>
                        <tr>
                            <th>시간</th>
                            <th>제품</th>
                            <th>불량 유형</th>
                            <th>결과</th>
                            <th>이상 점수</th>
                            <th>작업자</th>
                            <th>작업</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${inspections.map(item => `
                            <tr>
                                <td>${item.timestamp}</td>
                                <td>${item.product}</td>
                                <td>${item.defect}</td>
                                <td>
                                    <span class="badge ${item.result === 'anomaly' ? 'badge-danger' : 'badge-success'}">
                                        ${item.result === 'anomaly' ? '⚠️ 이상' : '✅ 정상'}
                                    </span>
                                </td>
                                <td>${item.score.toFixed(4)}</td>
                                <td>${item.worker}</td>
                                <td>
                                    <button class="action-btn" onclick="viewDetail(${item.id})">상세</button>
                                </td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            `;
        }
    } catch (error) {
        console.error('검사 내역 로드 실패:', error);
        container.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon">❌</div>
                <div class="empty-state-text">데이터 로드에 실패했습니다</div>
            </div>
        `;
    }
}

/**
 * 제품별 통계 로드
 */
async function loadProductStats() {
    const container = document.getElementById('productStats');
    
    try {
        const response = await fetch('/api/admin/dashboard/products/stats');
        const result = await response.json();
        
        if (result.status === 'success') {
            const products = result.data;
            
            container.innerHTML = `
                <table>
                    <thead>
                        <tr>
                            <th>제품명</th>
                            <th>정상 이미지</th>
                            <th>불량 이미지</th>
                            <th>총 검사</th>
                            <th>불량 검출율</th>
                            <th>작업</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${products.map(item => `
                            <tr>
                                <td><strong>${item.name_ko || item.name}</strong></td>
                                <td>${item.normalImages.toLocaleString()}</td>
                                <td>${item.defectImages.toLocaleString()}</td>
                                <td>${item.totalInspections.toLocaleString()}</td>
                                <td>
                                    <span class="badge ${item.defectRate > 0.2 ? 'badge-warning' : 'badge-success'}">
                                        ${(item.defectRate * 100).toFixed(1)}%
                                    </span>
                                </td>
                                <td>
                                    <button class="action-btn" onclick="manageProduct('${item.name}')">관리</button>
                                </td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            `;
        }
    } catch (error) {
        console.error('제품 통계 로드 실패:', error);
        container.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon">❌</div>
                <div class="empty-state-text">데이터 로드에 실패했습니다</div>
            </div>
        `;
    }
}

/**
 * 이벤트 리스너 초기화
 */
function initEventListeners() {
    // 필터 버튼
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            this.parentElement.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            this.classList.add('active');
            
            const chartType = this.closest('.chart-card').querySelector('.chart-title').textContent;
            const period = this.dataset.period;
            
            console.log(`[DASHBOARD] 차트 필터 변경: ${chartType} - ${period}`);
            // TODO: 차트 데이터 갱신
        });
    });
    
    // 검색
    const searchInput = document.getElementById('searchInput');
    if (searchInput) {
        searchInput.addEventListener('input', function(e) {
            const searchTerm = e.target.value.toLowerCase();
            console.log(`[DASHBOARD] 검색: ${searchTerm}`);
            // TODO: 검색 필터링
        });
    }
}

/**
 * 상세 보기
 */
function viewDetail(id) {
    // TODO: 상세 페이지 구현
    alert(`검사 ID ${id}의 상세 정보를 표시합니다 (추후 구현)`);
}

/**
 * 제품 관리
 */
function manageProduct(productName) {
    window.location.href = `/admin/product.html?product=${productName}`;
}

/**
 * 로그아웃
 */
async function logout() {
    if (!confirm('로그아웃 하시겠습니까?')) return;
    
    try {
        await fetch('/api/auth/logout', { method: 'POST' });
        window.location.href = '/login.html';
    } catch (error) {
        console.error('로그아웃 실패:', error);
        alert('로그아웃에 실패했습니다');
    }
}
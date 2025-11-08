// web/static/js/matching.js

// API 서버 주소
const API_BASE_URL = '';

// 전역 변수
let selectedFile = null;
let searchResults = null;
let uploadedImagePath = null;
let defectConfig = null;

// DOM 요소
let uploadArea, fileInput, previewImage, searchButton, detectButton;
let topKSlider, topKValue, resultsContainer, anomalyResultsContainer;
let resultsStats, statusMessage, anomalyStatusMessage;
let checkIndexBtn, rebuildIndexBtn, indexStatus;

// 초기화
document.addEventListener('DOMContentLoaded', () => {
    initializeElements();
    initializeEventListeners();
    loadDefectConfig();
});

function initializeElements() {
    uploadArea = document.getElementById('uploadArea');
    fileInput = document.getElementById('fileInput');
    previewImage = document.getElementById('previewImage');
    searchButton = document.getElementById('searchButton');
    detectButton = document.getElementById('detectButton');
    topKSlider = document.getElementById('topKSlider');
    topKValue = document.getElementById('topKValue');
    resultsContainer = document.getElementById('resultsContainer');
    anomalyResultsContainer = document.getElementById('anomalyResultsContainer');
    resultsStats = document.getElementById('resultsStats');
    statusMessage = document.getElementById('statusMessage');
    anomalyStatusMessage = document.getElementById('anomalyStatusMessage');
    checkIndexBtn = document.getElementById('checkIndexBtn');
    rebuildIndexBtn = document.getElementById('rebuildIndexBtn');
    indexStatus = document.getElementById('indexStatus');
}

function initializeEventListeners() {
    // 탭 전환
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', () => switchTab(tab));
    });

    // 파일 업로드
    uploadArea.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', (e) => handleFileSelect(e.target.files[0]));
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);

    // 슬라이더
    topKSlider.addEventListener('input', (e) => {
        topKValue.textContent = e.target.value;
    });

    // 버튼
    searchButton.addEventListener('click', performSearch);
    detectButton.addEventListener('click', performAnomalyDetection);
    checkIndexBtn.addEventListener('click', checkIndexStatus);
    rebuildIndexBtn.addEventListener('click', rebuildIndex);
}

// 탭 전환
function switchTab(tab) {
    const targetTab = tab.dataset.tab;
    
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    document.getElementById(`${targetTab}-tab`).classList.add('active');
}

// 설정 파일 로드
async function loadDefectConfig() {
    try {
        const response = await fetch('/defect_config.json');
        defectConfig = await response.json();
    } catch (error) {
        console.error('설정 파일 로드 실패:', error);
        defectConfig = {
            products: {
                prod1: { name: "제품1", defects: ["hole", "burr", "scratch"] }
            }
        };
    }
}

// 파일 처리
function handleDragOver(e) {
    e.preventDefault();
    uploadArea.classList.add('dragover');
}

function handleDragLeave() {
    uploadArea.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    handleFileSelect(e.dataTransfer.files[0]);
}

function handleFileSelect(file) {
    if (!file || !file.type.startsWith('image/')) {
        showStatus('이미지 파일만 업로드 가능합니다.', 'error');
        return;
    }

    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewImage.style.display = 'block';
    };
    reader.readAsDataURL(file);

    showStatus('이미지가 업로드되었습니다. 검색 버튼을 클릭하세요.', 'success');
    
    document.getElementById('anomalyInputInfo').innerHTML = `✅ ${file.name}`;
}

// 검색 실행
async function performSearch() {
    if (!selectedFile) {
        showStatus('먼저 이미지를 업로드하세요.', 'error');
        return;
    }

    const topK = parseInt(topKSlider.value);
    
    searchButton.disabled = true;
    searchButton.innerHTML = '<span class="loading"></span> 검색 중...';
    showStatus('유사 이미지를 검색하는 중...', 'info');

    try {
        const formData = new FormData();
        formData.append('file', selectedFile);

        const response = await fetch(`${API_BASE_URL}/search/upload?top_k=${topK}`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || '검색 실패');
        }

        const data = await response.json();
        searchResults = data;
        uploadedImagePath = `./uploads/${selectedFile.name}`;
        
        displayResults(data);
        showStatus(`검색 완료! ${data.top_k_results.length}개의 유사 이미지를 찾았습니다.`, 'success');

        if (data.top_k_results.length > 0) {
            detectButton.disabled = false;
            document.getElementById('anomalyRefInfo').innerHTML = 
                `✅ ${data.top_k_results[0].image_name} (유사도: ${(data.top_k_results[0].similarity_score * 100).toFixed(1)}%)`;
        }

    } catch (error) {
        console.error('검색 오류:', error);
        showStatus(`검색 실패: ${error.message}`, 'error');
    } finally {
        searchButton.disabled = false;
        searchButton.innerHTML = '🔍 유사 이미지 검색';
    }
}

// 결과 표시
function displayResults(data) {
    const results = data.top_k_results;

    resultsStats.innerHTML = `
        검색된 이미지: <strong>${results.length}개</strong> | 
        전체 DB: <strong>${data.total_gallery_size}개</strong>
    `;

    let html = '<div class="results-grid">';
    results.forEach((result, index) => {
        const isMain = index === 0;
        const cardClass = isMain ? 'result-card main-result' : 'result-card';
        const similarity = (result.similarity_score * 100).toFixed(1);
        
        html += `
            <div class="${cardClass}" onclick="${isMain ? '' : `swapTopResult(${index})`}" 
                 style="${isMain ? '' : 'cursor: pointer;'}">
                <span class="rank-badge">${isMain ? '🏆 TOP 1' : `#${result.rank}`}</span>
                ${!isMain ? '<div class="swap-hint">클릭하여 TOP-1로 변경</div>' : ''}
                <img src="/api/image/${encodeURIComponent(result.image_path)}" 
                     alt="Result ${result.rank}"
                     onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%22300%22 height=%22200%22><rect fill=%22%23ddd%22 width=%22300%22 height=%22200%22/><text x=%2250%%22 y=%2250%%22 text-anchor=%22middle%22 fill=%22%23999%22>이미지 로드 실패</text></svg>'">
                <div class="result-info">
                    <div class="image-name" title="${result.image_name}">
                        📁 ${result.image_name}
                    </div>
                    <div class="similarity-score">
                        <div class="similarity-bar">
                            <div class="similarity-fill" style="width: ${similarity}%"></div>
                        </div>
                        <span class="similarity-value">${similarity}%</span>
                    </div>
                </div>
            </div>
        `;
    });
    html += '</div>';
    
    html += `
        <div style="margin-top: 30px; text-align: center;">
            <button onclick="openDefectRegistration()" 
                    style="padding: 15px 30px; background: #28a745; color: white; border: none; border-radius: 10px; font-size: 1.1em; font-weight: 600; cursor: pointer; transition: transform 0.2s;">
                ➕ 해당하는 불량이 없습니다 - 불량 이미지 등록
            </button>
        </div>
    `;
    
    resultsContainer.innerHTML = html;
}

// TOP-1 스왑
function swapTopResult(clickedIndex) {
    if (!searchResults || clickedIndex === 0) return;
    
    const results = searchResults.top_k_results;
    
    const temp = results[0];
    results[0] = results[clickedIndex];
    results[clickedIndex] = temp;
    
    results.forEach((r, idx) => {
        r.rank = idx + 1;
    });
    
    displayResults(searchResults);
    
    document.getElementById('anomalyRefInfo').innerHTML = 
        `✅ ${results[0].image_name} (유사도: ${(results[0].similarity_score * 100).toFixed(1)}%)`;
    
    showStatus(`TOP-1이 ${results[0].image_name}으로 변경되었습니다.`, 'success');
}

// 이상 검출
async function performAnomalyDetection() {
    if (!searchResults || searchResults.top_k_results.length === 0) {
        showAnomalyStatus('먼저 유사도 검색을 실행하세요.', 'error');
        return;
    }

    detectButton.disabled = true;
    detectButton.innerHTML = '<span class="loading"></span> 검출 중...';
    showAnomalyStatus('이상 영역을 검출하는 중...', 'info');

    try {
        const response = await fetch(`${API_BASE_URL}/detect_anomaly`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                test_image_path: uploadedImagePath,
                reference_image_path: null,  // 자동 검색
                product_name: null
            })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || '이상 검출 실패');
        }

        const data = await response.json();
        displayAnomalyResults(data);
        showAnomalyStatus('이상 검출 완료!', 'success');

    } catch (error) {
        console.error('이상 검출 오류:', error);
        showAnomalyStatus(`이상 검출 실패: ${error.message}`, 'error');
    } finally {
        detectButton.disabled = false;
        detectButton.innerHTML = '🎯 이상 영역 검출';
    }
}

// 이상 검출 결과 표시
function displayAnomalyResults(data) {
    const html = `
        <div class="anomaly-results">
            <div class="anomaly-card">
                <h3>📸 정상 기준 이미지</h3>
                <img src="${data.reference_normal_url}" alt="Normal Reference">
                <div class="anomaly-score">
                    <span>제품: <strong>${data.product_name}</strong></span>
                    <span class="anomaly-badge ${data.is_anomaly ? 'anomaly' : 'normal'}">
                        ${data.is_anomaly ? '⚠️ 이상 감지' : '✅ 정상'}
                    </span>
                </div>
            </div>

            <div class="anomaly-card">
                <h3>🎭 이상 영역 마스크</h3>
                <img src="${data.mask_url}" alt="Mask">
                <div class="anomaly-score">
                    <span>이상 점수</span>
                    <span><strong>${data.image_score.toFixed(4)}</strong></span>
                </div>
            </div>

            <div class="anomaly-card" style="grid-column: span 2;">
                <h3>📊 비교 결과</h3>
                <img src="${data.comparison_url}" alt="Comparison">
                <p style="font-size: 0.85em; color: #6c757d; margin-top: 10px;">
                    좌측: 정상 기준 이미지 | 우측: 이상 영역 표시 (빨간색)
                </p>
            </div>
        </div>
    `;
    
    anomalyResultsContainer.innerHTML = html;
}


// 모달 열 때 통계 표시 (선택사항)
async function openDefectRegistration() {
    if (!selectedFile) {
        alert('먼저 이미지를 업로드하세요.');
        return;
    }
    
    const filename = selectedFile.name;
    const autoProduct = filename.split('_')[0];
    
    let productOptions = '';
    for (const [key, value] of Object.entries(defectConfig.products)) {
        const selected = key === autoProduct ? 'selected' : '';
        productOptions += `<option value="${key}" ${selected}>${value.name} (${key})</option>`;
    }
    
    const modalHTML = `
        <div id="defectModal" style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.7); display: flex; justify-content: center; align-items: center; z-index: 9999;">
            <div style="background: white; padding: 30px; border-radius: 15px; max-width: 500px; width: 90%;">
                <h2 style="margin-bottom: 20px; color: #343a40;">불량 이미지 등록</h2>
                
                <div style="margin-bottom: 15px;">
                    <label style="display: block; margin-bottom: 8px; font-weight: 600;">제품명</label>
                    <select id="productSelectModal" onchange="updateDefectOptions()" style="width: 100%; padding: 10px; border-radius: 6px; border: 1px solid #dee2e6;">
                        ${productOptions}
                    </select>
                </div>
                
                <div style="margin-bottom: 15px;">
                    <label style="display: block; margin-bottom: 8px; font-weight: 600;">불량명</label>
                    <select id="defectSelectModal" onchange="updateDefectStats()" style="width: 100%; padding: 10px; border-radius: 6px; border: 1px solid #dee2e6;">
                    </select>
                </div>
                
                <div id="defectStatsDiv" style="margin-bottom: 15px; padding: 10px; background: #f8f9fa; border-radius: 6px; font-size: 0.9em; display: none;">
                </div>
                
                <div style="margin-bottom: 20px;">
                    <label style="display: block; margin-bottom: 8px; font-weight: 600;">업로드된 이미지</label>
                    <div style="padding: 10px; background: #f8f9fa; border-radius: 6px;">
                        ${selectedFile.name}
                    </div>
                </div>
                
                <div style="display: flex; gap: 10px;">
                    <button onclick="submitDefectRegistration()" style="flex: 1; padding: 12px; background: #28a745; color: white; border: none; border-radius: 8px; font-weight: 600; cursor: pointer;">
                        등록
                    </button>
                    <button onclick="closeDefectModal()" style="flex: 1; padding: 12px; background: #6c757d; color: white; border: none; border-radius: 8px; font-weight: 600; cursor: pointer;">
                        취소
                    </button>
                </div>
            </div>
        </div>
    `;
    
    document.body.insertAdjacentHTML('beforeend', modalHTML);
    updateDefectOptions();
}

async function updateDefectStats() {
    const product = document.getElementById('productSelectModal').value;
    const defect = document.getElementById('defectSelectModal').value;
    const statsDiv = document.getElementById('defectStatsDiv');
    
    try {
        const response = await fetch(`${API_BASE_URL}/defect/stats/${product}/${defect}`);
        const data = await response.json();
        
        statsDiv.innerHTML = `
            📊 현재 등록: <strong>${data.total_count}개</strong><br>
            🔢 다음 번호: <strong>${data.next_seqno}</strong>
        `;
        statsDiv.style.display = 'block';
    } catch (error) {
        statsDiv.style.display = 'none';
    }
}

function updateDefectOptions() {
    const productSelect = document.getElementById('productSelectModal');
    const defectSelect = document.getElementById('defectSelectModal');
    const selectedProduct = productSelect.value;
    
    const defects = defectConfig.products[selectedProduct].defects;
    defectSelect.innerHTML = defects.map(d => `<option value="${d}">${d}</option>`).join('');
    
    updateDefectStats();
}

function closeDefectModal() {
    const modal = document.getElementById('defectModal');
    if (modal) modal.remove();
}

async function submitDefectRegistration() {
    const product = document.getElementById('productSelectModal').value;
    const defect = document.getElementById('defectSelectModal').value;
    
    try {
        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('product_name', product);
        formData.append('defect_name', defect);
        
        // 로딩 표시
        const submitBtn = document.querySelector('#defectModal button[onclick="submitDefectRegistration()"]');
        const originalText = submitBtn.textContent;
        submitBtn.textContent = '등록 중...';
        submitBtn.disabled = true;
        
        const response = await fetch(`${API_BASE_URL}/register_defect`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || '등록 실패');
        }
        
        const data = await response.json();
        
        alert(
            `✅ ${data.message}\n\n` +
            `📁 파일명: ${data.filename}\n` +
            `📊 SEQ 번호: ${data.seqno}\n` +
            `📈 총 등록 수: ${data.total_defects}개\n` +
            `🔄 인덱스 재구축: ${data.index_rebuilt ? '완료' : '미실행'}\n\n` +
            `저장 경로: ${data.saved_path}`
        );
        
        closeDefectModal();
        
        // 인덱스가 재구축되었으면 상태 갱신
        if (data.index_rebuilt) {
            setTimeout(() => checkIndexStatus(), 1000);
        }
        
    } catch (error) {
        console.error('등록 오류:', error);
        alert(`❌ 등록 실패: ${error.message}`);
        
        // 버튼 복원
        const submitBtn = document.querySelector('#defectModal button[onclick="submitDefectRegistration()"]');
        if (submitBtn) {
            submitBtn.textContent = '등록';
            submitBtn.disabled = false;
        }
    }
}
// 인덱스 관리
async function checkIndexStatus() {
    checkIndexBtn.disabled = true;
    checkIndexBtn.textContent = '확인 중...';
    
    try {
        const response = await fetch(`${API_BASE_URL}/index/info`);
        const data = await response.json();
        
        if (data.status === 'index_built') {
            indexStatus.innerHTML = `
                ✅ <strong>인덱스 구축됨</strong><br>
                📁 이미지 수: ${data.gallery_size}개<br>
                🤖 모델: ${data.model_id}<br>
                💻 디바이스: ${data.device}<br>
                ⚡ FAISS: ${data.faiss_enabled ? '활성화' : '비활성화'}
            `;
            indexStatus.style.display = 'block';
            indexStatus.style.background = '#d4edda';
            indexStatus.style.color = '#155724';
        } else {
            indexStatus.innerHTML = '❌ 인덱스가 구축되지 않았습니다';
            indexStatus.style.display = 'block';
            indexStatus.style.background = '#f8d7da';
            indexStatus.style.color = '#721c24';
        }
    } catch (error) {
        indexStatus.innerHTML = `❌ 상태 확인 실패: ${error.message}`;
        indexStatus.style.display = 'block';
        indexStatus.style.background = '#f8d7da';
        indexStatus.style.color = '#721c24';
    } finally {
        checkIndexBtn.disabled = false;
        checkIndexBtn.textContent = '📊 상태 확인';
    }
}

async function rebuildIndex() {
    if (!confirm('인덱스를 재구축하시겠습니까?')) {
        return;
    }
    
    rebuildIndexBtn.disabled = true;
    rebuildIndexBtn.textContent = '구축 중...';
    
    indexStatus.innerHTML = '🔄 인덱스 구축 중...';
    indexStatus.style.display = 'block';
    indexStatus.style.background = '#d1ecf1';
    indexStatus.style.color = '#0c5460';
    
    try {
        const response = await fetch(`${API_BASE_URL}/build_index`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                gallery_dir: '../data/def_split',
                save_index: true,
                index_save_dir: './index_cache'
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || '인덱스 구축 실패');
        }
        
        const data = await response.json();
        
        indexStatus.innerHTML = `
            ✅ <strong>인덱스 구축 완료!</strong><br>
            📁 이미지 수: ${data.num_images}개<br>
            💾 저장됨: ${data.index_saved ? 'Yes' : 'No'}
        `;
        indexStatus.style.background = '#d4edda';
        indexStatus.style.color = '#155724';
        
        setTimeout(() => checkIndexStatus(), 1000);
        
    } catch (error) {
        indexStatus.innerHTML = `❌ 구축 실패: ${error.message}`;
        indexStatus.style.background = '#f8d7da';
        indexStatus.style.color = '#721c24';
    } finally {
        rebuildIndexBtn.disabled = false;
        rebuildIndexBtn.textContent = '🔄 재구축';
    }
}

// 상태 메시지
function showStatus(message, type) {
    statusMessage.className = `status-message ${type}`;
    statusMessage.textContent = message;
    statusMessage.style.display = 'block';
}

function showAnomalyStatus(message, type) {
    anomalyStatusMessage.className = `status-message ${type}`;
    anomalyStatusMessage.textContent = message;
    anomalyStatusMessage.style.display = 'block';
}

// 페이지 로드 시
window.addEventListener('load', async () => {
    try {
        const response = await fetch(`${API_BASE_URL}/health2`);
        const data = await response.json();
        
        if (data.status === 'healthy') {
            if (!data.index_built) {
                showStatus('⚠️ 인덱스가 구축되지 않았습니다.', 'info');
            }
        }
    } catch (error) {
        showStatus('⚠️ API 서버에 연결할 수 없습니다.', 'error');
    }
    
    setTimeout(() => checkIndexStatus(), 2000);
});
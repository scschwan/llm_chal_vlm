// web/static/js/matching.js

// API 서버 주소
const API_BASE_URL = '';

// 전역 변수
let selectedFile = null;
let searchResults = null;
let uploadedImagePath = null;
let defectConfig = null;

// 전역 변수에 추가
let currentSearchResult = null;
let currentAnomalyResult = null;


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
        
        // 서버에서 반환한 실제 저장 경로 사용
        uploadedImagePath = data.uploaded_file || `uploads/${selectedFile.name}`;
        
        console.log("업로드된 이미지 경로:", uploadedImagePath);
        
        // 검색 결과 저장 (매뉴얼 생성용)
        currentSearchResult = data.top_k_results[0];
        
        displayResults(data);
        showStatus(`검색 완료! ${data.top_k_results.length}개의 유사 이미지를 찾았습니다.`, 'success');

        // 매뉴얼 생성 버튼 표시
        if (data.top_k_results.length > 0) {
            detectButton.disabled = false;
            //document.getElementById('search-manual-button-container').style.display = 'block';
            const cont = document.getElementById('search-manual-button-container');
            if (cont) cont.style.display = 'block';
        }

        // ❗ 검색 응답을 전역으로 보존
        window.searchResults = data;

        // 응답 키가 top_k_results가 아닐 수도 있으니 안전하게 추출
        const candidates =
        (data && (data.top_k_results || data.results || data.items)) || [];
        window.currentSearchResult = candidates.length ? candidates[0] : null;

        // (선택) 디버그
        console.log('[performSearch] topK len =', candidates.length);
        console.log('[performSearch] top1 =', window.currentSearchResult);

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
    const smbc = document.getElementById('search-manual-button-container');
    if (smbc && data.top_k_results?.length) smbc.style.display = 'block';

    window.searchResults = data;
    const candidates =
        (data && (data.top_k_results || data.results || data.items)) || [];
    if (!window.currentSearchResult && candidates.length) {
        window.currentSearchResult = candidates[0];
    }
}

// TOP-1 스왑
function swapTopResult(clickedIndex) {
  const sr = window.searchResults;
  const list =
    (sr && (sr.top_k_results || sr.results || sr.items)) || [];

  if (!list.length || clickedIndex <= 0 || clickedIndex >= list.length) return;

  const tmp = list[0];
  list[0] = list[clickedIndex];
  list[clickedIndex] = tmp;

  // ❗ 전역 Top-1 갱신
  window.currentSearchResult = list[0];

  // 다시 렌더(카드 data-*도 재설정됨)
  displayResults(window.searchResults);

  // 선택 표시/안내 텍스트(있을 때만)
  const refInfo = document.getElementById('anomalyRefInfo');
  if (refInfo && list[0]) {
    const s = (list[0].similarity_score ?? 0) * 100;
    refInfo.innerHTML = `✅ ${list[0].image_name || ''} (유사도: ${s.toFixed(1)}%)`;
  }

  showStatus(`TOP-1이 ${list[0].image_name || ''}으로 변경되었습니다.`, 'success');
}


// 이상 검출
// 기존 performAnomalyDetection 함수 수정 - 결과 저장 및 버튼 표시
async function performAnomalyDetection() {
    if (!searchResults || searchResults.top_k_results.length === 0) {
        showAnomalyStatus('먼저 유사도 검색을 실행하세요.', 'error');
        return;
    }

    const top1 = currentSearchResult;                 // ✅
    const refPath = top1?.image_path || null;         // ✅
    const filename = top1?.image_name || '';
    const parts = filename.split('_');
    const product = parts[0] || null;                 // 파일명 규칙: prod_defect_xxx

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
                //reference_image_path: null,
                //product_name: null
                reference_image_path: refPath,
                product_name: product
            })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || '이상 검출 실패');
        }

        const data = await response.json();
        
        // 이상 검출 결과 저장 (매뉴얼 생성용)
        currentAnomalyResult = data;
        
        displayAnomalyResults(data);
        showAnomalyStatus('이상 검출 완료!', 'success');

        // 매뉴얼 생성 버튼 표시
        //document.getElementById('anomaly-manual-button-container').style.display = 'block';
        const anomBtns = document.getElementById('anomaly-manual-button-container');
        if (anomBtns) anomBtns.style.display = 'block';

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

// 유사도 검색 탭에서 매뉴얼 생성
function generateManualFromSearch() {
    if (!currentSearchResult || !uploadedImagePath) {
        showStatus('먼저 유사도 검색을 수행해주세요.', 'error');
        return;
    }
    
    // 파일명에서 제품명/불량명 추출
    const filename = currentSearchResult.image_name || currentSearchResult.path.split('/').pop();
    const parts = filename.split('_');
    
    if (parts.length < 2) {
        showStatus('파일명 형식 오류입니다. (제품명_불량명_번호 형식이어야 합니다)', 'error');
        return;
    }
    
    const product = parts[0];
    const defect = parts[1];
    
    // 매뉴얼 탭으로 전환
    const manualTab = document.querySelector('[data-tab="manual"]');
    switchTab(manualTab);
    
    // 고급 분석 실행
    executeAdvancedAnalysis(uploadedImagePath, product, defect);
}

// 이상 영역 검출 탭에서 매뉴얼 생성
function generateManualFromAnomaly() {
    if (!currentAnomalyResult || !uploadedImagePath) {
        showAnomalyStatus('먼저 이상 영역 검출을 수행해주세요.', 'error');
        return;
    }
    
    // 매뉴얼 탭으로 전환
    const manualTab = document.querySelector('[data-tab="manual"]');
    switchTab(manualTab);
    
    // 고급 분석 실행
    executeAdvancedAnalysis(uploadedImagePath);
}

// 고급 분석 실행 (통합)
async function executeAdvancedAnalysis(imagePath) {
    // UI 초기화
    document.getElementById('manual-info-section').style.display = 'none';
    document.getElementById('manual-result-section').style.display = 'none';
    document.getElementById('manual-error-section').style.display = 'none';
    document.getElementById('manual-processing').style.display = 'block';
    
    const stepElement = document.getElementById('manual-processing-step');
    
    try {
        stepElement.textContent = '🔍 종합 분석을 시작합니다...';
        
        // API 호출
        const response = await fetch(`${API_BASE_URL}/generate_manual_advanced`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image_path: imagePath
            })
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || '분석 중 오류가 발생했습니다.');
        }
        
        const data = await response.json();
        
        if (data.status !== 'success') {
            throw new Error('분석에 실패했습니다.');
        }
        
        // 처리 완료
        document.getElementById('manual-processing').style.display = 'none';
        
        // 분석 정보 표시
        displayManualInfo(data);
        
        // 결과 표시
        displayManualResult(data);
        
        showStatus('AI 분석이 완료되었습니다!', 'success');
        
    } catch (error) {
        console.error('매뉴얼 생성 오류:', error);
        document.getElementById('manual-processing').style.display = 'none';
        document.getElementById('manual-error-section').style.display = 'block';
        document.getElementById('manual-error-message').textContent = error.message;
    }
}

// 분석 정보 표시
function displayManualInfo(data) {
    const infoSection = document.getElementById('manual-info-section');
    
    document.getElementById('manual-product').textContent = 
        data.similarity?.product || data.defect_info?.product || 'N/A';
    
    const defectKo = data.defect_info?.ko || 'N/A';
    const defectEn = data.defect_info?.en || 'N/A';
    document.getElementById('manual-defect').textContent = 
        `${defectKo} (${defectEn})`;
    
    const score = data.anomaly?.score || 0;
    document.getElementById('manual-score').textContent = 
        `${(score * 100).toFixed(1)}%`;
    
    infoSection.style.display = 'block';
}

// 결과 표시
function displayManualResult(data) {
    const resultSection = document.getElementById('manual-result-section');
    
    // 이미지 표시
    if (data.anomaly) {
        document.getElementById('manual-normal-image').src = data.anomaly.normal_image_url || '';
        document.getElementById('manual-overlay-image').src = data.anomaly.overlay_image_url || '';
    }
    
    // 입력 이미지 표시
    if (uploadedImagePath) {
        document.getElementById('manual-defect-image').src = `/api/image/${uploadedImagePath}`;
    }
    
    // 참조 매뉴얼 표시
    if (data.manual) {
        const causesDiv = document.getElementById('manual-causes');
        const actionsDiv = document.getElementById('manual-actions');
        
        causesDiv.innerHTML = data.manual.원인 && data.manual.원인.length > 0
            ? data.manual.원인.map(c => `<div>${c}</div>`).join('')
            : '<div style="color: #94a3b8;">매뉴얼 정보가 없습니다.</div>';
        
        actionsDiv.innerHTML = data.manual.조치 && data.manual.조치.length > 0
            ? data.manual.조치.map(a => `<div>${a}</div>`).join('')
            : '<div style="color: #94a3b8;">매뉴얼 정보가 없습니다.</div>';
    }
    
    // VLM 분석 결과 표시
    if (data.vlm_analysis) {
        document.getElementById('manual-vlm-analysis').textContent = data.vlm_analysis;
    } else {
        document.getElementById('manual-vlm-analysis').textContent = 
            'VLM 분석 결과를 가져올 수 없습니다.';
    }
    
    // 처리 시간
    if (data.processing_time) {
        document.getElementById('manual-processing-time').textContent = data.processing_time;
    }
    
    resultSection.style.display = 'block';
}

// 매뉴얼 상세 토글
function toggleManualDetail() {
    const detailSection = document.getElementById('manual-detail-section');
    const toggleBtn = document.getElementById('toggle-manual-btn');
    
    if (detailSection.style.display === 'none') {
        detailSection.style.display = 'block';
        toggleBtn.textContent = '접기';
    } else {
        detailSection.style.display = 'none';
        toggleBtn.textContent = '펼치기';
    }
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

// 기존 함수 교체
function getTop1Meta() {
  // 1) 전역 TOP-1 가장 우선
  let top1 =
    window.currentSearchResult ??
    (window.searchResults &&
      (window.searchResults.top_k_results ||
       window.searchResults.results ||
       window.searchResults.items || [])[0]) ??
    null;

  // 2) 그래도 없으면 DOM에서 복구 (첫 카드 기준)
  if (!top1) {
    const card =
      document.querySelector('.result-card.active') ||
      document.querySelector('.result-card');
    if (card) {
      top1 = {
        image_path: card.dataset.imagePath || null,
        image_name: card.dataset.imageName || null,
      };
    }
  }

  if (!top1) {
    console.warn('[getTop1Meta] no top1 in globals/DOM');
    return { product: null, defect: null, top1_image_path: null };
  }

  // 파일명 결정
  const rawName =
    top1.image_name ||
    (top1.image_path ? top1.image_path.split('/').pop() : '') ||
    '';
  const name = rawName.trim();
  const stem = name.replace(/\.[a-z0-9]+$/i, '').toLowerCase();

  // product/defect 추출 (언더바/대시 모두 허용)
  let product = null, defect = null;
  let parts = stem.split('_');
  if (parts.length >= 2) {
    product = parts[0]; defect = parts[1];
  } else {
    parts = stem.split('-');
    if (parts.length >= 2) { product = parts[0]; defect = parts[1]; }
    else {
      const m = /^([^_-]+)[_-]([^_-]+)/.exec(stem);
      if (m) { product = m[1]; defect = m[2]; }
    }
  }

  const top1_image_path = top1.image_path || null;
  console.log('[getTop1Meta]', { name, product, defect, top1_image_path });
  return { product, defect, top1_image_path };
}



// [추가] manual 탭 버튼 핸들러 바인딩
document.addEventListener('DOMContentLoaded', () => {
  const btnLLM = document.getElementById('btn-generate-llm');
  const btnVLM = document.getElementById('btn-generate-vlm');

  if (btnLLM) btnLLM.addEventListener('click', async () => {
    await generateManualBy('llm');
  });
  if (btnVLM) btnVLM.addEventListener('click', async () => {
    await generateManualBy('vlm');
  });
});

// [추가] 생성 공통 함수
// generateManualBy 함수의 VLM 응답 처리 부분 수정

async function generateManualBy(mode /* 'llm' | 'vlm' */) {
  try {
        if (!uploadedImagePath) {
        showStatus('먼저 유사도 검색으로 이미지를 업로드하세요.', 'error');
        return;
        }
        const { product, defect, top1_image_path } = getTop1Meta();
        if (!product || !defect) {
        showStatus('TOP-1 이미지에서 제품/불량을 식별할 수 없습니다. (파일명 규칙 확인)', 'error');
        return;
        }

        // anomaly 점수 있으면 같이 보냄(매뉴얼 의존도 ↑ 프롬프트에서 사용)
        const anomaly_score = window.currentAnomalyResult?.image_score ?? null;
        const is_anomaly    = window.currentAnomalyResult?.is_anomaly ?? null;

        const body = {
        image_path: uploadedImagePath,
        top1_image_path,
        product_name: product,
        defect_name: defect,
        anomaly_score,
        is_anomaly,
        max_new_tokens: 512,
        temperature: 0.7
        };

        const url = mode === 'vlm'
        ? `${API_BASE_URL}/manual/generate/vlm`
        : `${API_BASE_URL}/manual/generate/llm`;

        // 로딩 표시
        const manualStatus = document.getElementById('manual-error-section');
        if (manualStatus) manualStatus.style.display = 'none';
        showStatus(`(${mode.toUpperCase()}) 생성 중…`, 'info');

        const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
        });

        const data = await res.json();
        
        // ✅ 디버깅: 응답 전체 출력
        console.log('[generateManualBy] Full Response:', data);
        console.log('[generateManualBy] vlm_analysis:', data.vlm_analysis);
        console.log('[generateManualBy] llm_analysis:', data.llm_analysis);
        
        if (!res.ok) {
        throw new Error(data?.detail || data?.message || '생성 실패');
        }

    // UI 반영
        // 1) 기본 정보
        const productEl = document.getElementById('manual-product');
        const defectKoEl = document.getElementById('manual-defect-ko');
        const defectEnEl = document.getElementById('manual-defect-en');
        const fullNameKoEl = document.getElementById('manual-full-name-ko');
        const anomalyScoreEl = document.getElementById('manual-anomaly-score');
        const isAnomalyEl = document.getElementById('manual-is-anomaly');
        
        if (productEl) productEl.textContent = data.product || product || '';
        if (defectKoEl) defectKoEl.textContent = data.defect_ko || '';
        if (defectEnEl) defectEnEl.textContent = data.defect_en || '';
        if (fullNameKoEl) fullNameKoEl.textContent = data.full_name_ko || '';
        if (anomalyScoreEl) {
            const score = data.anomaly_score ?? anomaly_score ?? 0;
            anomalyScoreEl.textContent = typeof score === 'number' ? score.toFixed(4) : score;
        }
        if (isAnomalyEl) {
            isAnomalyEl.textContent = (data.is_anomaly ?? is_anomaly) ? '불량' : '정상';
        }
        
        // 2) 매뉴얼(원인/조치)
        const causesEl = document.getElementById('manual-causes');
        const actionsEl = document.getElementById('manual-actions');
        
        if (causesEl) {
            const causes = (data.manual?.원인 || []).map(t => `<li>${t}</li>`).join('');
            causesEl.innerHTML = causes ? `<ul>${causes}</ul>` : '매뉴얼 정보 없음';
        }
        
        if (actionsEl) {
            const actions = (data.manual?.조치 || []).map(t => `<li>${t}</li>`).join('');
            actionsEl.innerHTML = actions ? `<ul>${actions}</ul>` : '매뉴얼 정보 없음';
        }
        
        // 3) 분석 결과 영역 - ✅ 수정된 부분
        if (mode === 'llm') {
            // LLM 모드
            const vlmAnalysisEl = document.getElementById('manual-vlm-analysis');
            if (vlmAnalysisEl) {
                vlmAnalysisEl.style.display = 'none'; // VLM 영역 숨기기
            }
            
            // LLM 영역 표시
            let llmAnalysisEl = document.getElementById('manual-llm-analysis');
            if (!llmAnalysisEl) {
                // LLM 영역이 없으면 생성
                const container = document.querySelector('#manual-tab .manual-container');
                if (container) {
                    llmAnalysisEl = document.createElement('div');
                    llmAnalysisEl.id = 'manual-llm-analysis';
                    llmAnalysisEl.className = 'manual-section';
                    llmAnalysisEl.style.display = 'block';
                    container.appendChild(llmAnalysisEl);
                }
            }
            
            if (llmAnalysisEl) {
                llmAnalysisEl.style.display = 'block';
                llmAnalysisEl.innerHTML = `
                    <h3>🧠 LLM 분석 결과</h3>
                    <div class="analysis-content">
                        ${(data.llm_analysis || '분석 결과가 없습니다.').replace(/\n/g, '<br>')}
                    </div>
                `;
            }
            
        } else {
            // VLM 모드
            const llmAnalysisEl = document.getElementById('manual-llm-analysis');
            if (llmAnalysisEl) {
                llmAnalysisEl.style.display = 'none'; // LLM 영역 숨기기
            }
            
            // VLM 영역 표시
            const vlmAnalysisEl = document.getElementById('manual-vlm-analysis');
            if (vlmAnalysisEl) {
                vlmAnalysisEl.style.display = 'block';
                
                // ✅ vlm_analysis 전체 텍스트 처리
                let vlmText = data.vlm_analysis || '';
                
                // "ASSISTANT:" 이후 텍스트만 추출 (서버에서 처리했으면 불필요)
                if (vlmText.includes('ASSISTANT:')) {
                    vlmText = vlmText.split('ASSISTANT:').pop().trim();
                }
                
                // "USER:" 부분 제거
                if (vlmText.includes('USER:')) {
                    vlmText = vlmText.split('USER:')[0].trim();
                }
                
                console.log('[VLM] Processed text:', vlmText); // 디버깅
                
                vlmAnalysisEl.innerHTML = `
                    <h3>🤖 VLM 분석 결과</h3>
                    <div class="analysis-content">
                        ${vlmText ? vlmText.replace(/\n/g, '<br>') : '분석 결과가 없습니다.'}
                    </div>
                `;
            }
        }
        
        // 4) 처리 시간
        const processingTimeEl = document.getElementById('manual-processing-time');
        if (processingTimeEl && data.processing_time) {
            processingTimeEl.textContent = typeof data.processing_time === 'number' 
                ? data.processing_time.toFixed(2) 
                : data.processing_time;
        }
        
        showStatus(`(${mode.toUpperCase()}) 생성 완료`, 'success');
        
        // manual 탭으로 전환
        const manualTab = document.querySelector('.tab[data-tab="manual"]');
        if (manualTab) switchTab(manualTab);
        
    } catch (err) {
        console.error('[generateManualBy] Error:', err);
        const msg = String(err?.message || err);
        const errCtn = document.getElementById('manual-error-section');
        const errMsg = document.getElementById('manual-error-message');
        if (errCtn && errMsg) {
            errCtn.style.display = 'block';
            errMsg.textContent = msg;
        }
        showStatus(`생성 실패: ${msg}`, 'error');
    }
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
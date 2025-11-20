/**
 * 유사도 매칭 화면 스크립트 (V2 API 사용)
 */

// 전역 변수
let currentResults = [];
let uploadedImageData = null;
let search_id = 0;
let top1_similarity = 0;


// DOM 요소
const queryImage = document.getElementById('queryImage');
const queryImageName = document.getElementById('queryImageName');
const topKSlider = document.getElementById('topKSlider');
const topKValue = document.getElementById('topKValue');
const searchBtn = document.getElementById('searchBtn');
const searchProgress = document.getElementById('searchProgress');
const searchResults = document.getElementById('searchResults');
const totalResults = document.getElementById('totalResults');
const gallerySize = document.getElementById('gallerySize');
const mainResultImage = document.getElementById('mainResultImage');
const mainSimilarity = document.getElementById('mainSimilarity');
const mainProduct = document.getElementById('mainProduct');
const mainDefect = document.getElementById('mainDefect');
const mainFilename = document.getElementById('mainFilename');
const mainScore = document.getElementById('mainScore');
const thumbnailGrid = document.getElementById('thumbnailGrid');
const statsCard = document.getElementById('statsCard');
const avgSimilarity = document.getElementById('avgSimilarity');
const maxSimilarity = document.getElementById('maxSimilarity');
const minSimilarity = document.getElementById('minSimilarity');
const reSearchBtn = document.getElementById('reSearchBtn');
const nextBtn = document.getElementById('nextBtn');
const registerBtn = document.getElementById('registerBtn');
const registerModal = document.getElementById('registerModal');
const modalClose = document.getElementById('modalClose');
const modalCancelBtn = document.getElementById('modalCancelBtn');
const modalConfirmBtn = document.getElementById('modalConfirmBtn');
const productSelect = document.getElementById('productSelect');
const defectSelect = document.getElementById('defectSelect');

 // 로그아웃 함수
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
    
    
// 페이지 로드 시 인증 확인
document.addEventListener('DOMContentLoaded', async () => {
    try {
        const response = await fetch('/api/auth/check');
        const data = await response.json();
        
        if (!data.authenticated) {
            window.location.href = '/login.html';
        }
    } catch (error) {
        console.error('인증 확인 실패:', error);
        window.location.href = '/login.html';
    }
});

// 페이지 로드 시 초기화
document.addEventListener('DOMContentLoaded', () => {
    console.log('[SEARCH] 페이지 로드 완료');
    
    // 세션에서 업로드 이미지 복원
    restoreUploadedImage();
    
    // 이벤트 리스너 등록
    initEventListeners();
    
    // 인덱스 상태 확인
    checkSearchIndexStatus();

    //불량 데이터 목록 조회
    loadDefectMapping();

    setupProductChangeListener();
});

/**
 * 이벤트 리스너 초기화
 */
function initEventListeners() {
    // TOP-K 슬라이더
    topKSlider.addEventListener('input', (e) => {
        topKValue.textContent = e.target.value;
    });
    
    // 검색 버튼
    searchBtn.addEventListener('click', performSearch);
    
    // 다시 검색
    reSearchBtn.addEventListener('click', () => {
        searchResults.style.display = 'none';
        statsCard.style.display = 'none';
    });
    
    // 다음 단계
    nextBtn.addEventListener('click', goToNextPage);
    
    // 불량 등록
    registerBtn.addEventListener('click', openRegisterModal);
    
    // 모달 닫기
    modalClose.addEventListener('click', closeRegisterModal);
    modalCancelBtn.addEventListener('click', closeRegisterModal);
    
    // 모달 확인
    modalConfirmBtn.addEventListener('click', confirmRegister);
    
    // 제품/불량 선택 시 파일명 미리보기 업데이트
    //productSelect.addEventListener('change', updateFilenamePreview);
    //defectSelect.addEventListener('change', updateFilenamePreview);
}

/**
 * 업로드된 이미지 복원
 */
function restoreUploadedImage() {
    const savedData = SessionData.get('uploadedImage');
    
    if (!savedData || !savedData.preview) {
        console.warn('[SEARCH] 업로드된 이미지가 없습니다');
        showNotification('먼저 이미지를 업로드해주세요', 'warning');
        setTimeout(() => {
            window.location.href = '/upload.html';
        }, 2000);
        return;
    }
    
    uploadedImageData = savedData;
    
    // 쿼리 이미지 표시
    queryImage.src = savedData.preview;
    queryImageName.textContent = savedData.filename;
    
    console.log('[SEARCH] 이미지 복원 완료:', savedData.filename);
}

/**
 * 유사도 검색 수행 (V2 API 사용)
 */
async function performSearch() {
    if (!uploadedImageData) {
        showNotification('업로드된 이미지가 없습니다', 'error');
        return;
    }
    
    const topK = parseInt(topKSlider.value);
    
    console.log(`[SEARCH V2] 검색 시작: TOP-${topK}`);
    
    try {
        // UI 상태 변경
        searchBtn.disabled = true;
        searchBtn.textContent = '검색 중...';
        searchResults.style.display = 'none';
        searchProgress.style.display = 'block';
        statsCard.style.display = 'none';
        
        // V2 API로 검색 요청
        const response = await fetch(`${API_BASE_URL}/v2/search/similarity`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query_image_path: uploadedImageData.file_path,
                top_k: topK,
                index_type: 'defect'
            })
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || '검색 실패');
        }
        
        const data = await response.json();
        console.log('[SEARCH V2] 검색 완료:', data);
        console.log('[SEARCH V2] search_id:', data.search_id);  // ✅ 추가
        console.log('[SEARCH V2] top1_similarity:', data.top1_similarity);  // ✅ 추가
        search_id = data.search_id;
        top1_similarity = data.top1_similarity;
        // 결과 저장 (V2 응답 구조 사용)
        currentResults = data.results;
        
        // 결과 표시
        displayResults(data);
        
        // 통계 표시
        displayStatistics(data.results);
        
        // 세션에 저장
        SessionData.set('searchResults', {
            results: data.results,
            query_image: data.query_image,
            top1: data.results[0],
            search_id: data.search_id,              // ✅ 추가
            top1_similarity: data.top1_similarity   // ✅ 추가
        });
        
        showNotification('검색 완료', 'success');
        
    } catch (error) {
        console.error('[SEARCH V2] 검색 실패:', error);
        showNotification(`검색 실패: ${error.message}`, 'error');
    } finally {
        searchBtn.disabled = false;
        searchBtn.textContent = '🔍 유사 이미지 검색';
        searchProgress.style.display = 'none';
    }
}

/**
 * 검색 결과 표시 (V2 메타데이터 사용)
 */
function displayResults(data) {
    // 총 결과 수 표시
    totalResults.textContent = data.results.length;
    gallerySize.textContent = data.total_gallery_size;
    
    if (data.results.length === 0) {
        showNotification('검색 결과가 없습니다', 'warning');
        return;
    }
    
    // TOP-1 메인 결과 표시
    const top1 = data.results[0];
    displayMainResult(top1);
    
    // 나머지 썸네일 표시
    if (data.results.length > 1) {
        displayThumbnails(data.results.slice(1));
    } else {
        thumbnailGrid.innerHTML = '<p style="text-align: center; color: var(--text-secondary);">추가 결과가 없습니다</p>';
    }
    
    // 결과 섹션 표시
    searchResults.style.display = 'block';
}

/**
 * TOP-1 메인 결과 표시 (V2 메타데이터 사용)
 */
function displayMainResult(result) {
    // V2 API는 storage_url 제공
    //const imageUrl = result.storage_url || `/api/image/${result.local_path}`;
    const imageUrl = `/api/image/${result.local_path}` ;
    
    mainResultImage.src = imageUrl;
    mainSimilarity.textContent = `${(result.similarity_score * 100).toFixed(1)}%`;
    
    // V2에서는 product_name, defect_name 사용
    mainProduct.textContent = result.product_name || result.product_code || '-';
    mainDefect.textContent = result.defect_name || result.defect_code || '-';
    mainFilename.textContent = result.file_name || '-';
    mainScore.textContent = result.similarity_score.toFixed(4);
    
    // 유사도에 따라 배지 색상 변경
    const similarity = result.similarity_score * 100;
    if (similarity >= 90) {
        mainSimilarity.style.background = 'var(--success-color)';
    } else if (similarity >= 70) {
        mainSimilarity.style.background = 'var(--warning-color)';
    } else {
        mainSimilarity.style.background = 'var(--danger-color)';
    }
}

/**
 * 썸네일 결과 표시 (V2 메타데이터 사용)
 */
function displayThumbnails(results) {
    thumbnailGrid.innerHTML = results.map((result, index) => {
        //const imageUrl = result.storage_url || `/api/image/${result.local_path}`;
        const imageUrl = `/api/image/${result.local_path}`;
        const productName = result.product_name || result.product_code || '-';
        const defectName = result.defect_name || result.defect_code || '-';
        const fileName = result.file_name || '-';
        
        return `
        <div class="thumbnail-item" onclick="swapWithMain(${index + 1})">
            <img 
                src="${imageUrl}" 
                class="thumbnail-image" 
                alt="Similar ${index + 2}"
            >
            <div class="thumbnail-info">
                <p class="similarity">${(result.similarity_score * 100).toFixed(1)}%</p>
                <p><strong>${productName}</strong> - ${defectName}</p>
                <p style="font-size: 0.75rem; color: var(--text-secondary);">
                    ${fileName}
                </p>
            </div>
        </div>
    `}).join('');
}

/**
 * 썸네일과 메인 이미지 교체
 */
function swapWithMain(index) {
    console.log(`[SEARCH V2] 이미지 교체: ${index}번 → TOP-1`);
    
    // 현재 TOP-1과 선택된 썸네일 교체
    const temp = currentResults[0];
    currentResults[0] = currentResults[index];
    currentResults[index] = temp;
    
    // UI 업데이트
    displayMainResult(currentResults[0]);
    displayThumbnails(currentResults.slice(1));
    
    // 세션 업데이트
    SessionData.set('searchResults', {
        results: currentResults,
        top1: currentResults[0]
    });
    
    showNotification('TOP-1 이미지 변경됨', 'success');
}

/**
 * 통계 표시
 */
function displayStatistics(results) {
    if (results.length === 0) return;
    
    const similarities = results.map(r => r.similarity_score * 100);
    const avg = similarities.reduce((a, b) => a + b, 0) / similarities.length;
    const max = Math.max(...similarities);
    const min = Math.min(...similarities);
    
    avgSimilarity.textContent = `${avg.toFixed(1)}%`;
    maxSimilarity.textContent = `${max.toFixed(1)}%`;
    minSimilarity.textContent = `${min.toFixed(1)}%`;
    
    statsCard.style.display = 'block';
}

/**
 * 인덱스 상태 확인 (V2 API)
 */
async function checkSearchIndexStatus() {
    try {
        const response = await fetch(`${API_BASE_URL}/v2/search/index/status`);
        const data = await response.json();
        
        const statusEl = document.getElementById('indexStatus');
        if (!statusEl) return;
        
        if (data.status === 'success' && data.index_built) {
            statusEl.innerHTML = `
                <p class="status-ok">
                    ✅ 불량 이미지 인덱스 활성<br>
                    <small>${data.gallery_count}개 이미지</small>
                </p>
            `;
        } else {
            statusEl.innerHTML = `
                <p class="status-error">❌ 인덱스 미구축</p>
            `;
        }
        
    } catch (error) {
        console.error('[SEARCH V2] 인덱스 상태 확인 실패:', error);
    }
}

/**
 * 다음 페이지로 이동
 */
function goToNextPage() {
    if (!currentResults || currentResults.length === 0) {
        showNotification('먼저 검색을 수행해주세요', 'warning');
        return;
    }
    
    // TOP-1 정보를 세션에 저장 (V2 메타데이터 포함)
    const top1 = currentResults[0];
    SessionData.set('selectedMatch', {
        image_path: top1.local_path,
        product_code: top1.product_code,
        product_name: top1.product_name,
        defect_code: top1.defect_code,
        defect_name: top1.defect_name,
        similarity: top1.similarity_score,
        search_id:  search_id || searchResults?.search_id ,           // ✅ 추가
        top1_similarity:top1_similarity || searchResults?.top1_similarity // ✅ 추가
    });
    
    console.log('[SEARCH V2] 이상 검출 페이지로 이동');
    console.log('[SEARCH V2] 전달 데이터:', {
        search_id: searchResults?.search_id,
        similarity: searchResults?.top1_similarity
    });
    window.location.href = '/anomaly.html';
}

/**
 * 불량 등록 모달 열기
 */
function openRegisterModal() {
   if (!uploadedImageData) {
        showNotification('업로드된 이미지가 없습니다', 'error');
        return;
    }
    
    registerModal.style.display = 'flex';
    initializeProductSelect();
    
}

/**
 * 불량 등록 모달 닫기
 */
function closeRegisterModal() {
    registerModal.style.display = 'none';
    productSelect.value = '';
    defectSelect.value = '';
}


/**
 * 불량 등록 확인
 */
async function confirmRegister() {
    const productSelect = document.getElementById('productSelect');
    const defectSelect = document.getElementById('defectSelect');
    
    if (!productSelect || !defectSelect) {
        alert('필수 요소를 찾을 수 없습니다.');
        return;
    }
    
    const product = productSelect.value;
    const defect = defectSelect.value;
    
    if (!product || !defect || !uploadedImageData || !uploadedImageData.file_path) {
        alert('모든 필드를 입력해주세요.');
        return;
    }
    
    try {
        console.log('[REGISTER] 등록 시작:', { 
            product, 
            defect, 
            filename: uploadedImageData.filename,
            file_path: uploadedImageData.file_path
        });
        
        // FormData 생성 - 파일 경로 전달
        const formData = new FormData();
        formData.append('product_name', product);
        formData.append('defect_type', defect);
        formData.append('file_path', uploadedImageData.file_path);
        
        const response = await fetch('/v2/search/register_defect', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            console.error('[REGISTER] 서버 응답 에러:', errorData);
            throw new Error(errorData.detail || '등록 실패');
        }
        
        const result = await response.json();
        console.log('[REGISTER] 등록 성공:', result);
        
        alert('불량 이미지가 성공적으로 등록되었습니다.');
        
        // 모달 닫기 및 초기화
        registerModal.style.display = 'none';
        productSelect.value = '';
        defectSelect.innerHTML = '<option value="">먼저 제품을 선택하세요</option>';
        
    } catch (error) {
        console.error('[REGISTER] 등록 실패:', error);
        alert(`등록 실패: ${error.message}`);
    }
}

// defect_mapping.json 로드 및 select 박스 초기화
let defectMapping = null;

// 페이지 로드 시 defect_mapping.json 로드
async function loadDefectMapping() {
    try {
        const response = await fetch('/api/defect_mapping');
        defectMapping = await response.json();
        initializeProductSelect();
    } catch (error) {
        console.error('defect_mapping.json 로드 실패:', error);
        alert('제품 정보를 불러오는데 실패했습니다.');
    }
}

// 제품명 select 박스 초기화
function initializeProductSelect() {
    const productSelect = document.getElementById('productSelect');
    if (!productSelect) return;

    productSelect.innerHTML = '<option value="">선택하세요</option>';
    
    if (defectMapping && defectMapping.products) {
        // products 객체의 키를 순회 (키가 곧 product_code)
        Object.keys(defectMapping.products).forEach(productCode => {
            const product = defectMapping.products[productCode];
            const option = document.createElement('option');
            option.value = productCode;  // 키 값이 product_code (예: "prod1")
            option.textContent = `${productCode} (${product.name_ko})`;
            productSelect.appendChild(option);
        });
    }
}

// 제품 선택 시 불량 유형 select 박스 업데이트
function setupProductChangeListener() {
    const productSelect = document.getElementById('productSelect');
    if (!productSelect) return;
    
    productSelect.addEventListener('change', function() {
        const productCode = this.value;
        const defectSelect = document.getElementById('defectSelect');
        
        if (!defectSelect) return;
        
        defectSelect.innerHTML = '<option value="">선택하세요</option>';
        
        if (productCode && defectMapping && defectMapping.products[productCode]) {
            const defects = defectMapping.products[productCode].defects;
            
            Object.keys(defects).forEach(defectCode => {
                const defect = defects[defectCode];
                const option = document.createElement('option');
                option.value = defectCode;
                option.textContent = `${defectCode} (${defect.ko})`;
                defectSelect.appendChild(option);
            });
        }
        
        //updateFilenamePreview();
    });
}


// 제품 선택 시 불량 유형 select 박스 업데이트
document.getElementById('productSelect').addEventListener('change', function() {
    const productCode = this.value;
    const defectSelect = document.getElementById('defectSelect');
    
    defectSelect.innerHTML = '<option value="">선택하세요</option>';
    
    if (productCode && defectMapping && defectMapping.products[productCode]) {
        const defects = defectMapping.products[productCode].defects;
        
        // defects 객체의 키를 순회 (키가 곧 defect_code)
        Object.keys(defects).forEach(defectCode => {
            const defect = defects[defectCode];
            const option = document.createElement('option');
            option.value = defectCode;  // 키 값이 defect_code (예: "hole", "burr")
            option.textContent = `${defectCode} (${defect.ko})`;
            defectSelect.appendChild(option);
        });
    }
});

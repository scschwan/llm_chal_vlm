/**
 * 유사도 매칭 화면 스크립트
 */

// 전역 변수
let currentResults = [];
let uploadedImageData = null;

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
const filenamePreview = document.getElementById('filenamePreview');

// 페이지 로드 시 초기화
document.addEventListener('DOMContentLoaded', () => {
    console.log('[SEARCH] 페이지 로드 완료');
    
    // 세션에서 업로드 이미지 복원
    restoreUploadedImage();
    
    // 이벤트 리스너 등록
    initEventListeners();
    
    // 인덱스 상태 확인
    checkSearchIndexStatus();
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
    productSelect.addEventListener('change', updateFilenamePreview);
    defectSelect.addEventListener('change', updateFilenamePreview);
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
 * 유사도 검색 수행
 */
async function performSearch() {
    if (!uploadedImageData) {
        showNotification('업로드된 이미지가 없습니다', 'error');
        return;
    }
    
    const topK = parseInt(topKSlider.value);
    
    console.log(`[SEARCH] 검색 시작: TOP-${topK}`);
    
    try {
        // UI 상태 변경
        searchBtn.disabled = true;
        searchBtn.textContent = '검색 중...';
        searchResults.style.display = 'none';
        searchProgress.style.display = 'block';
        statsCard.style.display = 'none';
        
        // 검색 요청
        const response = await fetch(`${API_BASE_URL}/search/similarity`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query_image_path: uploadedImageData.file_path,
                top_k: topK
            })
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || '검색 실패');
        }
        
        const data = await response.json();
        console.log('[SEARCH] 검색 완료:', data);
        
        // 결과 저장
        currentResults = data.top_k_results;
        
        // 결과 표시
        displayResults(data);
        
        // 통계 표시
        displayStatistics(data.top_k_results);
        
        // 세션에 저장
        SessionData.set('searchResults', {
            results: data.top_k_results,
            query_image: data.query_image,
            top1: data.top_k_results[0]
        });
        
        showNotification('검색 완료', 'success');
        
    } catch (error) {
        console.error('[SEARCH] 검색 실패:', error);
        showNotification(`검색 실패: ${error.message}`, 'error');
    } finally {
        searchBtn.disabled = false;
        searchBtn.textContent = '🔍 유사 이미지 검색';
        searchProgress.style.display = 'none';
    }
}

/**
 * 검색 결과 표시
 */
function displayResults(data) {
    // 총 결과 수 표시
    totalResults.textContent = data.top_k_results.length;
    gallerySize.textContent = data.total_gallery_size;
    
    if (data.top_k_results.length === 0) {
        showNotification('검색 결과가 없습니다', 'warning');
        return;
    }
    
    // TOP-1 메인 결과 표시
    const top1 = data.top_k_results[0];
    displayMainResult(top1);
    
    // 나머지 썸네일 표시
    if (data.top_k_results.length > 1) {
        displayThumbnails(data.top_k_results.slice(1));
    } else {
        thumbnailGrid.innerHTML = '<p style="text-align: center; color: var(--text-secondary);">추가 결과가 없습니다</p>';
    }
    
    // 결과 섹션 표시
    searchResults.style.display = 'block';
}

/**
 * TOP-1 메인 결과 표시
 */
function displayMainResult(result) {
    mainResultImage.src = `/api/image/${result.image_path}`;
    mainSimilarity.textContent = `${(result.similarity_score * 100).toFixed(1)}%`;
    mainProduct.textContent = result.product;
    mainDefect.textContent = result.defect;
    mainFilename.textContent = result.image_path.split('/').pop();
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
 * 썸네일 결과 표시
 */
function displayThumbnails(results) {
    thumbnailGrid.innerHTML = results.map((result, index) => `
        <div class="thumbnail-item" onclick="swapWithMain(${index + 1})">
            <img 
                src="/api/image/${result.image_path}" 
                class="thumbnail-image" 
                alt="Similar ${index + 2}"
            >
            <div class="thumbnail-info">
                <p class="similarity">${(result.similarity_score * 100).toFixed(1)}%</p>
                <p><strong>${result.product}</strong> - ${result.defect}</p>
                <p style="font-size: 0.75rem; color: var(--text-secondary);">
                    ${result.image_path.split('/').pop()}
                </p>
            </div>
        </div>
    `).join('');
}

/**
 * 썸네일과 메인 이미지 교체
 */
function swapWithMain(index) {
    console.log(`[SEARCH] 이미지 교체: ${index}번 → TOP-1`);
    
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
 * 인덱스 상태 확인
 */
async function checkSearchIndexStatus() {
    try {
        const response = await fetch(`${API_BASE_URL}/search/index/status`);
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
        console.error('[SEARCH] 인덱스 상태 확인 실패:', error);
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
    
    // TOP-1 정보를 세션에 저장
    const top1 = currentResults[0];
    SessionData.set('selectedMatch', {
        image_path: top1.image_path,
        product: top1.product,
        defect: top1.defect,
        similarity: top1.similarity_score
    });
    
    console.log('[SEARCH] 이상 검출 페이지로 이동');
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
    updateFilenamePreview();
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
 * 파일명 미리보기 업데이트
 */
function updateFilenamePreview() {
    const product = productSelect.value || 'prod1';
    const defect = defectSelect.value || 'hole';
    
    filenamePreview.textContent = `${product}_${defect}_XXX.jpg`;
}

/**
 * 불량 등록 확인
 */
async function confirmRegister() {
    const product = productSelect.value;
    const defect = defectSelect.value;
    
    if (!product || !defect) {
        showNotification('제품명과 불량 유형을 선택해주세요', 'warning');
        return;
    }
    
    if (!uploadedImageData || !uploadedImageData.file_path) {
        showNotification('업로드된 이미지가 없습니다', 'error');
        return;
    }
    
    try {
        modalConfirmBtn.disabled = true;
        modalConfirmBtn.textContent = '등록 중...';
        
        // 파일 경로에서 실제 파일 가져오기
        const filePath = uploadedImageData.file_path;
        const filename = filePath.split('/').pop();
        
        // 업로드 디렉토리에서 파일 읽기
        const fileResponse = await fetch(`/api/image/${filePath}`);
        const blob = await fileResponse.blob();
        
        // FormData 생성
        const formData = new FormData();
        formData.append('file', blob, filename);
        formData.append('product_name', product);
        formData.append('defect_name', defect);
        
        // 등록 요청
        const response = await fetch(`${API_BASE_URL}/register_defect`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('등록 실패');
        }
        
        const data = await response.json();
        console.log('[REGISTER] 등록 완료:', data);
        
        showNotification(`불량 이미지 등록 완료: ${data.filename}`, 'success');
        
        closeRegisterModal();
        
        // 인덱스 상태 새로고침
        await checkSearchIndexStatus();
        
    } catch (error) {
        console.error('[REGISTER] 등록 실패:', error);
        showNotification(`등록 실패: ${error.message}`, 'error');
    } finally {
        modalConfirmBtn.disabled = false;
        modalConfirmBtn.textContent = '등록';
    }
}
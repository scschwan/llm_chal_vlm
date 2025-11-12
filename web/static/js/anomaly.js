/**
 * 이상 영역 검출 화면 스크립트
 */

// 전역 변수
let uploadedImageData = null;
let selectedMatchData = null;
let detectionResult = null;

// DOM 요소
const detectionProgress = document.getElementById('detectionProgress');
const progressMessage = document.getElementById('progressMessage');
const detectionResults = document.getElementById('detectionResults');
const detectionBadge = document.getElementById('detectionBadge');
const anomalyScore = document.getElementById('anomalyScore');
const normalImage = document.getElementById('normalImage');
const overlayImage = document.getElementById('overlayImage');
const maskImage = document.getElementById('maskImage');
const comparisonImage = document.getElementById('comparisonImage');
const productName = document.getElementById('productName');
const defectType = document.getElementById('defectType');
const imageScore = document.getElementById('imageScore');
const threshold = document.getElementById('threshold');
const judgment = document.getElementById('judgment');
const similarityScore = document.getElementById('similarityScore');
const reDetectBtn = document.getElementById('reDetectBtn');
const nextBtn = document.getElementById('nextBtn');

// 페이지 로드 시 초기화
document.addEventListener('DOMContentLoaded', () => {
    console.log('[ANOMALY] 페이지 로드 완료');
    
    // 세션에서 데이터 복원
    restoreSessionData();
    
    // 이벤트 리스너 등록
    initEventListeners();
    
    // 자동 검출 시작
    if (uploadedImageData && selectedMatchData) {
        performDetection();
    }
});

/**
 * 이벤트 리스너 초기화
 */
function initEventListeners() {
    // 다시 검출
    reDetectBtn.addEventListener('click', () => {
        detectionResults.style.display = 'none';
        performDetection();
    });
    
    // 다음 단계
    nextBtn.addEventListener('click', goToNextPage);
}

/**
 * 세션 데이터 복원
 */
function restoreSessionData() {
    // 업로드 이미지 데이터
    uploadedImageData = SessionData.get('uploadedImage');
    
    // 유사도 매칭 결과
    const searchResults = SessionData.get('searchResults');
    if (searchResults && searchResults.top1) {
        selectedMatchData = searchResults.top1;
    }
    
    // 데이터 검증
    if (!uploadedImageData) {
        console.warn('[ANOMALY] 업로드된 이미지가 없습니다');
        showNotification('업로드된 이미지가 없습니다', 'warning');
        setTimeout(() => {
            window.location.href = '/upload.html';
        }, 2000);
        return;
    }
    
    if (!selectedMatchData) {
        console.warn('[ANOMALY] 유사도 매칭 결과가 없습니다');
        showNotification('유사도 매칭을 먼저 수행해주세요', 'warning');
        setTimeout(() => {
            window.location.href = '/search.html';
        }, 2000);
        return;
    }
    
    console.log('[ANOMALY] 데이터 복원 완료');
    console.log('  입력 이미지:', uploadedImageData.filename);
    console.log('  TOP-1 매칭:', selectedMatchData.image_path);
    console.log('  제품:', selectedMatchData.product);
    console.log('  불량:', selectedMatchData.defect);
}

/**
 * 이상 검출 수행
 */
async function performDetection() {
    console.log('[ANOMALY] 이상 검출 시작');
    
    try {
        // UI 상태 변경
        detectionProgress.style.display = 'block';
        detectionResults.style.display = 'none';
        progressMessage.textContent = 'PatchCore 모델 로딩 중...';
        
        // 검출 요청
        const response = await fetch(`${API_BASE_URL}/anomaly/detect`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                test_image_path: uploadedImageData.file_path,
                reference_image_path: selectedMatchData.image_path,
                product_name: selectedMatchData.product
            })
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || '이상 검출 실패');
        }
        
        const data = await response.json();
        console.log('[ANOMALY] 검출 완료:', data);
        
        // 결과 저장
        detectionResult = data;
        
        // 결과 표시
        displayResults(data);
        
        // 세션에 저장
        SessionData.set('anomalyResult', {
            ...data,
            product: selectedMatchData.product,
            defect: selectedMatchData.defect,
            similarity: selectedMatchData.similarity_score
        });
        
        showNotification('이상 검출 완료', 'success');
        
    } catch (error) {
        console.error('[ANOMALY] 검출 실패:', error);
        showNotification(`이상 검출 실패: ${error.message}`, 'error');
        
        // 에러 시 검색 페이지로 돌아가기 옵션
        setTimeout(() => {
            if (confirm('이상 검출에 실패했습니다. 유사도 매칭 페이지로 돌아가시겠습니까?')) {
                window.location.href = '/search.html';
            }
        }, 1000);
    } finally {
        detectionProgress.style.display = 'none';
    }
}

/**
 * 검출 결과 표시
 */
function displayResults(data) {
    // 점수 표시
    const score = data.image_score;
    anomalyScore.textContent = score.toFixed(4);
    
    // 판정 배지
    if (data.is_anomaly) {
        detectionBadge.innerHTML = '<span class="badge-text">⚠️ 이상 검출</span>';
        detectionBadge.className = 'detection-badge anomaly';
    } else {
        detectionBadge.innerHTML = '<span class="badge-text">✅ 정상</span>';
        detectionBadge.className = 'detection-badge normal';
    }
    
    // 이미지 표시
    normalImage.src = `/api/image/${selectedMatchData.image_path}`;
    overlayImage.src = data.overlay_url;
    maskImage.src = data.mask_url;
    comparisonImage.src = data.comparison_url;
    
    // 상세 정보
    productName.textContent = selectedMatchData.product;
    defectType.textContent = selectedMatchData.defect;
    imageScore.textContent = score.toFixed(4);
    threshold.textContent = data.image_tau.toFixed(4);
    
    // 판정 결과 (색상 포함)
    if (data.is_anomaly) {
        judgment.textContent = '이상 (Anomaly)';
        judgment.style.color = 'var(--danger-color)';
    } else {
        judgment.textContent = '정상 (Normal)';
        judgment.style.color = 'var(--success-color)';
    }
    
    // 유사도 점수
    if (selectedMatchData.similarity_score !== undefined) {
        similarityScore.textContent = `${(selectedMatchData.similarity_score * 100).toFixed(1)}%`;
    } else {
        similarityScore.textContent = '-';
    }
    
    // 결과 섹션 표시
    detectionResults.style.display = 'block';
    
    // 스크롤 이동
    detectionResults.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/**
 * 다음 페이지로 이동
 */
function goToNextPage() {
    if (!detectionResult) {
        showNotification('먼저 이상 검출을 수행해주세요', 'warning');
        return;
    }
    
    console.log('[ANOMALY] 대응 매뉴얼 페이지로 이동');
    window.location.href = '/manual.html';
}
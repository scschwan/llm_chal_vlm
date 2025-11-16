/**
 * 이상 영역 검출 화면 스크립트
 */

// 전역 변수
let uploadedImageData = null;
let selectedMatchData = null;
let detectionResult = null;
let global_search_id = 0;
let global_top1_similarity = 0;

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
    const selectedMatch = SessionData.get('selectedMatch');

    // ✅ searchResults에서 search_id, top1_similarity 먼저 추출
    if (searchResults) {
        global_search_id = searchResults.search_id;
        global_top1_similarity = searchResults.top1_similarity;
        
        if (searchResults.top1) {
            selectedMatchData = searchResults.top1;
            // ✅ selectedMatchData에도 추가
            selectedMatchData.search_id = searchResults.search_id;
            selectedMatchData.top1_similarity = searchResults.top1_similarity;
        }
    }
    
    // ✅ selectedMatch가 있으면 우선 사용 (goToNextPage에서 저장한 데이터)
    if (selectedMatch) {
        selectedMatchData = selectedMatch;
        // selectedMatch에는 이미 search_id, top1_similarity가 포함되어 있음
        if (selectedMatch.search_id) {
            global_search_id = selectedMatch.search_id;
        }
        if (selectedMatch.top1_similarity) {
            global_top1_similarity = selectedMatch.top1_similarity;
        }
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
    console.log('  TOP-1 불량 이미지:', selectedMatchData.local_path || selectedMatchData.image_path);
    console.log('  제품:', selectedMatchData.product_name || selectedMatchData.product_code);
    console.log('  불량:', selectedMatchData.defect_name || selectedMatchData.defect_code);
    console.log('  search_id:', selectedMatchData.search_id || global_search_id);
    console.log('  similarity:', selectedMatchData.top1_similarity || selectedMatchData.similarity || global_top1_similarity);
}
/**
 * 이상 검출 수행
 */
async function performDetection() {
    console.log('[ANOMALY] 이상 검출 시작');
    console.log('[ANOMALY] 정상 이미지 인덱스로 전환하여 검출 수행');
    
    // V2 API 응답 구조에서 필요한 값 추출
    const productCode = selectedMatchData.product_code || selectedMatchData.product;
    const defectImagePath = selectedMatchData.local_path || selectedMatchData.image_path;
    const defectCode = selectedMatchData.defect_code || selectedMatchData.defect;

    // ✅ search_id와 similarity 추출 (우선순위: selectedMatchData > global 변수)
    let searchId = selectedMatchData.search_id || global_search_id;
    let similarity = selectedMatchData.top1_similarity || selectedMatchData.similarity || global_top1_similarity;
    
    console.log('[ANOMALY] searchId:', searchId);
    console.log('[ANOMALY] similarity:', similarity);
    
    try {
        // UI 상태 변경
        detectionProgress.style.display = 'block';
        detectionResults.style.display = 'none';
        progressMessage.textContent = '정상 이미지 인덱스 로딩 중...';
        
        // ✅ 검출 요청 (정상 이미지 기준)
        const response = await fetch(`${API_BASE_URL}/anomaly/detect`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                test_image_path: uploadedImageData.file_path,
                product_name: productCode,
                top1_defect_image: defectImagePath,
                defect_name: defectCode,
                search_id: searchId,           // ✅ 전달
                similarity_score: similarity   // ✅ 전달
            })
        });
        
        progressMessage.textContent = 'PatchCore 이상 검출 중...';
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || '이상 검출 실패');
        }
        
        const data = await response.json();
        console.log('[ANOMALY] 검출 완료:', data);
        console.log('[ANOMALY] response_id:', data.response_id);
        console.log('[ANOMALY] 정상 기준 이미지:', data.top1_normal_image);
        
        // 결과 저장
        detectionResult = data;
        
        // 결과 표시
        displayResults(data);
        
        // 세션에 저장 (V2 구조 반영)
        SessionData.set('anomalyResult', {
            ...data,
            product: selectedMatchData.product_name || selectedMatchData.product_code,
            product_code: selectedMatchData.product_code || productCode,
            product_name: selectedMatchData.product_name,
            defect: selectedMatchData.defect_name || selectedMatchData.defect_code,
            defect_code: selectedMatchData.defect_code || defectCode,
            defect_name: selectedMatchData.defect_name,
            similarity: similarity,  // ✅ 계산된 값 사용
            image_score: data.anomaly_score || data.image_score,
            top1_defect_image: defectImagePath,
            search_id: searchId,           // ✅ 계산된 값 사용
            response_id: data.response_id
        });
        
        showNotification('이상 검출 완료', 'success');
        
    } catch (error) {
        console.error('[ANOMALY] 검출 실패:', error);
        showNotification(`이상 검출 실패: ${error.message}`, 'error');
        
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
    const score = data.anomaly_score || data.image_score || 0;
    anomalyScore.textContent = score.toFixed(4);
    
    // 판정 배지
    if (data.is_anomaly) {
        detectionBadge.innerHTML = '<span class="badge-text">⚠️ 이상 검출</span>';
        detectionBadge.className = 'detection-badge anomaly';
    } else {
        detectionBadge.innerHTML = '<span class="badge-text">✅ 정상</span>';
        detectionBadge.className = 'detection-badge normal';
    }
    
    // ✅ 이미지 표시: top1_normal_image 우선 사용
    const normalImagePath = data.top1_normal_image || data.reference_normal_path;
    if (normalImagePath) {
        normalImage.src = `/api/image/${normalImagePath}`;
    }
    
    if (data.overlay_url) {
        overlayImage.src = data.overlay_url;
    }
    if (data.mask_url) {
        maskImage.src = data.mask_url;
    }
    if (data.comparison_url) {
        comparisonImage.src = data.comparison_url;
    }
    
    // 상세 정보 (V2 구조 반영)
    productName.textContent = selectedMatchData.product_name || selectedMatchData.product_code || '-';
    defectType.textContent = selectedMatchData.defect_name || selectedMatchData.defect_code || '-';
    imageScore.textContent = score.toFixed(4);

    // ✅ threshold 처리
    const thresholdValue = data.image_tau || data.threshold || 0.5;
    threshold.textContent = thresholdValue.toFixed(4);
    
    // 판정 결과 (색상 포함)
    if (data.is_anomaly) {
        judgment.textContent = '이상 (Anomaly)';
        judgment.style.color = 'var(--danger-color)';
    } else {
        judgment.textContent = '정상 (Normal)';
        judgment.style.color = 'var(--success-color)';
    }
    
    // ✅ 유사도 점수 (서버에서 받은 top1_similarity 또는 selectedMatchData 사용)
    const similarityValue = data.top1_similarity || selectedMatchData.similarity_score || selectedMatchData.similarity || global_top1_similarity;
    if (similarityValue !== undefined && similarityValue !== null) {
        similarityScore.textContent = `${(similarityValue * 100).toFixed(1)}%`;
    } else {
        similarityScore.textContent = '-';
    }
    
    // 결과 섹션 표시
    detectionResults.style.display = 'block';
    
    // 스크롤 이동
    detectionResults.scrollIntoView({ behavior: 'smooth', block: 'start' })
}

/**
 * 다음 페이지로 이동
 */
function goToNextPage() {
    if (!detectionResult) {
        showNotification('먼저 이상 검출을 수행해주세요', 'warning');
        return;
    }
    
    // ✅ response_id 확인
    const anomalyResult = SessionData.get('anomalyResult');
    console.log('[ANOMALY] 대응 매뉴얼 페이지로 이동');
    console.log('[ANOMALY] 전달 데이터:', {
        response_id: anomalyResult?.response_id,
        search_id: anomalyResult?.search_id
    });
    
    window.location.href = '/manual.html';
}
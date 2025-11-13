/**
 * 대응 매뉴얼 생성 화면 스크립트
 */

// 전역 변수
let uploadedImageData = null;
let selectedMatchData = null;
let anomalyResultData = null;
let selectedModel = null;
let selectedRating = null;
let generatedManual = null;

// DOM 요소
const top1Image = document.getElementById('top1Image');
const top1Product = document.getElementById('top1Product');
const top1Defect = document.getElementById('top1Defect');
const top1Similarity = document.getElementById('top1Similarity');
const segmentationImage = document.getElementById('segmentationImage');
const anomalyScoreDisplay = document.getElementById('anomalyScoreDisplay');
const judgmentDisplay = document.getElementById('judgmentDisplay');
const modelButtons = document.querySelectorAll('.model-btn');
const generationProgress = document.getElementById('generationProgress');
const progressText = document.getElementById('progressText');
const manualResponse = document.getElementById('manualResponse');
const selectedModelBadge = document.getElementById('selectedModelBadge');
const processingTime = document.getElementById('processingTime');
const responseContent = document.getElementById('responseContent');
const workerInputSection = document.getElementById('workerInputSection');
const workerName = document.getElementById('workerName');
const actionTaken = document.getElementById('actionTaken');
const ratingButtons = document.querySelectorAll('.rating-btn');
const submitBtn = document.getElementById('submitBtn');
const completionSection = document.getElementById('completionSection');
const newWorkflowBtn = document.getElementById('newWorkflowBtn');
const viewHistoryBtn = document.getElementById('viewHistoryBtn');

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
    console.log('[MANUAL] 페이지 로드 완료');
    
    // 세션에서 데이터 복원
    restoreSessionData();
    
    // 이벤트 리스너 등록
    initEventListeners();
});

/**
 * 이벤트 리스너 초기화
 */
function initEventListeners() {
    // 모델 선택 버튼
    modelButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const model = btn.dataset.model;
            selectModel(model);
        });
    });
    
    // 평점 버튼
    ratingButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const score = parseInt(btn.dataset.score);
            selectRating(score);
        });
    });
    
    // 등록 버튼
    submitBtn.addEventListener('click', submitAction);
    
    // 새 검사 시작
    newWorkflowBtn.addEventListener('click', () => {
        SessionData.startNewWorkflow();
        window.location.href = '/upload.html';
    });
    
    // 이력 조회 (TODO: 구현 예정)
    viewHistoryBtn.addEventListener('click', () => {
        showNotification('이력 조회 기능은 추후 구현 예정입니다', 'info');
    });
}

/**
 * 세션 데이터 복원
 */
function restoreSessionData() {
    // 업로드 이미지
    uploadedImageData = SessionData.get('uploadedImage');
    
    // 유사도 매칭 결과
    const searchResults = SessionData.get('searchResults');
    if (searchResults && searchResults.top1) {
        selectedMatchData = searchResults.top1;
    }
    
    // 이상 검출 결과
    anomalyResultData = SessionData.get('anomalyResult');
    
    // 데이터 검증
    if (!uploadedImageData || !selectedMatchData || !anomalyResultData) {
        console.warn('[MANUAL] 필요한 데이터가 없습니다');
        showNotification('이전 단계를 먼저 완료해주세요', 'warning');
        setTimeout(() => {
            if (!uploadedImageData) {
                window.location.href = '/upload.html';
            } else if (!selectedMatchData) {
                window.location.href = '/search.html';
            } else {
                window.location.href = '/anomaly.html';
            }
        }, 2000);
        return;
    }
    
    console.log('[MANUAL] 데이터 복원 완료');
    displayImages();
}

/**
 * 이미지 표시
 */
function displayImages() {
    // TOP-1 불량 이미지 (유사도 매칭에서 선택된 것)
    if (anomalyResultData.top1_defect_image) {
        top1Image.src = `/api/image/${anomalyResultData.top1_defect_image}`;
    } else {
        top1Image.src = `/api/image/${selectedMatchData.image_path}`;
    }
    
    top1Product.textContent = anomalyResultData.product || selectedMatchData.product;
    top1Defect.textContent = anomalyResultData.defect || selectedMatchData.defect;
    
    if (anomalyResultData.similarity !== undefined) {
        top1Similarity.textContent = `${(anomalyResultData.similarity * 100).toFixed(1)}%`;
    } else if (selectedMatchData.similarity_score !== undefined) {
        top1Similarity.textContent = `${(selectedMatchData.similarity_score * 100).toFixed(1)}%`;
    }
    
    // Segmentation 이미지 (이상 검출 결과)
    segmentationImage.src = anomalyResultData.overlay_url;
    anomalyScoreDisplay.textContent = anomalyResultData.image_score.toFixed(4);
    
    if (anomalyResultData.is_anomaly) {
        judgmentDisplay.textContent = '⚠️ 이상 (Anomaly)';
        judgmentDisplay.style.color = 'var(--danger-color)';
    } else {
        judgmentDisplay.textContent = '✅ 정상 (Normal)';
        judgmentDisplay.style.color = 'var(--success-color)';
    }
}

/**
 * 모델 선택
 */
function selectModel(model) {
    selectedModel = model;
    
    // UI 업데이트
    modelButtons.forEach(btn => {
        if (btn.dataset.model === model) {
            btn.classList.add('selected');
        } else {
            btn.classList.remove('selected');
        }
    });
    
    console.log('[MANUAL] 모델 선택:', model);
    
    // 매뉴얼 생성
    generateManual();
}

/**
 * 매뉴얼 생성
 */
async function generateManual() {
    if (!selectedModel) {
        showNotification('모델을 먼저 선택해주세요', 'warning');
        return;
    }
    
    console.log('[MANUAL] 매뉴얼 생성 시작:', selectedModel);
    
    try {
        // UI 상태 변경
        generationProgress.style.display = 'block';
        manualResponse.style.display = 'none';
        workerInputSection.style.display = 'none';
        
        progressText.textContent = `${getModelDisplayName(selectedModel)} 모델로 분석 중...`;
        
        // 매뉴얼 생성 요청
        //const response = await fetch(`${API_BASE_URL}/manual/generate`, {
        // ✅ 상대 경로로 수정
        const response = await fetch('/manual/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                product: anomalyResultData.product,
                defect: anomalyResultData.defect,
                anomaly_score: anomalyResultData.image_score,
                is_anomaly: anomalyResultData.is_anomaly,
                model_type: selectedModel,
                image_path: uploadedImageData.file_path
            })
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || '매뉴얼 생성 실패');
        }
        
        const data = await response.json();
        console.log('[MANUAL] 매뉴얼 생성 완료:', data);
        
        // 결과 저장
        generatedManual = data;
        
        // 결과 표시
        displayManual(data);
        
        // 작업자 입력 섹션 표시
        workerInputSection.style.display = 'block';
        
        showNotification('매뉴얼 생성 완료', 'success');
        
    } catch (error) {
        console.error('[MANUAL] 매뉴얼 생성 실패:', error);
        showNotification(`매뉴얼 생성 실패: ${error.message}`, 'error');
    } finally {
        generationProgress.style.display = 'none';
    }
}

/**
 * 매뉴얼 표시
 */
function displayManual(data) {
    // 모델 배지 업데이트
    selectedModelBadge.textContent = getModelDisplayName(data.model_type);
    processingTime.textContent = `${data.processing_time}초`;
    
    // AI 답변 파싱 및 표시
    const sections = parseManualResponse(data.llm_analysis);
    
    responseContent.innerHTML = `
        ${sections.status ? `
        <div class="response-section">
            <h4>📌 불량 현황</h4>
            <p>${sections.status}</p>
        </div>
        ` : ''}
        
        ${sections.cause ? `
        <div class="response-section">
            <h4>🔍 원인 분석</h4>
            <p>${sections.cause}</p>
        </div>
        ` : ''}
        
        ${sections.action ? `
        <div class="response-section">
            <h4>⚙️ 대응 방안</h4>
            <p>${sections.action}</p>
        </div>
        ` : ''}
        
        ${sections.prevention ? `
        <div class="response-section">
            <h4>🛡️ 예방 조치</h4>
            <p>${sections.prevention}</p>
        </div>
        ` : ''}
        
        ${!sections.status && !sections.cause && !sections.action && !sections.prevention ? `
        <div class="response-section">
            <p>${data.llm_analysis}</p>
        </div>
        ` : ''}
    `;
    
    // 답변 섹션 표시
    manualResponse.style.display = 'block';
    
    // 스크롤
    manualResponse.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/**
 * 매뉴얼 응답 파싱 (4개 섹션)
 */
function parseManualResponse(text) {
    const sections = {
        status: '',
        cause: '',
        action: '',
        prevention: ''
    };
    
    // 정규표현식으로 섹션 추출
    const statusMatch = text.match(/(?:1\.|【불량 현황】|##\s*불량\s*현황)([\s\S]*?)(?=(?:2\.|【원인 분석】|##\s*원인\s*분석)|$)/i);
    const causeMatch = text.match(/(?:2\.|【원인 분석】|##\s*원인\s*분석)([\s\S]*?)(?=(?:3\.|【대응 방안】|##\s*대응\s*방안)|$)/i);
    const actionMatch = text.match(/(?:3\.|【대응 방안】|##\s*대응\s*방안)([\s\S]*?)(?=(?:4\.|【예방 조치】|##\s*예방\s*조치)|$)/i);
    const preventionMatch = text.match(/(?:4\.|【예방 조치】|##\s*예방\s*조치)([\s\S]*?)$/i);
    
    if (statusMatch) sections.status = statusMatch[1].trim();
    if (causeMatch) sections.cause = causeMatch[1].trim();
    if (actionMatch) sections.action = actionMatch[1].trim();
    if (preventionMatch) sections.prevention = preventionMatch[1].trim();
    
    return sections;
}

/**
 * 모델 표시명
 */
function getModelDisplayName(model) {
    const names = {
        'hyperclovax': 'HyperCLOVAX',
        'exaone': 'EXAONE 3.5',
        'llava': 'LLaVA (VLM)'
    };
    return names[model] || model;
}

/**
 * 평점 선택
 */
function selectRating(score) {
    selectedRating = score;
    
    // UI 업데이트
    ratingButtons.forEach(btn => {
        if (parseInt(btn.dataset.score) === score) {
            btn.classList.add('selected');
        } else {
            btn.classList.remove('selected');
        }
    });
    
    console.log('[MANUAL] 평점 선택:', score);
}

/**
 * 조치 내역 등록
 */
async function submitAction() {
    // 입력 검증
    const worker = workerName.value.trim();
    const action = actionTaken.value.trim();
    
    if (!worker) {
        showNotification('작업자명을 입력해주세요', 'warning');
        workerName.focus();
        return;
    }
    
    if (!action) {
        showNotification('조치 내역을 입력해주세요', 'warning');
        actionTaken.focus();
        return;
    }
    
    if (!selectedRating) {
        showNotification('만족도를 선택해주세요', 'warning');
        return;
    }
    
    if (!generatedManual) {
        showNotification('매뉴얼을 먼저 생성해주세요', 'warning');
        return;
    }
    
    console.log('[MANUAL] 조치 내역 등록 시작');
    
    try {
        submitBtn.disabled = true;
        submitBtn.textContent = '등록 중...';
        
        // TODO: 실제 DB 저장 API 호출
        // const response = await fetch(`${API_BASE_URL}/history/save`, {
        //     method: 'POST',
        //     headers: { 'Content-Type': 'application/json' },
        //     body: JSON.stringify({
        //         product_name: anomalyResultData.product,
        //         defect_name: anomalyResultData.defect,
        //         input_image_path: uploadedImageData.file_path,
        //         top1_image_path: selectedMatchData.image_path,
        //         model_used: selectedModel,
        //         llm_response: generatedManual.llm_analysis,
        //         processing_time: generatedManual.processing_time,
        //         worker_name: worker,
        //         action_taken: action,
        //         feedback_score: selectedRating,
        //         anomaly_score: anomalyResultData.image_score,
        //         is_anomaly: anomalyResultData.is_anomaly
        //     })
        // });
        
        // 임시: 로컬 저장
        const historyData = {
            timestamp: new Date().toISOString(),
            product_name: anomalyResultData.product,
            defect_name: anomalyResultData.defect,
            input_image: uploadedImageData.filename,
            model_used: selectedModel,
            worker_name: worker,
            action_taken: action,
            feedback_score: selectedRating,
            anomaly_score: anomalyResultData.image_score,
            is_anomaly: anomalyResultData.is_anomaly
        };
        
        console.log('[MANUAL] 등록 데이터:', historyData);
        
        // 성공 시 완료 화면 표시
        workerInputSection.style.display = 'none';
        completionSection.style.display = 'block';
        
        showNotification('조치 내역이 등록되었습니다', 'success');
        
        // 스크롤
        completionSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        
    } catch (error) {
        console.error('[MANUAL] 등록 실패:', error);
        showNotification(`등록 실패: ${error.message}`, 'error');
    } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = '💾 조치 내역 등록';
    }
}
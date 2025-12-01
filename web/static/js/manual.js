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
let serverConfig = null;  // ✅ 추가: 서버 설정

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
    console.log('[MANUAL] 페이지 로드 완료');
    
    // ✅ 추가: 서버 설정 로드
    loadServerConfig();
    
    // 세션에서 데이터 복원
    restoreSessionData();
    
    // 이벤트 리스너 등록
    initEventListeners();
});

/**
 * ✅ 추가: 서버 설정 로드
 */
async function loadServerConfig() {
    try {
        const response = await fetch('/manual/server-config');
        const data = await response.json();
        
        serverConfig = data;
        console.log('[MANUAL] 서버 설정 로드:', serverConfig);
        
        // VLM 비활성화 처리
        if (!serverConfig.vlm_enabled) {
            const llavaBtn = document.querySelector('[data-model="llava"]');
            if (llavaBtn) {
                llavaBtn.disabled = true;
                llavaBtn.style.opacity = '0.5';
                llavaBtn.style.cursor = 'not-allowed';
                llavaBtn.title = 'CPU 서버에서는 VLM 모델을 사용할 수 없습니다';
                
                // 버튼에 비활성화 표시 추가
                const badge = document.createElement('span');
                badge.className = 'badge badge-warning';
                badge.textContent = '사용 불가';
                badge.style.marginLeft = '8px';
                badge.style.fontSize = '0.75rem';
                llavaBtn.appendChild(badge);
            }
            console.log('[MANUAL] VLM 버튼 비활성화 (CPU 서버)');
        }
        
    } catch (error) {
        console.error('[MANUAL] 서버 설정 로드 실패:', error);
        // 기본값 사용
        serverConfig = {
            is_cpu_server: false,
            vlm_enabled: true,
            timeout: 120
        };
    }
}

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
      // ✅ response_id 확인
    console.log('[MANUAL] 데이터 복원 완료');
    console.log('[MANUAL] response_id:', anomalyResultData.response_id);
    console.log('[MANUAL] search_id:', anomalyResultData.search_id);

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
    // ✅ 추가: VLM 선택 차단
    if (model === 'llava' && serverConfig && !serverConfig.vlm_enabled) {
        showNotification(
            'CPU 서버에서는 VLM(LLaVA) 모델을 사용할 수 없습니다. ' +
            'HyperCLOVAX 또는 EXAONE 모델을 선택해주세요.',
            'warning'
        );
        return;
    }
    
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
    console.log('[MANUAL] response_id:', anomalyResultData.response_id);
    
    try {
        // UI 상태 변경
        generationProgress.style.display = 'block';
        manualResponse.style.display = 'none';
        workerInputSection.style.display = 'none';
        
        progressText.textContent = `${getModelDisplayName(selectedModel)} 모델로 분석 중...`;
        
         
        // ✅ 추가: AbortController로 timeout 구현
        // 서버 timeout + 30초 여유분 (서버: 300초 → 클라이언트: 330초)
        const timeoutMs = (serverConfig?.timeout || 120) * 1000 + 30000;
        console.log(`[MANUAL] Fetch timeout: ${timeoutMs / 1000}초`);
        
        const controller = new AbortController();
        const timeoutId = setTimeout(() => {
            controller.abort();
            console.error('[MANUAL] Timeout 발생');
        }, timeoutMs)

        try {
             // 매뉴얼 생성 요청
            const response = await fetch('/manual/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    product: anomalyResultData.product_code,
                    defect: anomalyResultData.defect_code,
                    anomaly_score: anomalyResultData.image_score,
                    is_anomaly: anomalyResultData.is_anomaly,
                    model_type: selectedModel,
                    image_path: uploadedImageData.file_path,
                    response_id: anomalyResultData.response_id
                }),
                signal: controller.signal  // ✅ AbortSignal 추가
            });


                
            clearTimeout(timeoutId);  // ✅ timeout 타이머 해제
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || '매뉴얼 생성 실패');
            }
            
            const data = await response.json();
            console.log('[MANUAL] 매뉴얼 생성 완료:', data);
            console.log('[MANUAL] response_id 확인:', data.response_id);
            
            // 결과 저장
            generatedManual = data;
            
            // 결과 표시
            displayManual(data);
            
            // 작업자 입력 섹션 표시
            workerInputSection.style.display = 'block';
            
            showNotification('매뉴얼 생성 완료', 'success');
        }catch (fetchError) {
            clearTimeout(timeoutId);
            
            // ✅ Timeout 에러와 네트워크 에러 구분
            if (fetchError.name === 'AbortError') {
                throw new Error(`요청 시간 초과 (${timeoutMs / 1000}초). CPU 서버에서 모델 추론이 지연되고 있습니다.`);
            } else {
                throw fetchError;
            }
        }
       
        
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
        
        // 피드백 등록 API 호출
        const response = await fetch('/manual/feedback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                response_id: generatedManual.response_id,
                feedback_user: worker,
                feedback_rating: selectedRating,
                feedback_text: action
            })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || '피드백 등록 실패');
        }
        

        
        // 성공 시 완료 화면 표시
        workerInputSection.style.display = 'none';
        completionSection.style.display = 'block';
        
        const data = await response.json();
        console.log('[MANUAL] 등록 데이터:', data);

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
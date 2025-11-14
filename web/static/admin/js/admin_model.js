// 모델 선택 JavaScript

let availableModels = {
    clip: [],
    patchcore: []
};

let currentModels = {
    clip: null,
    patchcore: null
};

let selectedModel = null;

// 페이지 로드 시 초기화
document.addEventListener('DOMContentLoaded', function() {
    loadAvailableModels();
    loadCurrentModels();
});

/**
 * 사용 가능한 모델 목록 로드
 */
async function loadAvailableModels() {
    try {
        const response = await fetch('/api/admin/models/available');
        
        if (!response.ok) {
            throw new Error('모델 목록 로드 실패');
        }
        
        const data = await response.json();
        availableModels = data.models;
        
        displayModels('clip', availableModels.clip);
        displayModels('patchcore', availableModels.patchcore);
        displayComparisonTables();
        
    } catch (error) {
        console.error('Error loading available models:', error);
        showAlert('error', '모델 목록 로드 중 오류가 발생했습니다.');
    }
}

/**
 * 현재 선택된 모델 로드
 */
async function loadCurrentModels() {
    try {
        const response = await fetch('/api/admin/models/current');
        
        if (!response.ok) {
            throw new Error('현재 모델 조회 실패');
        }
        
        currentModels = await response.json();
        
        displayCurrentModel('clip', currentModels.clip);
        displayCurrentModel('patchcore', currentModels.patchcore);
        
    } catch (error) {
        console.error('Error loading current models:', error);
    }
}

/**
 * 현재 모델 표시
 */
function displayCurrentModel(modelType, modelData) {
    const containerId = modelType === 'clip' ? 'currentClip' : 'currentPatchcore';
    const container = document.getElementById(containerId);
    
    if (!modelData) {
        container.innerHTML = '<div class="empty-message">선택된 모델이 없습니다.</div>';
        return;
    }
    
    const params = modelData.params || {};
    const modelId = params.model_id || 'Unknown';
    
    const html = `
        <div class="current-model-info">
            <div class="current-model-name">
                ${modelId}
                <span class="active-badge">활성</span>
            </div>
            <div class="current-model-detail">제품: ${modelData.product_name || '전체'}</div>
            ${params.description ? `<div class="current-model-detail">${params.description}</div>` : ''}
        </div>
    `;
    
    container.innerHTML = html;
}

/**
 * 모델 카드 표시
 */
function displayModels(modelType, models) {
    const containerId = modelType === 'clip' ? 'clipModels' : 'patchcoreModels';
    const container = document.getElementById(containerId);
    
    if (!models || models.length === 0) {
        container.innerHTML = '<div class="empty-message">사용 가능한 모델이 없습니다.</div>';
        return;
    }
    
    const html = models.map(model => {
        const isSelected = currentModels[modelType]?.params?.model_id === model.model_id;
        const selectedClass = isSelected ? 'selected' : '';
        const buttonText = isSelected ? '사용 중' : '선택';
        const buttonClass = isSelected ? 'btn-success' : 'btn-primary';
        const buttonDisabled = isSelected ? 'disabled' : '';
        
        return `
            <div class="model-card ${selectedClass}">
                <div class="model-card-header">
                    <div class="model-name">${model.name}</div>
                </div>
                <div class="model-description">${model.description}</div>
                <div class="model-specs">
                    <div class="model-spec-item">
                        <span class="model-spec-label">파라미터 수:</span>
                        <span class="model-spec-value">${formatNumber(model.parameters)}</span>
                    </div>
                    <div class="model-spec-item">
                        <span class="model-spec-label">입력 크기:</span>
                        <span class="model-spec-value">${model.input_size}x${model.input_size}</span>
                    </div>
                </div>
                <div class="model-rating">
                    <div class="rating-item">
                        <div class="rating-label">성능</div>
                        <div class="rating-value">${model.performance}</div>
                    </div>
                    <div class="rating-item">
                        <div class="rating-label">속도</div>
                        <div class="rating-value">${model.speed}</div>
                    </div>
                </div>
                <div class="model-actions">
                    <button class="btn ${buttonClass}" 
                            onclick="selectModel('${modelType}', '${model.model_id}')"
                            ${buttonDisabled}>
                        ${buttonText}
                    </button>
                </div>
            </div>
        `;
    }).join('');
    
    container.innerHTML = html;
}

/**
 * 비교 테이블 표시
 */
function displayComparisonTables() {
    // CLIP 비교 테이블
    const clipTable = document.getElementById('clipComparisonTable');
    const clipRows = availableModels.clip.map(model => `
        <tr>
            <td><strong>${model.name}</strong></td>
            <td>${formatNumber(model.parameters)}</td>
            <td>${model.input_size}x${model.input_size}</td>
            <td>${model.performance}</td>
            <td>${model.speed}</td>
            <td>${getRecommendation(model.model_id, 'clip')}</td>
        </tr>
    `).join('');
    clipTable.innerHTML = clipRows;
    
    // PatchCore 비교 테이블
    const patchcoreTable = document.getElementById('patchcoreComparisonTable');
    const patchcoreRows = availableModels.patchcore.map(model => `
        <tr>
            <td><strong>${model.name}</strong></td>
            <td>${formatNumber(model.parameters)}</td>
            <td>${model.input_size}x${model.input_size}</td>
            <td>${model.performance}</td>
            <td>${model.speed}</td>
            <td>${getRecommendation(model.model_id, 'patchcore')}</td>
        </tr>
    `).join('');
    patchcoreTable.innerHTML = patchcoreRows;
}

/**
 * 추천 용도 반환
 */
function getRecommendation(modelId, modelType) {
    const recommendations = {
        'clip': {
            'ViT-B-32': '<span class="recommended">권장</span> 일반적인 사용',
            'ViT-B-16': '높은 정확도가 필요한 경우',
            'ViT-L-14': '최고 성능이 필요한 경우'
        },
        'patchcore': {
            'wide_resnet50_2': '<span class="recommended">권장</span> 대부분의 경우',
            'resnet18': '빠른 속도가 필요한 경우'
        }
    };
    
    return recommendations[modelType]?.[modelId] || '-';
}

/**
 * 모델 선택
 */
async function selectModel(modelType, modelId) {
    // 모델 정보 찾기
    const model = availableModels[modelType].find(m => m.model_id === modelId);
    if (!model) {
        showAlert('error', '모델 정보를 찾을 수 없습니다.');
        return;
    }
    
    selectedModel = {
        type: modelType,
        id: modelId,
        info: model
    };
    
    // 확인 모달 표시
    showSelectModal();
}

/**
 * 선택 확인 모달 표시
 */
function showSelectModal() {
    const modalBody = document.getElementById('modalBody');
    const model = selectedModel.info;
    
    const html = `
        <p style="margin-bottom: 16px;">다음 모델을 선택하시겠습니까?</p>
        <div style="background-color: #f8f9fa; padding: 16px; border-radius: 6px;">
            <div style="font-size: 16px; font-weight: 600; margin-bottom: 8px;">${model.name}</div>
            <div style="font-size: 14px; color: #666; margin-bottom: 12px;">${model.description}</div>
            <div style="font-size: 13px; color: #555;">
                <div>• 파라미터 수: ${formatNumber(model.parameters)}</div>
                <div>• 입력 크기: ${model.input_size}x${model.input_size}</div>
                <div>• 성능: ${model.performance}</div>
                <div>• 속도: ${model.speed}</div>
            </div>
        </div>
        <p style="margin-top: 16px; font-size: 13px; color: #666;">
            모델 변경 후 CLIP 또는 PatchCore 재구축이 필요할 수 있습니다.
        </p>
    `;
    
    modalBody.innerHTML = html;
    
    const modal = document.getElementById('selectModal');
    modal.classList.add('active');
}

/**
 * 모달 닫기
 */
function closeModal() {
    const modal = document.getElementById('selectModal');
    modal.classList.remove('active');
    selectedModel = null;
}

/**
 * 선택 확인
 */
async function confirmSelection() {
    if (!selectedModel) {
        return;
    }
    
    try {
        // 제품 ID는 1로 고정 (전체 적용)
        const response = await fetch('/api/admin/models/select', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                product_id: 1,
                model_type: selectedModel.type,
                params: {
                    model_id: selectedModel.id,
                    description: selectedModel.info.description
                }
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || '모델 선택 실패');
        }
        
        const data = await response.json();
        
        // 활성화
        const activateResponse = await fetch(`/api/admin/models/${data.param_id}/activate`, {
            method: 'POST'
        });
        
        if (!activateResponse.ok) {
            throw new Error('모델 활성화 실패');
        }
        
        showAlert('success', `${selectedModel.info.name} 모델이 선택되었습니다.`);
        closeModal();
        
        // 새로고침
        loadCurrentModels();
        loadAvailableModels();
        
    } catch (error) {
        console.error('Error selecting model:', error);
        showAlert('error', '모델 선택 중 오류가 발생했습니다: ' + error.message);
    }
}

/**
 * 숫자 포맷팅
 */
function formatNumber(num) {
    if (num >= 1000000) {
        return (num / 1000000).toFixed(1) + 'M';
    } else if (num >= 1000) {
        return (num / 1000).toFixed(1) + 'K';
    }
    return num.toString();
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
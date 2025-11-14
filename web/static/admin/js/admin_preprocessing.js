// 이미지 전처리 설정 JavaScript

let presets = [];
let configs = [];

// 페이지 로드 시 초기화
document.addEventListener('DOMContentLoaded', function() {
    loadPresets();
    loadActiveConfig();
    loadConfigList();
});

/**
 * 프리셋 로드
 */
async function loadPresets() {
    try {
        const response = await fetch('/api/admin/preprocessing/presets/default');
        
        if (!response.ok) {
            throw new Error('프리셋 로드 실패');
        }
        
        const data = await response.json();
        presets = data.presets;
        
        // 프리셋 선택 옵션 추가
        const select = document.getElementById('presetSelect');
        presets.forEach((preset, index) => {
            const option = document.createElement('option');
            option.value = index;
            option.textContent = preset.name;
            select.appendChild(option);
        });
        
    } catch (error) {
        console.error('Error loading presets:', error);
    }
}

/**
 * 프리셋 적용
 */
function applyPreset() {
    const select = document.getElementById('presetSelect');
    const selectedIndex = select.value;
    
    if (selectedIndex === '') {
        return;
    }
    
    const preset = presets[selectedIndex];
    
    // 폼 필드에 프리셋 값 적용
    document.getElementById('configName').value = preset.name;
    document.getElementById('resizeWidth').value = preset.resize_width;
    document.getElementById('resizeHeight').value = preset.resize_height;
    document.getElementById('normalize').checked = preset.normalize;
    
    if (preset.augmentation) {
        document.getElementById('enableAugmentation').checked = true;
        toggleAugmentation();
        
        document.getElementById('rotation').value = preset.augmentation.rotation || 15;
        document.getElementById('flipHorizontal').checked = preset.augmentation.flip_horizontal || false;
        document.getElementById('brightness').value = preset.augmentation.brightness || 0.2;
        document.getElementById('contrast').value = preset.augmentation.contrast || 0.2;
    } else {
        document.getElementById('enableAugmentation').checked = false;
        toggleAugmentation();
    }
}

/**
 * 증강 옵션 토글
 */
function toggleAugmentation() {
    const enabled = document.getElementById('enableAugmentation').checked;
    const options = document.getElementById('augmentationOptions');
    
    if (enabled) {
        options.style.display = 'block';
    } else {
        options.style.display = 'none';
    }
}

/**
 * 활성 설정 로드
 */
async function loadActiveConfig() {
    try {
        const response = await fetch('/api/admin/preprocessing/active');
        
        if (response.status === 404) {
            // 활성화된 설정 없음
            document.getElementById('activeConfigSection').innerHTML = 
                '<div class="empty-message">활성화된 설정이 없습니다.</div>';
            return;
        }
        
        if (!response.ok) {
            throw new Error('활성 설정 조회 실패');
        }
        
        const config = await response.json();
        displayActiveConfig(config);
        
    } catch (error) {
        console.error('Error loading active config:', error);
    }
}

/**
 * 활성 설정 표시
 */
function displayActiveConfig(config) {
    const section = document.getElementById('activeConfigSection');
    
    const augmentationText = config.augmentation 
        ? '활성화' 
        : '비활성화';
    
    const html = `
        <div class="active-config">
            <div class="active-config-header">
                <div class="active-config-name">${config.name}</div>
                <span class="active-badge">활성</span>
            </div>
            <div class="active-config-details">
                <div class="config-detail-item">
                    <span class="config-detail-label">이미지 크기:</span>
                    <span class="config-detail-value">${config.resize_width}x${config.resize_height}</span>
                </div>
                <div class="config-detail-item">
                    <span class="config-detail-label">정규화:</span>
                    <span class="config-detail-value">${config.normalize ? '활성화' : '비활성화'}</span>
                </div>
                <div class="config-detail-item">
                    <span class="config-detail-label">데이터 증강:</span>
                    <span class="config-detail-value">${augmentationText}</span>
                </div>
                <div class="config-detail-item">
                    <span class="config-detail-label">생성일:</span>
                    <span class="config-detail-value">${formatDate(config.created_at)}</span>
                </div>
            </div>
        </div>
    `;
    
    section.innerHTML = html;
}

/**
 * 설정 목록 로드
 */
async function loadConfigList() {
    try {
        const response = await fetch('/api/admin/preprocessing/');
        
        if (!response.ok) {
            throw new Error('설정 목록 조회 실패');
        }
        
        configs = await response.json();
        displayConfigList(configs);
        
    } catch (error) {
        console.error('Error loading config list:', error);
    }
}

/**
 * 설정 목록 표시
 */
function displayConfigList(configs) {
    const listElement = document.getElementById('configList');
    
    if (!configs || configs.length === 0) {
        listElement.innerHTML = '<div class="empty-message">저장된 설정이 없습니다.</div>';
        return;
    }
    
    const html = configs.map(config => {
        const activeClass = config.is_active ? 'active' : '';
        const activeBadge = config.is_active 
            ? '<span class="active-badge">활성</span>' 
            : '';
        
        const augmentationText = config.augmentation 
            ? '활성화' 
            : '비활성화';
        
        return `
            <div class="config-item ${activeClass}">
                <div class="config-item-header">
                    <div class="config-item-name">
                        ${config.name}
                        ${activeBadge}
                    </div>
                    <div class="config-item-actions">
                        ${!config.is_active ? `<button class="btn btn-success btn-sm" onclick="activateConfig(${config.id})">활성화</button>` : ''}
                        <button class="btn btn-secondary btn-sm" onclick="showConfigDetail(${config.id})">상세</button>
                        ${!config.is_active ? `<button class="btn btn-danger btn-sm" onclick="deleteConfig(${config.id})">삭제</button>` : ''}
                    </div>
                </div>
                <div class="config-item-details">
                    <div class="config-item-detail">
                        <div class="config-item-detail-label">이미지 크기</div>
                        <div class="config-item-detail-value">${config.resize_width}x${config.resize_height}</div>
                    </div>
                    <div class="config-item-detail">
                        <div class="config-item-detail-label">정규화</div>
                        <div class="config-item-detail-value">${config.normalize ? '활성화' : '비활성화'}</div>
                    </div>
                    <div class="config-item-detail">
                        <div class="config-item-detail-label">데이터 증강</div>
                        <div class="config-item-detail-value">${augmentationText}</div>
                    </div>
                    <div class="config-item-detail">
                        <div class="config-item-detail-label">생성일</div>
                        <div class="config-item-detail-value">${formatDate(config.created_at)}</div>
                    </div>
                </div>
            </div>
        `;
    }).join('');
    
    listElement.innerHTML = html;
}

/**
 * 설정 저장
 */
async function saveConfig(event) {
    event.preventDefault();
    
    const name = document.getElementById('configName').value;
    const resizeWidth = parseInt(document.getElementById('resizeWidth').value);
    const resizeHeight = parseInt(document.getElementById('resizeHeight').value);
    const normalize = document.getElementById('normalize').checked;
    
    let augmentation = null;
    if (document.getElementById('enableAugmentation').checked) {
        augmentation = {
            rotation: parseInt(document.getElementById('rotation').value),
            flip_horizontal: document.getElementById('flipHorizontal').checked,
            brightness: parseFloat(document.getElementById('brightness').value),
            contrast: parseFloat(document.getElementById('contrast').value)
        };
    }
    
    try {
        const response = await fetch('/api/admin/preprocessing/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                name,
                resize_width: resizeWidth,
                resize_height: resizeHeight,
                normalize,
                augmentation
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || '설정 저장 실패');
        }
        
        showAlert('success', '설정이 저장되었습니다.');
        resetForm();
        loadConfigList();
        
    } catch (error) {
        console.error('Error saving config:', error);
        showAlert('error', '설정 저장 중 오류가 발생했습니다: ' + error.message);
    }
}

/**
 * 설정 활성화
 */
async function activateConfig(configId) {
    if (!confirm('이 설정을 활성화하시겠습니까?')) {
        return;
    }
    
    try {
        const response = await fetch(`/api/admin/preprocessing/${configId}/activate`, {
            method: 'POST'
        });
        
        if (!response.ok) {
            throw new Error('설정 활성화 실패');
        }
        
        showAlert('success', '설정이 활성화되었습니다.');
        loadActiveConfig();
        loadConfigList();
        
    } catch (error) {
        console.error('Error activating config:', error);
        showAlert('error', '설정 활성화 중 오류가 발생했습니다: ' + error.message);
    }
}

/**
 * 설정 삭제
 */
async function deleteConfig(configId) {
    if (!confirm('이 설정을 삭제하시겠습니까?')) {
        return;
    }
    
    try {
        const response = await fetch(`/api/admin/preprocessing/${configId}`, {
            method: 'DELETE'
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || '설정 삭제 실패');
        }
        
        showAlert('success', '설정이 삭제되었습니다.');
        loadConfigList();
        
    } catch (error) {
        console.error('Error deleting config:', error);
        showAlert('error', '설정 삭제 중 오류가 발생했습니다: ' + error.message);
    }
}

/**
 * 설정 상세 보기
 */
async function showConfigDetail(configId) {
    try {
        const response = await fetch(`/api/admin/preprocessing/${configId}`);
        
        if (!response.ok) {
            throw new Error('설정 조회 실패');
        }
        
        const config = await response.json();
        displayConfigDetailModal(config);
        
    } catch (error) {
        console.error('Error loading config detail:', error);
        showAlert('error', '설정 조회 중 오류가 발생했습니다: ' + error.message);
    }
}

/**
 * 설정 상세 모달 표시
 */
function displayConfigDetailModal(config) {
    const modalBody = document.getElementById('modalBody');
    
    let augmentationHtml = '<div class="detail-item-value">비활성화</div>';
    if (config.augmentation) {
        augmentationHtml = `
            <div class="detail-grid">
                <div class="detail-item">
                    <span class="detail-item-label">회전 각도:</span>
                    <span class="detail-item-value">±${config.augmentation.rotation}°</span>
                </div>
                <div class="detail-item">
                    <span class="detail-item-label">좌우 반전:</span>
                    <span class="detail-item-value">${config.augmentation.flip_horizontal ? '활성화' : '비활성화'}</span>
                </div>
                <div class="detail-item">
                    <span class="detail-item-label">밝기 조정:</span>
                    <span class="detail-item-value">±${config.augmentation.brightness}</span>
                </div>
                <div class="detail-item">
                    <span class="detail-item-label">대비 조정:</span>
                    <span class="detail-item-value">±${config.augmentation.contrast}</span>
                </div>
            </div>
        `;
    }
    
    const html = `
        <div class="detail-section">
            <h4>기본 설정</h4>
            <div class="detail-grid">
                <div class="detail-item">
                    <span class="detail-item-label">설정 이름:</span>
                    <span class="detail-item-value">${config.name}</span>
                </div>
                <div class="detail-item">
                    <span class="detail-item-label">상태:</span>
                    <span class="detail-item-value">${config.is_active ? '활성' : '비활성'}</span>
                </div>
                <div class="detail-item">
                    <span class="detail-item-label">이미지 너비:</span>
                    <span class="detail-item-value">${config.resize_width}px</span>
                </div>
                <div class="detail-item">
                    <span class="detail-item-label">이미지 높이:</span>
                    <span class="detail-item-value">${config.resize_height}px</span>
                </div>
                <div class="detail-item">
                    <span class="detail-item-label">정규화:</span>
                    <span class="detail-item-value">${config.normalize ? '활성화' : '비활성화'}</span>
                </div>
                <div class="detail-item">
                    <span class="detail-item-label">생성일:</span>
                    <span class="detail-item-value">${formatDate(config.created_at)}</span>
                </div>
            </div>
        </div>
        
        <div class="detail-section">
            <h4>데이터 증강 설정</h4>
            ${augmentationHtml}
        </div>
    `;
    
    modalBody.innerHTML = html;
    
    const modal = document.getElementById('configModal');
    modal.classList.add('active');
}

/**
 * 모달 닫기
 */
function closeModal() {
    const modal = document.getElementById('configModal');
    modal.classList.remove('active');
}

/**
 * 폼 초기화
 */
function resetForm() {
    document.getElementById('configForm').reset();
    document.getElementById('presetSelect').value = '';
    document.getElementById('resizeWidth').value = 224;
    document.getElementById('resizeHeight').value = 224;
    document.getElementById('normalize').checked = true;
    document.getElementById('enableAugmentation').checked = false;
    toggleAugmentation();
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

/**
 * 날짜 포맷팅
 */
function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString('ko-KR', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit'
    });
}
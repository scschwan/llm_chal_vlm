/**
 * 이미지 업로드 화면 스크립트
 */

// 전역 변수
let uploadedFileData = null;

// DOM 요소
const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const uploadButton = document.getElementById('uploadButton');
const uploadProgress = document.getElementById('uploadProgress');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');
const previewSection = document.getElementById('previewSection');
const previewImage = document.getElementById('previewImage');
const fileName = document.getElementById('fileName');
const fileSize = document.getElementById('fileSize');
const resolution = document.getElementById('resolution');
const filePath = document.getElementById('filePath');
const reuploadBtn = document.getElementById('reuploadBtn');
const nextBtn = document.getElementById('nextBtn');
const checkIndexBtn = document.getElementById('checkIndexBtn');
const rebuildIndexBtn = document.getElementById('rebuildIndexBtn');
const recentFilesList = document.getElementById('recentFilesList');

// 페이지 로드 시 초기화
document.addEventListener('DOMContentLoaded', () => {
    console.log('[UPLOAD] 페이지 로드 완료');
    
    // 이벤트 리스너 등록
    initEventListeners();
    
    // 인덱스 상태 확인
    checkIndexStatus();
    
    // 최근 파일 목록 로드
    loadRecentFiles();
    
    // 세션에서 이전 업로드 정보 복원
    restoreSessionData();
});

/**
 * 이벤트 리스너 초기화
 */
function initEventListeners() {
    // 업로드 버튼 클릭
    uploadButton.addEventListener('click', () => {
        fileInput.click();
    });
    
    // 업로드 존 클릭
    uploadZone.addEventListener('click', (e) => {
        if (e.target !== uploadButton) {
            fileInput.click();
        }
    });
    
    // 파일 선택
    fileInput.addEventListener('change', handleFileSelect);
    
    // 드래그 앤 드롭
    uploadZone.addEventListener('dragover', handleDragOver);
    uploadZone.addEventListener('dragleave', handleDragLeave);
    uploadZone.addEventListener('drop', handleDrop);
    
    // 다시 업로드
    reuploadBtn.addEventListener('click', () => {
        resetUpload();
    });
    
    // 다음 단계
    nextBtn.addEventListener('click', goToNextPage);
    
    // 인덱스 관리
    checkIndexBtn.addEventListener('click', checkIndexStatus);
    rebuildIndexBtn.addEventListener('click', rebuildIndex);
}

/**
 * 파일 선택 핸들러
 */
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        uploadFile(file);
    }
}

/**
 * 드래그 오버 핸들러
 */
function handleDragOver(event) {
    event.preventDefault();
    event.stopPropagation();
    uploadZone.classList.add('dragover');
}

/**
 * 드래그 리브 핸들러
 */
function handleDragLeave(event) {
    event.preventDefault();
    event.stopPropagation();
    uploadZone.classList.remove('dragover');
}

/**
 * 드롭 핸들러
 */
function handleDrop(event) {
    event.preventDefault();
    event.stopPropagation();
    uploadZone.classList.remove('dragover');
    
    const files = event.dataTransfer.files;
    if (files.length > 0) {
        uploadFile(files[0]);
    }
}

/**
 * 파일 업로드
 */
async function uploadFile(file) {
    console.log('[UPLOAD] 파일 업로드 시작:', file.name);
    
    // 파일 크기 검증 (10MB 제한)
    const maxSize = 10 * 1024 * 1024; // 10MB
    if (file.size > maxSize) {
        showNotification('파일 크기는 10MB 이하여야 합니다', 'error');
        return;
    }
    
    // 파일 형식 검증
    const allowedTypes = ['image/jpeg', 'image/png', 'image/webp'];
    if (!allowedTypes.includes(file.type)) {
        showNotification('JPG, PNG, WEBP 형식만 지원합니다', 'error');
        return;
    }
    
    try {
        // UI 상태 변경
        uploadZone.style.display = 'none';
        uploadProgress.style.display = 'block';
        progressFill.style.width = '0%';
        progressText.textContent = '업로드 중...';
        
        // FormData 생성
        const formData = new FormData();
        formData.append('file', file);
        
        // 업로드 요청
        const response = await fetch(`${API_BASE_URL}/upload/image`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`업로드 실패: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('[UPLOAD] 업로드 완료:', data);
        
        // 진행바 100%
        progressFill.style.width = '100%';
        progressText.textContent = '업로드 완료!';
        
        // 프리뷰 표시
        await showPreview(file, data);
        
        // 최근 파일 목록 새로고침
        loadRecentFiles();
        
        showNotification('파일 업로드 완료', 'success');
        
    } catch (error) {
        console.error('[UPLOAD] 업로드 실패:', error);
        showNotification(`업로드 실패: ${error.message}`, 'error');
        
        // UI 초기화
        resetUpload();
    }
}

/**
 * 프리뷰 표시
 */
async function showPreview(file, uploadData) {
    // 파일 데이터 저장
    uploadedFileData = uploadData;
    
    // 이미지 로드
    const reader = new FileReader();
    reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
            // 이미지 표시
            previewImage.src = e.target.result;
            
            // 정보 표시
            fileName.textContent = uploadData.filename;
            fileSize.textContent = formatFileSize(uploadData.file_size);
            resolution.textContent = `${img.width} × ${img.height}`;
            filePath.textContent = uploadData.file_path;
            
            // 세션 저장
            SessionData.set('uploadedImage', {
                filename: uploadData.filename,
                file_path: uploadData.file_path,
                file_size: uploadData.file_size,
                preview: e.target.result,
                resolution: `${img.width} × ${img.height}`
            });
            
            // UI 전환
            uploadProgress.style.display = 'none';
            previewSection.style.display = 'block';
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

/**
 * 업로드 초기화
 */
function resetUpload() {
    uploadZone.style.display = 'block';
    uploadProgress.style.display = 'none';
    previewSection.style.display = 'none';
    fileInput.value = '';
    uploadedFileData = null;
    progressFill.style.width = '0%';
}

/**
 * 세션 데이터 복원
 */
function restoreSessionData() {
    const savedData = SessionData.get('uploadedImage');
    if (savedData && savedData.preview) {
        console.log('[UPLOAD] 세션 데이터 복원:', savedData.filename);
        
        // 이미지 표시
        previewImage.src = savedData.preview;
        fileName.textContent = savedData.filename;
        fileSize.textContent = formatFileSize(savedData.file_size);
        resolution.textContent = savedData.resolution;
        filePath.textContent = savedData.file_path;
        
        // UI 전환
        uploadZone.style.display = 'none';
        previewSection.style.display = 'block';
        
        uploadedFileData = savedData;
    }
}

/**
 * 다음 페이지로 이동
 */
function goToNextPage() {
    if (!uploadedFileData) {
        showNotification('먼저 이미지를 업로드해주세요', 'warning');
        return;
    }
    
    console.log('[UPLOAD] 유사도 매칭 페이지로 이동');
    window.location.href = '/search.html';
}

/**
 * 최근 파일 목록 로드
 */
async function loadRecentFiles() {
    try {
        const response = await fetch(`${API_BASE_URL}/upload/list`);
        
        if (!response.ok) {
            throw new Error('파일 목록 조회 실패');
        }
        
        const data = await response.json();
        console.log('[UPLOAD] 최근 파일:', data.total_count);
        
        // 목록 표시
        if (data.files && data.files.length > 0) {
            recentFilesList.innerHTML = data.files.slice(0, 5).map(file => `
                <li onclick="loadFile('${file.file_path}', '${file.filename}')">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: 500;">${file.filename}</span>
                        <span style="font-size: 0.8rem; color: var(--text-secondary);">
                            ${formatFileSize(file.file_size)}
                        </span>
                    </div>
                    <div style="font-size: 0.75rem; color: var(--text-secondary); margin-top: 4px;">
                        ${formatDateTime(file.modified_at)}
                    </div>
                </li>
            `).join('');
        } else {
            recentFilesList.innerHTML = '<li class="no-files">업로드된 파일이 없습니다</li>';
        }
        
    } catch (error) {
        console.error('[UPLOAD] 파일 목록 로드 실패:', error);
        recentFilesList.innerHTML = '<li class="no-files">목록 로드 실패</li>';
    }
}

/**
 * 파일 로드
 */
function loadFile(filePath, filename) {
    console.log('[UPLOAD] 파일 로드:', filename);
    
    // 세션에 저장
    SessionData.set('uploadedImage', {
        filename: filename,
        file_path: filePath
    });
    
    showNotification('파일 선택됨: ' + filename, 'success');
    
    // 페이지 새로고침하여 프리뷰 표시
    location.reload();
}

/**
 * 인덱스 재구축
 */
async function rebuildIndex() {
    if (!confirm('인덱스를 재구축하시겠습니까? 시간이 다소 걸릴 수 있습니다.')) {
        return;
    }
    
    try {
        rebuildIndexBtn.disabled = true;
        rebuildIndexBtn.textContent = '재구축 중...';
        
        // 불량 이미지 인덱스 재구축
        const response = await fetch(`${API_BASE_URL}/build_index`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                gallery_dir: '../data/def_split',
                save_index: true
            })
        });
        
        if (!response.ok) {
            throw new Error('인덱스 재구축 실패');
        }
        
        const data = await response.json();
        console.log('[INDEX] 재구축 완료:', data);
        
        showNotification(`인덱스 재구축 완료 (${data.num_images}개 이미지)`, 'success');
        
        // 상태 새로고침
        await checkIndexStatus();
        
    } catch (error) {
        console.error('[INDEX] 재구축 실패:', error);
        showNotification(`재구축 실패: ${error.message}`, 'error');
    } finally {
        rebuildIndexBtn.disabled = false;
        rebuildIndexBtn.textContent = '🔄 재구축';
    }
}
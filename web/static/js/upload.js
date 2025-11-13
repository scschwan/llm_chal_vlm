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
const preprocessedImage = document.getElementById('preprocessedImage');
const imageInfoCard = document.getElementById('imageInfoCard');
const fileName = document.getElementById('fileName');
const fileSize = document.getElementById('fileSize');
const resolution = document.getElementById('resolution');
const reuploadBtn = document.getElementById('reuploadBtn');
const nextBtn = document.getElementById('nextBtn');
const checkIndexBtn = document.getElementById('checkIndexBtn');
const rebuildIndexBtn = document.getElementById('rebuildIndexBtn');

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
    console.log('[UPLOAD] 페이지 로드 완료');
    
    // ✅ 업로드 페이지 진입 시에만 세션 초기화 (뒤로가기 제외)
    // performance.navigation.type: 0=일반, 1=새로고침, 2=뒤로/앞으로
    const navigationType = performance.navigation.type;
    
    if (navigationType === 0 || navigationType === 1) {
        // 일반 진입이나 새로고침인 경우에만 초기화
        // 단, 세션에 uploadedImage가 없는 경우에만
        const existingData = SessionData.get('uploadedImage');
        if (!existingData) {
            console.log('[UPLOAD] 새 세션 시작 - 초기화');
            SessionData.clear();
        } else {
            console.log('[UPLOAD] 기존 세션 유지');
        }
    }
    
    // 이벤트 리스너 등록
    initEventListeners();
    
    // 인덱스 상태 확인
    checkIndexStatus();
    
    // ✅ 세션 데이터 복원 (있으면)
    restoreSessionData();
});

/**
 * 세션 데이터 복원
 */
function restoreSessionData() {
    const savedData = SessionData.get('uploadedImage');
    if (savedData && savedData.preview) {
        console.log('[UPLOAD] 세션 데이터 복원:', savedData.filename);
        
        // 이미지 표시
        previewImage.src = savedData.preview;
        preprocessedImage.src = savedData.preview;
        fileName.textContent = savedData.filename;
        fileSize.textContent = formatFileSize(savedData.file_size);
        resolution.textContent = savedData.resolution;
        
        // UI 전환
        uploadZone.style.display = 'none';
        previewSection.style.display = 'block';
        imageInfoCard.style.display = 'block';
        
        uploadedFileData = savedData;
    }
}

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
    const maxSize = 10 * 1024 * 1024;
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
            // 원본 이미지 표시
            previewImage.src = e.target.result;
            
            // 전처리 이미지 표시 (현재는 동일)
            preprocessedImage.src = e.target.result;
            
            // 정보 표시
            fileName.textContent = uploadData.filename;
            fileSize.textContent = formatFileSize(uploadData.file_size);
            resolution.textContent = `${img.width} × ${img.height}`;
            
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
            imageInfoCard.style.display = 'block';
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
    imageInfoCard.style.display = 'none';
    fileInput.value = '';
    uploadedFileData = null;
    progressFill.style.width = '0%';
    
     // ✅ 전체 워크플로우 초기화
    SessionData.startNewWorkflow();
    
    // ✅ 세션 데이터도 삭제
    // ✅ 세션 데이터 삭제 (다시 업로드 버튼만)
    SessionData.remove('uploadedImage');
    SessionData.remove('searchResults');
    SessionData.remove('selectedMatch');
    SessionData.remove('anomalyResult');
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
 * 인덱스 재구축
 */
async function rebuildIndex() {
    if (!confirm('인덱스를 재구축하시겠습니까? 시간이 다소 걸릴 수 있습니다.')) {
        return;
    }
    
    try {
        rebuildIndexBtn.disabled = true;
        rebuildIndexBtn.textContent = '재구축 중...';
        
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
        
        await checkIndexStatus();
        
    } catch (error) {
        console.error('[INDEX] 재구축 실패:', error);
        showNotification(`재구축 실패: ${error.message}`, 'error');
    } finally {
        rebuildIndexBtn.disabled = false;
        rebuildIndexBtn.textContent = '🔄 재구축';
    }
}
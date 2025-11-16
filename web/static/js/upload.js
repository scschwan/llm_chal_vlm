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
const imageInfoCard = document.getElementById('imageInfoCard');
const fileName = document.getElementById('fileName');
const fileSize = document.getElementById('fileSize');
const resolution = document.getElementById('resolution');
const reuploadBtn = document.getElementById('reuploadBtn');
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

// ✅ 단 하나의 DOMContentLoaded 이벤트
document.addEventListener('DOMContentLoaded', async () => {
    console.log('[UPLOAD] 페이지 로드 시작');
    
    try {
        // 1. 인증 확인
        console.log('[UPLOAD] 인증 확인 중...');
        const authResponse = await fetch('/api/auth/check');
        
        if (!authResponse.ok) {
            console.error('[UPLOAD] 인증 실패 - 로그인 페이지로 이동');
            window.location.href = '/login.html';
            return;
        }
        
        const authData = await authResponse.json();
        console.log('[UPLOAD] 인증 응답:', authData);
        
        if (!authData.authenticated) {
            console.warn('[UPLOAD] 미인증 사용자 - 로그인 페이지로 이동');
            window.location.href = '/login.html';
            return;
        }
        
        console.log('[UPLOAD] 인증 성공');
        
        // 2. 사용자 이름 표시
        if (authData.full_name) {
            const userNameElement = document.getElementById('userName');
            if (userNameElement) {
                userNameElement.textContent = authData.full_name;
                console.log('[UPLOAD] 사용자 이름 표시:', authData.full_name);
            } else {
                console.warn('[UPLOAD] userName 엘리먼트를 찾을 수 없음');
            }
        }
        
        // 3. 로그인 직후 세션 초기화
        const isNewLogin = sessionStorage.getItem('isNewLogin');
        if (isNewLogin === 'true') {
            console.log('[UPLOAD] 로그인 직후 - 세션 데이터 초기화');
            if (typeof SessionData !== 'undefined' && SessionData.clear) {
                SessionData.clear();
            }
            sessionStorage.removeItem('isNewLogin');
        }
        
        // 4. 기존 업로드 이미지 복원 시도
        if (typeof SessionData !== 'undefined') {
            const savedData = SessionData.get('uploadedImage');
            
            if (savedData && savedData.preview) {
                console.log('[UPLOAD] 이전 업로드 이미지 복원:', savedData.filename);
                restoreUploadedImage(savedData);
            } else {
                console.log('[UPLOAD] 새로운 업로드 대기 중');
                resetUploadState();
            }
        } else {
            console.error('[UPLOAD] SessionData 유틸리티를 찾을 수 없음');
            resetUploadState();
        }
        
        // 5. 이벤트 리스너 등록
        initEventListeners();
        
        console.log('[UPLOAD] 페이지 로드 완료');
        
    } catch (error) {
        console.error('[UPLOAD] 초기화 실패:', error);
        alert('페이지 로드 중 오류가 발생했습니다: ' + error.message);
        // 에러 발생 시에도 기본 UI는 표시
        resetUploadState();
        initEventListeners();
    }
});

/**
 * 업로드 상태 초기화
 */
function resetUploadState() {
    const uploadSection = document.getElementById('uploadSection');
    if (uploadSection) {
        uploadSection.style.display = 'block';
    }
    if (previewSection) {
        previewSection.style.display = 'none';
    }
    uploadedFileData = null;
    if (fileInput) {
        fileInput.value = '';
    }
}

/**
 * 업로드된 이미지 복원
 */
function restoreUploadedImage(savedData) {
    try {
        // 이미지 표시
        previewImage.src = savedData.preview;
        fileName.textContent = savedData.filename;
        fileSize.textContent = formatFileSize(savedData.file_size);
        resolution.textContent = savedData.resolution;
        
        // UI 전환
        const uploadSection = document.getElementById('uploadSection');
        if (uploadSection) {
            uploadSection.style.display = 'none';
        }
        previewSection.style.display = 'block';
        imageInfoCard.style.display = 'block';
        
        uploadedFileData = savedData;
    } catch (error) {
        console.error('[UPLOAD] 이미지 복원 실패:', error);
        resetUploadState();
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
    
    // 세션 데이터 삭제
    if (typeof SessionData !== 'undefined') {
        SessionData.remove('uploadedImage');
        SessionData.remove('searchResults');
        SessionData.remove('selectedMatch');
        SessionData.remove('anomalyResult');
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
 * 파일 크기 포맷팅
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
}
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

 // 로그아웃 함수
   let uploadedFile = null;
    let uploadedFilePath = null;

    // 로그인 체크
    if (!sessionStorage.getItem('isLoggedIn')) {
        alert('로그인이 필요합니다.');
        window.location.href = '/';
    }

    // 사용자 이름 표시
    const userName = sessionStorage.getItem('userName') || '작업자';
    document.getElementById('userName').textContent = userName;

    // 로그아웃
    function logout() {
        sessionStorage.clear();
        window.location.href = '/';
    }

    

    // 파일 선택 버튼 클릭
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
    fileInput.addEventListener('change', (e) => {
        handleFile(e.target.files[0]);
    });

    // 드래그 앤 드롭
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('dragover');
    });

    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('dragover');
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
        handleFile(e.dataTransfer.files[0]);
    });

    // 파일 처리
    async function handleFile(file) {
        if (!file) return;

        // 파일 타입 검증
        if (!file.type.startsWith('image/')) {
            alert('이미지 파일만 업로드 가능합니다.');
            return;
        }

        // 파일 크기 검증 (10MB)
        if (file.size > 10 * 1024 * 1024) {
            alert('파일 크기는 10MB를 초과할 수 없습니다.');
            return;
        }

        uploadedFile = file;

        // 프리뷰 표시
        const reader = new FileReader();
        reader.onload = (e) => {
            document.getElementById('previewImage').src = e.target.result;
            document.getElementById('fileName').textContent = file.name;
            document.getElementById('fileSize').textContent = formatFileSize(file.size);
            
            // 이미지 해상도 확인
            const img = new Image();
            img.onload = () => {
                document.getElementById('resolution').textContent = `${img.width} x ${img.height}`;
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);

        // 서버 업로드
        await uploadToServer(file);

        // 프리뷰 섹션 표시
        document.getElementById('uploadProgress').style.display = 'none';
        document.getElementById('previewSection').style.display = 'block';
    }

    // 서버 업로드
    async function uploadToServer(file) {
        const formData = new FormData();
        formData.append('file', file);

        document.getElementById('uploadProgress').style.display = 'block';
        document.getElementById('progressText').textContent = '업로드 중...';

        try {
            const response = await fetch('/upload/image', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.status === 'success') {
                uploadedFilePath = data.file_path;
                sessionStorage.setItem('uploadedImage', uploadedFilePath);
                document.getElementById('progressFill').style.width = '100%';
                document.getElementById('progressText').textContent = '업로드 완료!';
            } else {
                throw new Error(data.message || '업로드 실패');
            }
        } catch (error) {
            alert('업로드 중 오류가 발생했습니다: ' + error.message);
            document.getElementById('uploadProgress').style.display = 'none';
        }
    }

    // 파일 크기 포맷
    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
    }

    // 다음 단계로
    function goToSearch() {
        if (!uploadedFilePath) {
            alert('업로드된 이미지가 없습니다.');
            return;
        }
        window.location.href = '/search.html';
    }

    // 다시 업로드
    function resetUpload() {
        uploadedFile = null;
        uploadedFilePath = null;
        fileInput.value = '';
        document.getElementById('previewSection').style.display = 'none';
        sessionStorage.removeItem('uploadedImage');
    }

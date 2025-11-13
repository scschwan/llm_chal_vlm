const API_BASE = '/api/admin/image';
        const PRODUCT_API = '/api/admin/product';
        const DEFECT_TYPE_API = '/api/admin/defect-type';
        
        let selectedFiles = [];
        let allImages = [];
        let allDefectTypes = [];
        
        // 페이지 로드 시 초기화
        document.addEventListener('DOMContentLoaded', () => {
            loadProducts();
            loadImages();
            loadAllDefectTypes();
            setupDragAndDrop();
        });
        
        // 제품 목록 불러오기
        async function loadProducts() {
            try {
                const response = await fetch(PRODUCT_API);
                const products = await response.json();
                
                const productSelect = document.getElementById('productId');
                productSelect.innerHTML = '<option value="">-- 제품을 선택하세요 --</option>';
                
                const filterSelect = document.getElementById('filterProduct');
                filterSelect.innerHTML = '<option value="">전체 제품</option>';
                
                products.forEach(product => {
                    const option1 = document.createElement('option');
                    option1.value = product.product_id;
                    option1.textContent = `${product.product_name} (${product.product_code})`;
                    productSelect.appendChild(option1);
                    
                    const option2 = option1.cloneNode(true);
                    filterSelect.appendChild(option2);
                });
            } catch (error) {
                showMessage('제품 목록을 불러오는데 실패했습니다: ' + error.message, 'error');
            }
        }
        
        // 전체 불량 유형 불러오기
        async function loadAllDefectTypes() {
            try {
                const response = await fetch(DEFECT_TYPE_API);
                allDefectTypes = await response.json();
            } catch (error) {
                console.error('불량 유형 목록 로드 실패:', error);
            }
        }
        
        // 제품별 불량 유형 불러오기 (업로드 폼용)
        async function loadDefectTypes() {
            const productId = document.getElementById('productId').value;
            const defectTypeSelect = document.getElementById('defectTypeId');
            
            if (!productId) {
                defectTypeSelect.innerHTML = '<option value="">-- 제품을 먼저 선택하세요 --</option>';
                return;
            }
            
            try {
                const response = await fetch(`${DEFECT_TYPE_API}?product_id=${productId}`);
                const defectTypes = await response.json();
                
                defectTypeSelect.innerHTML = '<option value="">-- 불량 유형을 선택하세요 --</option>';
                
                defectTypes.forEach(defect => {
                    const option = document.createElement('option');
                    option.value = defect.defect_type_id;
                    option.textContent = `${defect.defect_name_ko} (${defect.defect_code})`;
                    defectTypeSelect.appendChild(option);
                });
            } catch (error) {
                showMessage('불량 유형 목록을 불러오는데 실패했습니다: ' + error.message, 'error');
            }
        }
        
        // 필터용 불량 유형 불러오기
        async function loadFilterDefectTypes() {
            const productId = document.getElementById('filterProduct').value;
            const filterDefectType = document.getElementById('filterDefectType');
            
            filterDefectType.innerHTML = '<option value="">전체 불량</option>';
            
            if (!productId) {
                filterImages();
                return;
            }
            
            const filtered = allDefectTypes.filter(d => d.product_id === parseInt(productId));
            
            filtered.forEach(defect => {
                const option = document.createElement('option');
                option.value = defect.defect_type_id;
                option.textContent = `${defect.defect_name_ko} (${defect.defect_code})`;
                filterDefectType.appendChild(option);
            });
            
            filterImages();
        }
        
        // 드래그 앤 드롭 설정
        function setupDragAndDrop() {
            const uploadArea = document.getElementById('uploadArea');
            const fileInput = document.getElementById('fileInput');
            
            uploadArea.addEventListener('click', () => fileInput.click());
            
            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.classList.add('dragover');
            });
            
            uploadArea.addEventListener('dragleave', () => {
                uploadArea.classList.remove('dragover');
            });
            
            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.classList.remove('dragover');
                
                const files = Array.from(e.dataTransfer.files);
                addFiles(files);
            });
            
            fileInput.addEventListener('change', (e) => {
                const files = Array.from(e.target.files);
                addFiles(files);
            });
        }
        
        // 파일 추가
        function addFiles(files) {
            files.forEach(file => {
                const validExtensions = ['.jpg', '.jpeg', '.png', '.zip'];
                const isValid = validExtensions.some(ext => 
                    file.name.toLowerCase().endsWith(ext)
                );
                
                if (isValid) {
                    selectedFiles.push(file);
                }
            });
            
            updateFileList();
        }
        
        // 파일 목록 업데이트
        function updateFileList() {
            const fileList = document.getElementById('fileList');
            
            if (selectedFiles.length === 0) {
                fileList.style.display = 'none';
                return;
            }
            
            fileList.style.display = 'block';
            fileList.innerHTML = '';
            
            selectedFiles.forEach((file, index) => {
                const fileItem = document.createElement('div');
                fileItem.className = 'file-item';
                fileItem.innerHTML = `
                    <span>${file.name} (${formatFileSize(file.size)})</span>
                    <button type="button" onclick="removeFile(${index})">삭제</button>
                `;
                fileList.appendChild(fileItem);
            });
        }
        
        // 파일 제거
        function removeFile(index) {
            selectedFiles.splice(index, 1);
            updateFileList();
        }
        
        // 파일 크기 포맷
        function formatFileSize(bytes) {
            if (bytes < 1024) return bytes + ' B';
            if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(2) + ' KB';
            return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
        }
        
        // 업로드 폼 제출
        document.getElementById('uploadForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const productId = document.getElementById('productId').value;
            const defectTypeId = document.getElementById('defectTypeId').value;
            
            if (!productId || !defectTypeId) {
                showMessage('제품과 불량 유형을 선택해주세요', 'error');
                return;
            }
            
            if (selectedFiles.length === 0) {
                showMessage('파일을 선택해주세요', 'error');
                return;
            }
            
            const formData = new FormData();
            formData.append('product_id', productId);
            formData.append('defect_type_id', defectTypeId);
            
            selectedFiles.forEach(file => {
                formData.append('files', file);
            });
            
            try {
                const uploadBtn = document.getElementById('uploadBtn');
                const progressContainer = document.getElementById('progressContainer');
                const progressFill = document.getElementById('progressFill');
                const progressInfo = document.getElementById('progressInfo');
                
                uploadBtn.disabled = true;
                progressContainer.style.display = 'block';
                
                const xhr = new XMLHttpRequest();
                
                xhr.upload.addEventListener('progress', (e) => {
                    if (e.lengthComputable) {
                        const percentComplete = (e.loaded / e.total) * 100;
                        progressFill.style.width = percentComplete + '%';
                        progressFill.textContent = Math.round(percentComplete) + '%';
                        progressInfo.textContent = `${selectedFiles.length}개 파일 업로드 중...`;
                    }
                });
                
                xhr.addEventListener('load', () => {
                    uploadBtn.disabled = false;
                    
                    if (xhr.status === 200) {
                        const result = JSON.parse(xhr.responseText);
                        showMessage(
                            `업로드 완료: ${result.uploaded_count}개 성공, ${result.failed_count}개 실패`,
                            result.uploaded_count > 0 ? 'success' : 'error'
                        );
                        resetForm();
                        loadImages();
                        
                        setTimeout(() => {
                            progressContainer.style.display = 'none';
                        }, 2000);
                    } else {
                        const data = JSON.parse(xhr.responseText);
                        showMessage(data.detail || '이미지 업로드에 실패했습니다', 'error');
                        progressContainer.style.display = 'none';
                    }
                });
                
                xhr.addEventListener('error', () => {
                    uploadBtn.disabled = false;
                    showMessage('서버 오류가 발생했습니다', 'error');
                    progressContainer.style.display = 'none';
                });
                
                xhr.open('POST', `${API_BASE}/defect`);
                xhr.send(formData);
                
            } catch (error) {
                showMessage('서버 오류가 발생했습니다: ' + error.message, 'error');
                document.getElementById('uploadBtn').disabled = false;
            }
        });
        
        // 이미지 목록 불러오기
        async function loadImages() {
            try {
                const response = await fetch(`${API_BASE}/defect?limit=1000`);
                allImages = await response.json();
                
                document.getElementById('totalCount').textContent = allImages.length;
                
                renderImages(allImages);
            } catch (error) {
                showMessage('이미지 목록을 불러오는데 실패했습니다: ' + error.message, 'error');
            }
        }
        
        // 이미지 렌더링
        function renderImages(images) {
            const imageGrid = document.getElementById('imageGrid');
            imageGrid.innerHTML = '';
            
            if (images.length === 0) {
                imageGrid.innerHTML = '<p style="grid-column: 1/-1; text-align:center; color:#999;">등록된 이미지가 없습니다</p>';
                return;
            }
            
            images.forEach(image => {
                const card = document.createElement('div');
                card.className = 'image-card';
                card.innerHTML = `
                    <img src="${image.display_url}" alt="${image.file_name}" 
                         onerror="this.src='data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 width=%22200%22 height=%22200%22%3E%3Crect fill=%22%23f0f0f0%22 width=%22200%22 height=%22200%22/%3E%3Ctext x=%2250%25%22 y=%2250%25%22 dominant-baseline=%22middle%22 text-anchor=%22middle%22 fill=%22%23999%22%3ENo Image%3C/text%3E%3C/svg%3E'">
                    <div class="image-info">
                        <div class="defect-badge">${image.defect_name || image.defect_code}</div>
                        <div class="image-name">${image.file_name}</div>
                        <div class="image-actions">
                            <button class="btn btn-danger" onclick="deleteImage(${image.image_id})">삭제</button>
                        </div>
                    </div>
                `;
                imageGrid.appendChild(card);
            });
        }
        
        // 이미지 필터링
        function filterImages() {
            const productId = document.getElementById('filterProduct').value;
            const defectTypeId = document.getElementById('filterDefectType').value;
            
            let filtered = allImages;
            
            if (productId) {
                filtered = filtered.filter(img => img.product_id === parseInt(productId));
            }
            
            if (defectTypeId) {
                filtered = filtered.filter(img => img.defect_type_id === parseInt(defectTypeId));
            }
            
            renderImages(filtered);
        }
        
        // 이미지 삭제
        async function deleteImage(imageId) {
            if (!confirm('정말로 이 이미지를 삭제하시겠습니까?')) return;
            
            try {
                const response = await fetch(`${API_BASE}/${imageId}`, {
                    method: 'DELETE'
                });
                
                if (response.ok) {
                    showMessage('이미지가 삭제되었습니다', 'success');
                    loadImages();
                } else {
                    const data = await response.json();
                    showMessage(data.detail || '이미지 삭제에 실패했습니다', 'error');
                }
            } catch (error) {
                showMessage('서버 오류가 발생했습니다: ' + error.message, 'error');
            }
        }
        
        // 폼 초기화
        function resetForm() {
            document.getElementById('uploadForm').reset();
            selectedFiles = [];
            updateFileList();
            document.getElementById('defectTypeId').innerHTML = '<option value="">-- 제품을 먼저 선택하세요 --</option>';
        }
        
        // 메시지 표시
        function showMessage(text, type) {
            const messageDiv = document.getElementById('message');
            messageDiv.textContent = text;
            messageDiv.className = `message ${type}`;
            messageDiv.style.display = 'block';
            
            setTimeout(() => {
                messageDiv.style.display = 'none';
            }, 5000);
        }
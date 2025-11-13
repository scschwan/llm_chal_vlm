 const API_BASE = '/api/admin/manual';
        const PRODUCT_API = '/api/admin/product';
        
        let selectedFile = null;
        
        // 페이지 로드 시 초기화
        document.addEventListener('DOMContentLoaded', () => {
            loadProducts();
            loadManuals();
            setupDragAndDrop();
        });
        
        // 제품 목록 불러오기
        async function loadProducts() {
            try {
                const response = await fetch(PRODUCT_API);
                const products = await response.json();
                
                const productSelect = document.getElementById('productId');
                productSelect.innerHTML = '<option value="">-- 제품을 선택하세요 --</option>';
                
                products.forEach(product => {
                    const option = document.createElement('option');
                    option.value = product.product_id;
                    option.textContent = `${product.product_name} (${product.product_code})`;
                    productSelect.appendChild(option);
                });
            } catch (error) {
                showMessage('제품 목록을 불러오는데 실패했습니다: ' + error.message, 'error');
            }
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
                
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    handleFileSelect(files[0]);
                }
            });
            
            fileInput.addEventListener('change', (e) => {
                if (e.target.files.length > 0) {
                    handleFileSelect(e.target.files[0]);
                }
            });
        }
        
        // 파일 선택 처리
        function handleFileSelect(file) {
            if (!file.name.toLowerCase().endsWith('.pdf')) {
                showMessage('PDF 파일만 업로드 가능합니다', 'error');
                return;
            }
            
            selectedFile = file;
            document.getElementById('fileName').textContent = `선택된 파일: ${file.name} (${formatFileSize(file.size)})`;
        }
        
        // 파일 크기 포맷
        function formatFileSize(bytes) {
            if (bytes < 1024) return bytes + ' B';
            if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(2) + ' KB';
            return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
        }
        
        // 매뉴얼 업로드 폼 제출
        document.getElementById('manualForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const productId = document.getElementById('productId').value;
            
            if (!productId) {
                showMessage('제품을 선택해주세요', 'error');
                return;
            }
            
            if (!selectedFile) {
                showMessage('파일을 선택해주세요', 'error');
                return;
            }
            
            const formData = new FormData();
            formData.append('product_id', productId);
            formData.append('file', selectedFile);
            
            try {
                const progressBar = document.getElementById('progressBar');
                const progressFill = document.getElementById('progressFill');
                progressBar.style.display = 'block';
                
                const xhr = new XMLHttpRequest();
                
                xhr.upload.addEventListener('progress', (e) => {
                    if (e.lengthComputable) {
                        const percentComplete = (e.loaded / e.total) * 100;
                        progressFill.style.width = percentComplete + '%';
                        progressFill.textContent = Math.round(percentComplete) + '%';
                    }
                });
                
                xhr.addEventListener('load', () => {
                    if (xhr.status === 200) {
                        showMessage('매뉴얼이 성공적으로 업로드되었습니다', 'success');
                        resetForm();
                        loadManuals();
                        progressBar.style.display = 'none';
                    } else {
                        const data = JSON.parse(xhr.responseText);
                        showMessage(data.detail || '매뉴얼 업로드에 실패했습니다', 'error');
                        progressBar.style.display = 'none';
                    }
                });
                
                xhr.addEventListener('error', () => {
                    showMessage('서버 오류가 발생했습니다', 'error');
                    progressBar.style.display = 'none';
                });
                
                xhr.open('POST', API_BASE);
                xhr.send(formData);
                
            } catch (error) {
                showMessage('서버 오류가 발생했습니다: ' + error.message, 'error');
            }
        });
        
        // 매뉴얼 목록 불러오기
        async function loadManuals() {
            try {
                const response = await fetch(API_BASE);
                const manuals = await response.json();
                
                const tbody = document.querySelector('#manualTable tbody');
                tbody.innerHTML = '';
                
                manuals.forEach(manual => {
                    const tr = document.createElement('tr');
                    tr.innerHTML = `
                        <td>${manual.manual_id}</td>
                        <td>${manual.product_name} (${manual.product_code})</td>
                        <td>${manual.file_name}</td>
                        <td class="file-size">${formatFileSize(manual.file_size || 0)}</td>
                        <td class="${manual.vector_indexed ? 'status-indexed' : 'status-not-indexed'}">
                            ${manual.vector_indexed ? '완료' : '미완료'}
                        </td>
                        <td>${new Date(manual.created_at).toLocaleDateString()}</td>
                        <td class="action-buttons">
                            <button class="btn btn-secondary" onclick="downloadManual(${manual.manual_id})">다운로드</button>
                            <button class="btn btn-danger" onclick="deleteManual(${manual.manual_id})">삭제</button>
                        </td>
                    `;
                    tbody.appendChild(tr);
                });
            } catch (error) {
                showMessage('매뉴얼 목록을 불러오는데 실패했습니다: ' + error.message, 'error');
            }
        }
        
        // 매뉴얼 다운로드
        async function downloadManual(manualId) {
            try {
                window.location.href = `${API_BASE}/${manualId}/download`;
            } catch (error) {
                showMessage('매뉴얼 다운로드에 실패했습니다: ' + error.message, 'error');
            }
        }
        
        // 매뉴얼 삭제
        async function deleteManual(manualId) {
            if (!confirm('정말로 이 매뉴얼을 삭제하시겠습니까?')) return;
            
            try {
                const response = await fetch(`${API_BASE}/${manualId}`, {
                    method: 'DELETE'
                });
                
                if (response.ok) {
                    showMessage('매뉴얼이 삭제되었습니다', 'success');
                    loadManuals();
                } else {
                    const data = await response.json();
                    showMessage(data.detail || '매뉴얼 삭제에 실패했습니다', 'error');
                }
            } catch (error) {
                showMessage('서버 오류가 발생했습니다: ' + error.message, 'error');
            }
        }
        
        // 폼 초기화
        function resetForm() {
            document.getElementById('manualForm').reset();
            document.getElementById('fileName').textContent = '';
            selectedFile = null;
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
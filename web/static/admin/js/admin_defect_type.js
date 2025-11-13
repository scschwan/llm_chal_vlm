const API_BASE = '/api/admin/defect-type';
        const PRODUCT_API = '/api/admin/product';
        
        let allDefectTypes = [];
        
        // 페이지 로드 시 초기화
        document.addEventListener('DOMContentLoaded', () => {
            loadProducts();
            loadDefectTypes();
        });
        
        // 제품 목록 불러오기
        async function loadProducts() {
            try {
                const response = await fetch(PRODUCT_API);
                const products = await response.json();
                
                // 폼 제품 선택
                const productSelect = document.getElementById('productId');
                productSelect.innerHTML = '<option value="">-- 제품을 선택하세요 --</option>';
                
                // 필터 제품 선택
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
        
        // 불량 유형 등록 폼 제출
        document.getElementById('defectTypeForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const formData = {
                product_id: parseInt(document.getElementById('productId').value),
                defect_name_ko: document.getElementById('defectNameKo').value,
                defect_code: document.getElementById('defectCode').value.toLowerCase(),
                defect_name_en: document.getElementById('defectNameEn').value || null,
                full_name_ko: document.getElementById('fullNameKo').value || null
            };
            
            try {
                const response = await fetch(API_BASE, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(formData)
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    showMessage('불량 유형이 성공적으로 등록되었습니다', 'success');
                    resetForm();
                    loadDefectTypes();
                } else {
                    showMessage(data.detail || '불량 유형 등록에 실패했습니다', 'error');
                }
            } catch (error) {
                showMessage('서버 오류가 발생했습니다: ' + error.message, 'error');
            }
        });
        
        // 불량 유형 목록 불러오기
        async function loadDefectTypes() {
            try {
                const response = await fetch(API_BASE);
                allDefectTypes = await response.json();
                
                renderDefectTypes(allDefectTypes);
            } catch (error) {
                showMessage('불량 유형 목록을 불러오는데 실패했습니다: ' + error.message, 'error');
            }
        }
        
        // 불량 유형 테이블 렌더링
        function renderDefectTypes(defectTypes) {
            const tbody = document.querySelector('#defectTypeTable tbody');
            tbody.innerHTML = '';
            
            defectTypes.forEach(defect => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>${defect.defect_type_id}</td>
                    <td>${defect.product_name} (${defect.product_code})</td>
                    <td>${defect.defect_name_ko}</td>
                    <td>${defect.defect_code}</td>
                    <td>${new Date(defect.created_at).toLocaleDateString()}</td>
                    <td class="${defect.is_active ? 'status-active' : 'status-inactive'}">
                        ${defect.is_active ? '활성' : '비활성'}
                    </td>
                    <td class="action-buttons">
                        <button class="btn btn-secondary" onclick="editDefectType(${defect.defect_type_id})">수정</button>
                        <button class="btn btn-danger" onclick="deleteDefectType(${defect.defect_type_id})">삭제</button>
                    </td>
                `;
                tbody.appendChild(tr);
            });
        }
        
        // 필터링
        function filterDefectTypes() {
            const productId = document.getElementById('filterProduct').value;
            
            if (!productId) {
                renderDefectTypes(allDefectTypes);
            } else {
                const filtered = allDefectTypes.filter(d => d.product_id === parseInt(productId));
                renderDefectTypes(filtered);
            }
        }
        
        // 불량 유형 수정
        async function editDefectType(defectTypeId) {
            const defectNameKo = prompt('새 불량명(한글)을 입력하세요:');
            if (!defectNameKo) return;
            
            try {
                const response = await fetch(`${API_BASE}/${defectTypeId}`, {
                    method: 'PUT',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        defect_name_ko: defectNameKo
                    })
                });
                
                if (response.ok) {
                    showMessage('불량 유형이 수정되었습니다', 'success');
                    loadDefectTypes();
                } else {
                    const data = await response.json();
                    showMessage(data.detail || '불량 유형 수정에 실패했습니다', 'error');
                }
            } catch (error) {
                showMessage('서버 오류가 발생했습니다: ' + error.message, 'error');
            }
        }
        
        // 불량 유형 삭제
        async function deleteDefectType(defectTypeId) {
            if (!confirm('정말로 이 불량 유형을 삭제하시겠습니까?')) return;
            
            try {
                const response = await fetch(`${API_BASE}/${defectTypeId}`, {
                    method: 'DELETE'
                });
                
                if (response.ok) {
                    showMessage('불량 유형이 삭제되었습니다', 'success');
                    loadDefectTypes();
                } else {
                    const data = await response.json();
                    showMessage(data.detail || '불량 유형 삭제에 실패했습니다', 'error');
                }
            } catch (error) {
                showMessage('서버 오류가 발생했습니다: ' + error.message, 'error');
            }
        }
        
        // 폼 초기화
        function resetForm() {
            document.getElementById('defectTypeForm').reset();
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
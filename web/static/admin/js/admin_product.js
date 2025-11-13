const API_BASE = '/api/admin/product';
        
        // 페이지 로드 시 제품 목록 불러오기
        document.addEventListener('DOMContentLoaded', () => {
            loadProducts();
        });
        
        // 제품 등록 폼 제출
        document.getElementById('productForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const formData = {
                product_code: document.getElementById('productCode').value,
                product_name: document.getElementById('productName').value,
                description: document.getElementById('description').value || null
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
                    showMessage('제품이 성공적으로 등록되었습니다', 'success');
                    resetForm();
                    loadProducts();
                } else {
                    showMessage(data.detail || '제품 등록에 실패했습니다', 'error');
                }
            } catch (error) {
                showMessage('서버 오류가 발생했습니다: ' + error.message, 'error');
            }
        });
        
        // 제품 목록 불러오기
        async function loadProducts() {
            try {
                const response = await fetch(API_BASE);
                const products = await response.json();
                
                const tbody = document.querySelector('#productTable tbody');
                tbody.innerHTML = '';
                
                products.forEach(product => {
                    const tr = document.createElement('tr');
                    tr.innerHTML = `
                        <td>${product.product_id}</td>
                        <td>${product.product_code}</td>
                        <td>${product.product_name}</td>
                        <td>${new Date(product.created_at).toLocaleDateString()}</td>
                        <td class="${product.is_active ? 'status-active' : 'status-inactive'}">
                            ${product.is_active ? '활성' : '비활성'}
                        </td>
                        <td class="action-buttons">
                            <button class="btn btn-secondary" onclick="editProduct(${product.product_id})">수정</button>
                            <button class="btn btn-danger" onclick="deleteProduct(${product.product_id})">삭제</button>
                        </td>
                    `;
                    tbody.appendChild(tr);
                });
            } catch (error) {
                showMessage('제품 목록을 불러오는데 실패했습니다: ' + error.message, 'error');
            }
        }
        
        // 제품 수정
        async function editProduct(productId) {
            const productName = prompt('새 제품명을 입력하세요:');
            if (!productName) return;
            
            try {
                const response = await fetch(`${API_BASE}/${productId}`, {
                    method: 'PUT',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        product_name: productName
                    })
                });
                
                if (response.ok) {
                    showMessage('제품이 수정되었습니다', 'success');
                    loadProducts();
                } else {
                    const data = await response.json();
                    showMessage(data.detail || '제품 수정에 실패했습니다', 'error');
                }
            } catch (error) {
                showMessage('서버 오류가 발생했습니다: ' + error.message, 'error');
            }
        }
        
        // 제품 삭제
        async function deleteProduct(productId) {
            if (!confirm('정말로 이 제품을 삭제하시겠습니까?')) return;
            
            try {
                const response = await fetch(`${API_BASE}/${productId}`, {
                    method: 'DELETE'
                });
                
                if (response.ok) {
                    showMessage('제품이 삭제되었습니다', 'success');
                    loadProducts();
                } else {
                    const data = await response.json();
                    showMessage(data.detail || '제품 삭제에 실패했습니다', 'error');
                }
            } catch (error) {
                showMessage('서버 오류가 발생했습니다: ' + error.message, 'error');
            }
        }
        
        // 폼 초기화
        function resetForm() {
            document.getElementById('productForm').reset();
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
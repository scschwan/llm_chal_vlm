 // 페이지 로드 시 인증 확인
        document.addEventListener('DOMContentLoaded', async () => {
            try {
                const response = await fetch('/api/auth/session');
                
                if (!response.ok) {
                    window.location.href = '/login.html';
                    return;
                }
                
                const session = await response.json();
                
                if (session.user_type !== 'admin') {
                    window.location.href = '/upload.html';
                    return;
                }
                
                document.getElementById('userName').textContent = session.full_name;
                
                // 현재 페이지 활성화
                setActivePage();
                
            } catch (error) {
                console.error('세션 확인 실패:', error);
                window.location.href = '/login.html';
            }
        });
        
        // 현재 페이지 메뉴 활성화
        function setActivePage() {
            const currentPath = window.location.pathname;
            const menuItems = document.querySelectorAll('.menu-item');
            
            menuItems.forEach(item => {
                if (item.getAttribute('href') === currentPath) {
                    item.classList.add('active');
                }
            });
        }
        
        // 로그아웃
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
# 서비스 시작
sudo systemctl start api-server.service
sudo systemctl start llm-server.service

# 서비스 중지
sudo systemctl stop api-server.service
sudo systemctl stop llm-server.service

# 서비스 재시작
sudo systemctl restart api-server.service
sudo systemctl restart llm-server.service

# 부팅 시 자동 시작 활성화
sudo systemctl enable api-server.service
sudo systemctl enable llm-server.service

# 부팅 시 자동 시작 비활성화
sudo systemctl disable api-server.service
sudo systemctl disable llm-server.service

# 서비스 상태 확인
sudo systemctl status api-server.service
sudo systemctl status llm-server.service



# 80번 포트 확인
sudo lsof -i :80
# 또는
sudo netstat -tulpn | grep :80
# 또는
sudo ss -tulpn | grep :80

# 5001번 포트 확인
sudo lsof -i :5001
# 또는
sudo netstat -tulpn | grep :5001
# 또는
sudo ss -tulpn | grep :5001
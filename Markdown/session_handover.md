# 세션 인계서 (Session Handover)

**작성일**: 2025-01-13  
**이전 세션**: Claude Sonnet 4.5 (2025-01-13)  
**프로젝트**: 제조 불량 검출 AI 시스템  
**GitHub**: https://github.com/scschwan/llm_chal_vlm

---

## 📌 즉시 착수해야 할 작업 (최우선 순위)

### 🚨 1. 인덱스 자동 전환 기능 구현 (긴급)

**현재 문제:**
- 유사도 매칭과 이상 검출에서 동일한 인덱스 사용
- 불량 이미지(`def_split`)와 정상 이미지(`ok_split`)가 혼재되어 정확도 저하

**해결 방안:**
각 화면 진입 시 자동으로 적절한 인덱스로 전환

#### 구현 상세

**1단계: API 서버 수정 (`web/api_server.py`)**
```python
# 전역 변수 추가
current_index_type = None  # 'defect' or 'normal'

# 인덱스 전환 함수
async def switch_index(index_type: str):
    """
    index_type: 'defect' (불량 이미지용) 또는 'normal' (정상 이미지용)
    """
    global current_index_type
    
    if current_index_type == index_type:
        return {"status": "already_loaded", "index_type": index_type}
    
    if index_type == "defect":
        gallery_dir = project_root / "data" / "def_split"
    elif index_type == "normal":
        gallery_dir = project_root / "data" / "ok_split"
    else:
        raise ValueError(f"Invalid index_type: {index_type}")
    
    # 인덱스 구축
    matcher.build_index(str(gallery_dir))
    
    # 저장 (선택적)
    index_path = INDEX_DIR / index_type
    matcher.save_index(str(index_path))
    
    current_index_type = index_type
    
    return {
        "status": "success",
        "index_type": index_type,
        "gallery_dir": str(gallery_dir),
        "image_count": len(matcher.gallery_paths)
    }

# 엔드포인트 추가
@app.post("/index/switch")
async def switch_index_endpoint(index_type: str):
    """인덱스 타입 전환 (defect 또는 normal)"""
    try:
        result = await switch_index(index_type)
        return result
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@app.get("/index/status")
async def get_index_status():
    """현재 인덱스 상태 조회"""
    return {
        "current_index_type": current_index_type,
        "gallery_count": len(matcher.gallery_paths) if matcher.gallery_paths else 0
    }
```

**2단계: 프론트엔드 수정 (`web/static/js/matching.js`)**
```javascript
// 유사도 검색 화면 진입 시 (페이지 로드)
async function ensureDefectIndex() {
    try {
        const response = await fetch(`${API_BASE_URL}/index/switch?index_type=defect`, {
            method: 'POST'
        });
        const data = await response.json();
        console.log('[INDEX] 불량 이미지 인덱스 로드:', data);
        
        // UI 상태 표시
        updateIndexStatus('defect', data.image_count);
    } catch (err) {
        console.error('[INDEX] 전환 실패:', err);
    }
}

// 이상 검출 화면 진입 시
async function ensureNormalIndex() {
    try {
        const response = await fetch(`${API_BASE_URL}/index/switch?index_type=normal`, {
            method: 'POST'
        });
        const data = await response.json();
        console.log('[INDEX] 정상 이미지 인덱스 로드:', data);
        
        updateIndexStatus('normal', data.image_count);
    } catch (err) {
        console.error('[INDEX] 전환 실패:', err);
    }
}

// 인덱스 상태 UI 업데이트
function updateIndexStatus(type, count) {
    const statusEl = document.getElementById('indexStatus');
    if (statusEl) {
        const typeText = type === 'defect' ? '불량 이미지' : '정상 이미지';
        statusEl.innerHTML = `
            <span style="color: #10b981;">✅ ${typeText} 인덱스 활성</span>
            <span style="color: #6b7280;"> (${count}개 이미지)</span>
        `;
        statusEl.style.display = 'block';
    }
}

// 페이지 로드 시 자동 실행
document.addEventListener('DOMContentLoaded', () => {
    // 현재 탭에 따라 인덱스 전환
    const currentTab = document.querySelector('.tab.active').dataset.tab;
    
    if (currentTab === 'search') {
        ensureDefectIndex();  // 유사도 검색 → 불량 이미지
    } else if (currentTab === 'anomaly') {
        ensureNormalIndex();  // 이상 검출 → 정상 이미지
    }
});

// 탭 전환 시에도 인덱스 자동 전환
function switchTab(tabElement) {
    // ... 기존 코드 ...
    
    const tabName = tabElement.dataset.tab;
    
    if (tabName === 'search') {
        ensureDefectIndex();
    } else if (tabName === 'anomaly') {
        ensureNormalIndex();
    }
}
```

**3단계: HTML 수정 (`web/matching.html`)**
```html
<!-- 인덱스 관리 섹션에 상태 표시 추가 -->
<div class="index-management">
    <h4>🔧 인덱스 관리</h4>
    
    <!-- 현재 인덱스 상태 -->
    <div id="indexStatus" style="
        font-size: 0.9em; 
        padding: 8px; 
        background: #f0fdf4; 
        border: 1px solid #86efac;
        border-radius: 4px; 
        margin-bottom: 10px;
        display: none;
    "></div>
    
    <div style="display: flex; gap: 10px; margin-bottom: 10px;">
        <button class="index-btn" id="checkIndexBtn">📊 상태 확인</button>
        <button class="index-btn" id="rebuildIndexBtn">🔄 재구축</button>
    </div>
</div>
```

**테스트 시나리오:**
1. ✅ 유사도 검색 화면 진입 → `def_split` 인덱스 자동 로드 확인
2. ✅ 이상 검출 화면 진입 → `ok_split` 인덱스 자동 로드 확인
3. ✅ 탭 전환 시 인덱스 자동 전환 확인
4. ✅ UI에 현재 인덱스 타입 표시 확인

---

### 🎯 2. Phase 1: UI/UX 개선 (우선순위 1)

Phase 1의 4개 작업을 순차적으로 진행하세요.

#### 2.1 이미지 업로드 화면 분리

**목표**: 독립된 업로드 전용 페이지 생성

**작업 내용:**

1. **새 HTML 파일 생성**: `web/upload.html`
```html
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>이미지 업로드</title>
    <link rel="stylesheet" href="/static/css/upload.css">
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📸 이미지 업로드</h1>
            <p>불량 의심 이미지를 업로드하세요</p>
        </div>

        <div class="nav">
            <a href="upload.html" class="nav-btn active">이미지 업로드</a>
            <a href="matching.html" class="nav-btn">유사도 매칭</a>
            <a href="anomaly.html" class="nav-btn">이상 영역 검출</a>
            <a href="manual.html" class="nav-btn">대응 매뉴얼</a>
            <a href="dashboard.html" class="nav-btn">통계 대시보드</a>
        </div>

        <!-- 대형 업로드 영역 -->
        <div class="upload-container">
            <div class="upload-zone" id="uploadZone">
                <div class="upload-icon">📸</div>
                <h2>이미지를 드래그하거나 클릭하여 업로드</h2>
                <p>JPG, PNG, WEBP 지원</p>
                <input type="file" id="fileInput" accept="image/*">
            </div>

            <!-- 프리뷰 -->
            <div id="previewSection" style="display: none;">
                <img id="previewImage" class="preview-large">
                <div class="image-info">
                    <p>파일명: <span id="fileName"></span></p>
                    <p>크기: <span id="fileSize"></span></p>
                    <p>해상도: <span id="resolution"></span></p>
                </div>
                <button class="next-button" id="goToMatching">
                    다음: 유사도 매칭으로 이동 →
                </button>
            </div>
        </div>

        <!-- 인덱스 관리 (기존 유지) -->
        <div class="index-section">
            <h3>🔧 인덱스 관리</h3>
            <div id="indexStatus"></div>
            <button id="checkIndexBtn">📊 상태 확인</button>
            <button id="rebuildIndexBtn">🔄 재구축</button>
        </div>
    </div>

    <script src="/static/js/upload.js"></script>
</body>
</html>
```

2. **CSS 파일**: `web/static/css/upload.css`
```css
.upload-zone {
    width: 100%;
    height: 400px;
    border: 3px dashed #3b82f6;
    border-radius: 16px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    transition: all 0.3s;
}

.upload-zone:hover {
    background: #eff6ff;
    border-color: #2563eb;
}

.preview-large {
    max-width: 800px;
    max-height: 600px;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}
```

3. **JavaScript**: `web/static/js/upload.js`
```javascript
// 업로드 처리
document.getElementById('fileInput').addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // 파일 정보 표시
    document.getElementById('fileName').textContent = file.name;
    document.getElementById('fileSize').textContent = formatFileSize(file.size);

    // 이미지 프리뷰
    const reader = new FileReader();
    reader.onload = (event) => {
        const img = new Image();
        img.onload = () => {
            document.getElementById('resolution').textContent = `${img.width} × ${img.height}`;
            document.getElementById('previewImage').src = event.target.result;
            document.getElementById('previewSection').style.display = 'block';
            
            // 세션 스토리지에 저장 (다음 화면으로 전달)
            sessionStorage.setItem('uploadedImage', event.target.result);
            sessionStorage.setItem('uploadedFileName', file.name);
        };
        img.src = event.target.result;
    };
    reader.readAsDataURL(file);
});

// 다음 화면으로 이동
document.getElementById('goToMatching').addEventListener('click', () => {
    window.location.href = 'matching.html';
});
```

---

#### 2.2 유사도 매칭 결과 화면 분리

**목표**: 독립된 매칭 결과 페이지 생성

**작업 내용:**

1. **새 HTML 파일**: `web/matching.html` (기존 파일 개선)
```html
<!-- LLM/VLM 버튼 제거, 다음 단계 버튼만 유지 -->
<div class="results-actions">
    <!-- 불량 등록 버튼 (기존 유지) -->
    <button id="registerDefectBtn" class="register-button">
        💾 불량 이미지 등록
    </button>
    
    <!-- 다음 단계 버튼 추가 -->
    <button id="goToAnomalyBtn" class="next-button">
        다음: 이상 영역 검출로 이동 →
    </button>
</div>
```

2. **JavaScript 수정**: `web/static/js/matching.js`
```javascript
// 페이지 로드 시 업로드된 이미지 복원
document.addEventListener('DOMContentLoaded', () => {
    const uploadedImage = sessionStorage.getItem('uploadedImage');
    if (uploadedImage) {
        document.getElementById('previewImage').src = uploadedImage;
        // 자동으로 검색 실행할지 여부는 선택
    }
});

// 다음 화면으로 이동 (제품/불량 정보 전달)
document.getElementById('goToAnomalyBtn').addEventListener('click', () => {
    const top1Result = currentSearchResults[0];  // TOP-1 결과
    
    // URL 파라미터로 전달
    const params = new URLSearchParams({
        product: top1Result.product,
        defect: top1Result.defect,
        normalImagePath: top1Result.image_path,
        inputImagePath: currentInputImagePath
    });
    
    window.location.href = `anomaly.html?${params.toString()}`;
});
```

---

#### 2.3 이상 영역 검출 화면 분리 및 자동화

**목표**: 독립 페이지 + 자동 검출 + 간소화된 시각화

**작업 내용:**

1. **새 HTML 파일**: `web/anomaly.html`
```html
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>이상 영역 검출</title>
    <link rel="stylesheet" href="/static/css/anomaly.css">
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 이상 영역 검출</h1>
            <p id="productInfo">제품: <span id="productName"></span> | 불량: <span id="defectName"></span></p>
        </div>

        <div class="nav">
            <a href="upload.html" class="nav-btn">이미지 업로드</a>
            <a href="matching.html" class="nav-btn">유사도 매칭</a>
            <a href="anomaly.html" class="nav-btn active">이상 영역 검출</a>
            <a href="manual.html" class="nav-btn">대응 매뉴얼</a>
            <a href="dashboard.html" class="nav-btn">통계 대시보드</a>
        </div>

        <!-- 자동 검출 진행 상태 -->
        <div id="detectingStatus" class="status-box">
            <div class="spinner"></div>
            <p>이상 영역을 자동으로 검출하고 있습니다...</p>
        </div>

        <!-- 검출 결과 (비교 이미지만 표시) -->
        <div id="resultSection" style="display: none;">
            <h2>🔍 비교 결과</h2>
            
            <div class="comparison-grid">
                <div class="comparison-item">
                    <h3>정상 기준 이미지</h3>
                    <img id="normalImage" class="result-image">
                </div>
                <div class="comparison-item">
                    <h3>이상 영역 표시</h3>
                    <img id="overlayImage" class="result-image">
                    <div class="anomaly-info">
                        <p>이상 점수: <span id="anomalyScore"></span></p>
                        <p>판정: <span id="anomalyJudgment"></span></p>
                    </div>
                </div>
            </div>

            <!-- 매뉴얼 생성 버튼 -->
            <div class="manual-buttons">
                <h3>AI 대응 매뉴얼 생성</h3>
                <button id="btnHyperClovax" class="model-button">
                    🧠 HyperCLOVAX
                </button>
                <button id="btnExaone" class="model-button">
                    🤖 EXAONE 3.5
                </button>
                <button id="btnVLM" class="model-button">
                    🖼️ VLM (LLaVA)
                </button>
            </div>
        </div>
    </div>

    <script src="/static/js/anomaly.js"></script>
</body>
</html>
```

2. **JavaScript 자동 검출**: `web/static/js/anomaly.js`
```javascript
// 페이지 로드 시 자동 실행
document.addEventListener('DOMContentLoaded', async () => {
    // URL 파라미터에서 정보 추출
    const params = new URLSearchParams(window.location.search);
    const product = params.get('product');
    const defect = params.get('defect');
    const normalImagePath = params.get('normalImagePath');
    const inputImagePath = params.get('inputImagePath');

    // UI에 제품/불량 정보 표시
    document.getElementById('productName').textContent = product;
    document.getElementById('defectName').textContent = defect;

    // 정상 이미지 인덱스로 자동 전환
    await fetch(`${API_BASE_URL}/index/switch?index_type=normal`, {
        method: 'POST'
    });

    // 자동으로 이상 검출 실행
    try {
        const response = await fetch(`${API_BASE_URL}/detect_anomaly`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query_image_path: inputImagePath,
                reference_image_path: normalImagePath,
                product: product
            })
        });

        const data = await response.json();

        // 결과 표시
        document.getElementById('detectingStatus').style.display = 'none';
        document.getElementById('resultSection').style.display = 'block';

        document.getElementById('normalImage').src = data.reference_image_url;
        document.getElementById('overlayImage').src = data.overlay_image_url;
        document.getElementById('anomalyScore').textContent = data.anomaly_score.toFixed(4);
        document.getElementById('anomalyJudgment').textContent = 
            data.is_anomaly ? '❌ 불량 검출' : '✅ 정상 범위';

        // 전역 변수에 저장 (매뉴얼 생성 시 사용)
        window.anomalyData = data;

    } catch (err) {
        console.error('[ANOMALY] 검출 실패:', err);
        alert('이상 검출 중 오류가 발생했습니다.');
    }
});

// 매뉴얼 생성 버튼 핸들러
document.getElementById('btnHyperClovax').addEventListener('click', () => {
    goToManual('llm');
});

document.getElementById('btnExaone').addEventListener('click', () => {
    goToManual('llm_exaone');
});

document.getElementById('btnVLM').addEventListener('click', () => {
    goToManual('vlm');
});

function goToManual(model) {
    // 데이터를 세션 스토리지에 저장
    sessionStorage.setItem('anomalyData', JSON.stringify(window.anomalyData));
    sessionStorage.setItem('selectedModel', model);
    
    window.location.href = 'manual.html';
}
```

---

#### 2.4 대응 매뉴얼 생성 화면 개선

**목표**: 3개 섹션 구성 + 작업자 입력 + DB 저장

**작업 내용:**

1. **새 HTML 파일**: `web/manual.html`
```html
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>대응 매뉴얼 생성</title>
    <link rel="stylesheet" href="/static/css/manual.css">
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 AI 대응 매뉴얼</h1>
        </div>

        <!-- 섹션 1: 이미지 비교 -->
        <div class="section image-section">
            <h2>🖼️ 이미지 비교</h2>
            <div class="image-grid">
                <div class="image-item">
                    <h3>정상 기준 (TOP-1)</h3>
                    <img id="top1Image" class="comparison-image">
                </div>
                <div class="image-item">
                    <h3>이상 영역 표시</h3>
                    <img id="segmentationImage" class="comparison-image">
                </div>
            </div>
        </div>

        <!-- 섹션 2: AI 생성 답변 -->
        <div class="section analysis-section">
            <h2>🧠 AI 분석 결과 (<span id="modelName"></span>)</h2>
            <div id="llmResponse" class="llm-content"></div>
            <p class="processing-time">처리 시간: <span id="processingTime"></span>초</p>
        </div>

        <!-- 섹션 3: 작업자 입력 -->
        <div class="section input-section">
            <h2>✍️ 작업자 조치 입력</h2>
            
            <div class="form-group">
                <label for="workerName">작업자명 *</label>
                <input type="text" id="workerName" placeholder="홍길동" required>
            </div>

            <div class="form-group">
                <label for="actionTaken">조치 내역 *</label>
                <textarea id="actionTaken" rows="5" 
                    placeholder="실제로 취한 조치 사항을 상세히 기록하세요..."
                    required></textarea>
            </div>

            <div class="form-group">
                <label>피드백 점수 *</label>
                <div class="rating-group">
                    <label><input type="radio" name="feedback" value="1"> 1점 (매우 나쁨)</label>
                    <label><input type="radio" name="feedback" value="2"> 2점 (나쁨)</label>
                    <label><input type="radio" name="feedback" value="3"> 3점 (보통)</label>
                    <label><input type="radio" name="feedback" value="4"> 4점 (좋음)</label>
                    <label><input type="radio" name="feedback" value="5"> 5점 (매우 좋음)</label>
                </div>
            </div>

            <button id="submitActionBtn" class="submit-button">
                💾 조치내역 등록
            </button>
        </div>
    </div>

    <script src="/static/js/manual.js"></script>
</body>
</html>
```

2. **JavaScript**: `web/static/js/manual.js`
```javascript
document.addEventListener('DOMContentLoaded', async () => {
    // 세션 스토리지에서 데이터 복원
    const anomalyData = JSON.parse(sessionStorage.getItem('anomalyData'));
    const selectedModel = sessionStorage.getItem('selectedModel');

    // 이미지 표시
    document.getElementById('top1Image').src = anomalyData.reference_image_url;
    document.getElementById('segmentationImage').src = anomalyData.overlay_image_url;

    // 모델명 표시
    const modelNames = {
        'llm': 'HyperCLOVAX',
        'llm_exaone': 'EXAONE 3.5',
        'vlm': 'LLaVA'
    };
    document.getElementById('modelName').textContent = modelNames[selectedModel];

    // LLM 호출
    try {
        const endpoint = selectedModel === 'vlm' 
            ? '/manual/generate/vlm'
            : selectedModel === 'llm_exaone'
            ? '/manual/generate/llm_exaone'
            : '/manual/generate/llm';

        const response = await fetch(`${API_BASE_URL}${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image_path: anomalyData.query_image_path,
                product: anomalyData.product,
                defect: anomalyData.defect,
                anomaly_score: anomalyData.anomaly_score,
                is_anomaly: anomalyData.is_anomaly
            })
        });

        const data = await response.json();

        // AI 답변 표시
        const analysisKey = selectedModel === 'vlm' ? 'vlm_analysis' : 'llm_analysis';
        document.getElementById('llmResponse').innerHTML = 
            data[analysisKey].replace(/\n/g, '<br>');
        document.getElementById('processingTime').textContent = 
            data.processing_time;

        // 전역 변수에 저장 (DB 저장 시 사용)
        window.manualData = data;

    } catch (err) {
        console.error('[MANUAL] 생성 실패:', err);
        alert('매뉴얼 생성 중 오류가 발생했습니다.');
    }
});

// 조치내역 등록
document.getElementById('submitActionBtn').addEventListener('click', async () => {
    const workerName = document.getElementById('workerName').value.trim();
    const actionTaken = document.getElementById('actionTaken').value.trim();
    const feedbackScore = document.querySelector('input[name="feedback"]:checked')?.value;

    // 유효성 검사
    if (!workerName || !actionTaken || !feedbackScore) {
        alert('모든 필수 항목을 입력해주세요.');
        return;
    }

    // DB 저장 요청
    try {
        const response = await fetch(`${API_BASE_URL}/history/save`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                search_id: generateSearchId(),
                product_name: window.manualData.product,
                defect_name: window.manualData.defect_ko,
                input_image_path: window.anomalyData.query_image_path,
                top1_image_path: window.anomalyData.reference_image_path,
                model_used: sessionStorage.getItem('selectedModel'),
                llm_response: document.getElementById('llmResponse').innerText,
                processing_time: window.manualData.processing_time,
                has_feedback: true,
                worker_name: workerName,
                action_taken: actionTaken,
                feedback_score: parseInt(feedbackScore),
                anomaly_score: window.anomalyData.anomaly_score,
                is_anomaly: window.anomalyData.is_anomaly
            })
        });

        if (response.ok) {
            alert('✅ 조치내역이 성공적으로 등록되었습니다.');
            window.location.href = 'dashboard.html';
        } else {
            throw new Error('저장 실패');
        }

    } catch (err) {
        console.error('[SAVE] 저장 실패:', err);
        alert('❌ 저장 중 오류가 발생했습니다.');
    }
});

function generateSearchId() {
    return `SEARCH_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}
```

---

### 🗄️ 3. 데이터베이스 구현 (우선순위 2)

**작업 순서:**

1. **PostgreSQL 설치 및 설정**
```bash
# Rocky Linux
sudo dnf install postgresql-server postgresql-contrib
sudo postgresql-setup --initdb
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

2. **데이터베이스 생성**
```sql
CREATE DATABASE defect_analysis;
\c defect_analysis

CREATE TABLE defect_analysis_history (
    id SERIAL PRIMARY KEY,
    search_id VARCHAR(100) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    product_name VARCHAR(50) NOT NULL,
    defect_name VARCHAR(50) NOT NULL,
    input_image_path VARCHAR(255) NOT NULL,
    top1_image_path VARCHAR(255) NOT NULL,
    model_used VARCHAR(20) NOT NULL CHECK (model_used IN ('llm', 'llm_exaone', 'vlm')),
    llm_response TEXT NOT NULL,
    processing_time FLOAT NOT NULL,
    has_feedback BOOLEAN DEFAULT FALSE,
    worker_name VARCHAR(100),
    action_taken TEXT,
    feedback_score INT CHECK (feedback_score BETWEEN 1 AND 5),
    anomaly_score FLOAT,
    is_anomaly BOOLEAN
);

CREATE INDEX idx_created_at ON defect_analysis_history(created_at DESC);
CREATE INDEX idx_product_defect ON defect_analysis_history(product_name, defect_name);
CREATE INDEX idx_model ON defect_analysis_history(model_used);
```

3. **API 서버에 DB 연결 추가** (`web/api_server.py`)
```python
import asyncpg
from datetime import datetime

# DB 연결 풀
db_pool = None

@app.on_event("startup")
async def init_db():
    global db_pool
    db_pool = await asyncpg.create_pool(
        host='localhost',
        port=5432,
        user='postgres',
        password='your_password',
        database='defect_analysis',
        min_size=5,
        max_size=20
    )

@app.on_event("shutdown")
async def close_db():
    await db_pool.close()

# 저장 API
@app.post("/history/save")
async def save_history(data: dict):
    async with db_pool.acquire() as conn:
        await conn.execute('''
            INSERT INTO defect_analysis_history (
                search_id, product_name, defect_name,
                input_image_path, top1_image_path,
                model_used, llm_response, processing_time,
                has_feedback, worker_name, action_taken,
                feedback_score, anomaly_score, is_anomaly
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
        ''', 
            data['search_id'], data['product_name'], data['defect_name'],
            data['input_image_path'], data['top1_image_path'],
            data['model_used'], data['llm_response'], data['processing_time'],
            data['has_feedback'], data['worker_name'], data['action_taken'],
            data['feedback_score'], data['anomaly_score'], data['is_anomaly']
        )
    
    return {"status": "success", "search_id": data['search_id']}

# 조회 API
@app.get("/history/list")
async def get_history(page: int = 1, per_page: int = 20):
    offset = (page - 1) * per_page
    
    async with db_pool.acquire() as conn:
        rows = await conn.fetch('''
            SELECT * FROM defect_analysis_history
            ORDER BY created_at DESC
            LIMIT $1 OFFSET $2
        ''', per_page, offset)
        
        total = await conn.fetchval('SELECT COUNT(*) FROM defect_analysis_history')
    
    return {
        "data": [dict(row) for row in rows],
        "total": total,
        "page": page,
        "per_page": per_page
    }
```

---

## 📚 참고 정보

### 현재 프로젝트 상태

**구현 완료:**
- ✅ CLIP 유사도 검색 (TOP-K)
- ✅ PatchCore 이상 검출
- ✅ 3개 LLM/VLM 모델 (HyperCLOVAX, EXAONE, LLaVA)
- ✅ RAG 매뉴얼 검색
- ✅ 4개 섹션 표준 출력
- ✅ 웹 UI (탭 기반)

**미구현:**
- ❌ 인덱스 자동 전환 (긴급)
- ❌ 화면 분리 (Phase 1)
- ❌ 데이터베이스 저장
- ❌ 대시보드

### 디렉토리 구조
```
llm_chal_vlm/
├── llm_server/
│   └── llm_server.py           # LLM 서버 (포트 5001)
├── web/
│   ├── api_server.py           # API 서버 (포트 5000)
│   ├── matching.html           # 현재 통합 UI
│   └── static/
│       ├── css/
│       └── js/
├── modules/
│   ├── similarity_matcher.py   # CLIP
│   ├── anomaly_detector.py     # PatchCore
│   └── vlm/
│       ├── rag.py
│       └── defect_mapper.py
├── data/
│   ├── def_split/              # 불량 이미지 (유사도 검색용)
│   └── ok_split/               # 정상 이미지 (이상 검출용)
└── markdown/
    ├── project_status.md       # 프로젝트 현황
    └── session_handover.md     # 이 파일
```

### 환경 정보

**서버:**
- OS: Rocky Linux 8.10
- GPU: Tesla T4
- Python: 3.9

**포트:**
- ALB: 80 → Backend 5000
- API 서버: 5000
- LLM 서버: 5001
- NLB SSH: 2022

**모델:**
- HyperCLOVAX: FP16
- EXAONE 3.5: BF16
- LLaVA: FP16

### 서버 실행
```bash
# LLM 서버
cd llm_server
python llm_server.py  # 포트 5001

# API 서버
cd web
python api_server.py  # 포트 5000
```

### 주요 API 엔드포인트

**LLM 서버 (5001):**
- `POST /analyze` - HyperCLOVAX
- `POST /analyze_exaone` - EXAONE 3.5
- `POST /analyze_vlm` - LLaVA

**API 서버 (5000):**
- `POST /search` - 유사도 검색
- `POST /detect_anomaly` - 이상 검출
- `POST /manual/generate/llm` - LLM 매뉴얼
- `POST /manual/generate/llm_exaone` - EXAONE 매뉴얼
- `POST /manual/generate/vlm` - VLM 매뉴얼
- `POST /index/switch` - 인덱스 전환 (추가 예정)
- `POST /history/save` - 이력 저장 (추가 예정)

---

## ⚠️ 주의사항

1. **인덱스 전환 기능을 최우선으로 구현하세요**
   - 현재 정확도 문제의 주요 원인
   - 다른 작업보다 먼저 처리 필요

2. **세션 스토리지 활용**
   - 화면 간 데이터 전달에 `sessionStorage` 사용
   - 페이지 새로고침 시 데이터 유지 가능

3. **Git 커밋 메시지 규칙**
```
   feat: 인덱스 자동 전환 기능 추가
   fix: 유사도 검색 버그 수정
   refactor: UI 화면 분리
   docs: 문서 업데이트
```

4. **테스트 시나리오**
   - 각 화면 단위로 독립적으로 테스트
   - 화면 간 데이터 전달 확인
   - 인덱스 전환 정상 작동 확인

---

## 📞 연락처

**개발자**: dhkim@dmillions.co.kr  
**GitHub**: https://github.com/scschwan/llm_chal_vlm  

---

**작성자**: Claude Sonnet 4.5  
**다음 세션**: 인덱스 자동 전환 → Phase 1 UI 개선 → DB 구축 순서로 진행하세요.
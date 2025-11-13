
## 📌 핵심 개념

**제품 단위 관리**: 제품명(product_id) 기준으로 전체 불량 목록을 관리
- 제품 생성 시 매뉴얼 DOCX 파일에서 자동으로 불량 추출
- en, ko만 저장 (keywords는 코드에서 자동 확장)

## 🚀 빠른 시작

### 초기화

```python
from pathlib import Path
from defect_mapping_manager import DefectMappingManager

manager = DefectMappingManager(
    mapping_file_path=Path("web/defect_mapping.json"),
    verbose=True  # 로그 출력
)
```

## 📝 주요 메서드

### 1. 제품 생성

```python
# 방법1: 매뉴얼 없이 빈 제품 생성
manager.create_product(
    product_id="prod5",
    product_name_ko="새제품"
)

# 방법2: 매뉴얼 파일에서 자동 추출 (권장)
manager.create_product(
    product_id="prod5",
    product_name_ko="새제품",
    manual_docx_path=Path("manual_store/prod5_manual.docx")
)
```

**반환값**: `True` (성공) / `False` (실패 - 이미 존재)

---

### 2. 배치 생성 (디렉토리 전체)

```python
# manual_store 디렉토리의 모든 DOCX 파일 처리
created_count = manager.batch_create_from_directory(
    manual_dir=Path("manual_store"),
    product_name_mapping={
        "prod1_menual.docx": "주조제품",
        "grid_manual.docx": "그리드",
        "carpet_manual.docx": "카펫",
        "leather_manual.docx": "가죽"
    }
)

print(f"{created_count}개 제품 생성됨")
```

**자동 처리**:
- 파일명에서 제품 ID 추출 (예: prod1_menual.docx → prod1)
- DOCX에서 불량 유형 자동 파싱
- 이미 존재하는 제품은 스킵

---

### 3. 제품 업데이트

```python
# 시나리오 A: 매뉴얼 재등록 (기존 불량 유지 + 새 불량 추가)
manager.update_product(
    product_id="prod1",
    manual_docx_path=Path("manual_store/prod1_manual_v2.docx"),
    merge_defects=True  # 기존 불량 유지
)

# 시나리오 B: 매뉴얼 재등록 (기존 불량 삭제하고 완전 교체)
manager.update_product(
    product_id="prod1",
    manual_docx_path=Path("manual_store/prod1_manual_v2.docx"),
    merge_defects=False  # 완전 교체
)

# 시나리오 C: 제품명만 변경
manager.update_product(
    product_id="prod1",
    product_name_ko="주조제품 (신규)"
)
```

**merge_defects 파라미터**:
- `True`: 기존 불량 + 새 불량 (추가 방식)
- `False`: 기존 불량 삭제 후 새 불량만 (교체 방식)

---

### 4. 제품 삭제

```python
manager.delete_product("prod1")
```

**주의**: 제품과 모든 불량 정보가 함께 삭제됩니다.

---

### 5. 제품 조회

```python
# 단일 제품 정보
product_info = manager.get_product("prod1")
# {
#   "name_ko": "주조제품",
#   "defects": {
#     "hole": {"en": "hole", "ko": "기공"},
#     ...
#   }
# }

# 전체 제품 목록
products = manager.list_products()
# ['prod1', 'grid', 'carpet', 'leather']

# 전체 요약 출력
manager.print_summary()
```

---

### 6. 불량 개별 관리

```python
# 불량 추가
manager.add_defect(
    product_id="prod1",
    defect_en="crack",
    defect_ko="균열"
)

# 불량 수정 (한글명만 변경)
manager.update_defect(
    product_id="prod1",
    defect_en="crack",
    defect_ko="크랙"
)

# 불량 삭제
manager.delete_defect(
    product_id="prod1",
    defect_en="crack"
)
```

---

## 🎯 실전 시나리오

### 시나리오 1: 신규 제품 등록 (관리자 페이지)

```python
# 사용자가 제품명 + DOCX 파일 업로드
async def admin_create_product(
    product_id: str,
    product_name_ko: str,
    uploaded_file: UploadFile
):
    # 1. 임시 파일 저장
    temp_path = Path(f"/tmp/{uploaded_file.filename}")
    with open(temp_path, "wb") as f:
        f.write(await uploaded_file.read())
    
    # 2. 제품 생성
    success = manager.create_product(
        product_id=product_id,
        product_name_ko=product_name_ko,
        manual_docx_path=temp_path
    )
    
    # 3. 임시 파일 삭제
    temp_path.unlink()
    
    return {"success": success}
```

---

### 시나리오 2: 매뉴얼 업데이트

```python
# 사용자가 새 매뉴얼 업로드
async def admin_update_manual(
    product_id: str,
    uploaded_file: UploadFile,
    replace_all: bool = False  # True=교체, False=추가
):
    temp_path = Path(f"/tmp/{uploaded_file.filename}")
    with open(temp_path, "wb") as f:
        f.write(await uploaded_file.read())
    
    success = manager.update_product(
        product_id=product_id,
        manual_docx_path=temp_path,
        merge_defects=not replace_all  # replace_all=True면 merge=False
    )
    
    temp_path.unlink()
    
    return {"success": success}
```

---

### 시나리오 3: 제품 삭제

```python
async def admin_delete_product(product_id: str):
    # 1. 제품 정보 백업 (선택)
    product_info = manager.get_product(product_id)
    
    # 2. 삭제
    success = manager.delete_product(product_id)
    
    return {"success": success}
```

---

## 📋 DOCX 매뉴얼 형식 요구사항

자동 추출이 동작하려면 매뉴얼이 다음 형식이어야 합니다:

```
1️⃣ hole (기공)
발생 원인
...

2️⃣ burr (날개 버)
발생 원인
...

3️⃣ Bent Defect (휨·압흔 불량)
발생 원인
...
```

**패턴**:
- `1️⃣ 영문명 (한글명)`
- `1️⃣ 영문명 Defect (한글명 불량)`

**지원 구분자**: `·`, `/`, `,` (한글명에서 자동 분리)

---

## ⚠️ 주의사항

1. **product_id는 고유해야 함**: 중복 생성 시 `False` 반환
2. **en(영문명)이 키로 사용됨**: 같은 제품 내에서 중복 불가
3. **DOCX 파일 필수**: 자동 추출을 사용하려면 올바른 형식의 DOCX 필요
4. **파일 저장 자동**: 모든 변경은 즉시 JSON 파일에 저장됨

---

## 🔧 트러블슈팅

### Q: 불량이 자동 추출되지 않아요
A: DOCX 형식 확인
```python
# 수동으로 추출 결과 확인
defects = manager.extract_defects_from_docx(Path("manual.docx"))
print(defects)  # [(en, ko), ...]
```

### Q: 기존 불량을 유지하고 싶어요
A: `merge_defects=True` 사용
```python
manager.update_product(..., merge_defects=True)
```

### Q: 전체 불량을 교체하고 싶어요
A: `merge_defects=False` 사용
```python
manager.update_product(..., merge_defects=False)
```

---

## 📊 JSON 구조 (참고)

```json
{
  "products": {
    "prod1": {
      "name_ko": "주조제품",
      "defects": {
        "hole": {
          "en": "hole",
          "ko": "기공"
        },
        "burr": {
          "en": "burr",
          "ko": "날개 버"
        }
      }
    }
  }
}
```

**간소화**: keywords, full_name_ko 제거됨 (코드에서 자동 처리)

EOF
cat /tmp/DEFECT_MAPPING_QUICKSTART.md
출력

# defect_mapping_manager.py 사용 가이드

## 📌 핵심 개념

**제품 단위 관리**: 제품명(product_id) 기준으로 전체 불량 목록을 관리
- 제품 생성 시 매뉴얼 DOCX 파일에서 자동으로 불량 추출
- en, ko만 저장 (keywords는 코드에서 자동 확장)

## 🚀 빠른 시작

### 초기화

```python
from pathlib import Path
from defect_mapping_manager import DefectMappingManager

manager = DefectMappingManager(
    mapping_file_path=Path("web/defect_mapping.json"),
    verbose=True  # 로그 출력
)
```

## 📝 주요 메서드

### 1. 제품 생성

```python
# 방법1: 매뉴얼 없이 빈 제품 생성
manager.create_product(
    product_id="prod5",
    product_name_ko="새제품"
)

# 방법2: 매뉴얼 파일에서 자동 추출 (권장)
manager.create_product(
    product_id="prod5",
    product_name_ko="새제품",
    manual_docx_path=Path("manual_store/prod5_manual.docx")
)
```

**반환값**: `True` (성공) / `False` (실패 - 이미 존재)

---

### 2. 배치 생성 (디렉토리 전체)

```python
# manual_store 디렉토리의 모든 DOCX 파일 처리
created_count = manager.batch_create_from_directory(
    manual_dir=Path("manual_store"),
    product_name_mapping={
        "prod1_menual.docx": "주조제품",
        "grid_manual.docx": "그리드",
        "carpet_manual.docx": "카펫",
        "leather_manual.docx": "가죽"
    }
)

print(f"{created_count}개 제품 생성됨")
```

**자동 처리**:
- 파일명에서 제품 ID 추출 (예: prod1_menual.docx → prod1)
- DOCX에서 불량 유형 자동 파싱
- 이미 존재하는 제품은 스킵

---

### 3. 제품 업데이트

```python
# 시나리오 A: 매뉴얼 재등록 (기존 불량 유지 + 새 불량 추가)
manager.update_product(
    product_id="prod1",
    manual_docx_path=Path("manual_store/prod1_manual_v2.docx"),
    merge_defects=True  # 기존 불량 유지
)

# 시나리오 B: 매뉴얼 재등록 (기존 불량 삭제하고 완전 교체)
manager.update_product(
    product_id="prod1",
    manual_docx_path=Path("manual_store/prod1_manual_v2.docx"),
    merge_defects=False  # 완전 교체
)

# 시나리오 C: 제품명만 변경
manager.update_product(
    product_id="prod1",
    product_name_ko="주조제품 (신규)"
)
```

**merge_defects 파라미터**:
- `True`: 기존 불량 + 새 불량 (추가 방식)
- `False`: 기존 불량 삭제 후 새 불량만 (교체 방식)

---

### 4. 제품 삭제

```python
manager.delete_product("prod1")
```

**주의**: 제품과 모든 불량 정보가 함께 삭제됩니다.

---

### 5. 제품 조회

```python
# 단일 제품 정보
product_info = manager.get_product("prod1")
# {
#   "name_ko": "주조제품",
#   "defects": {
#     "hole": {"en": "hole", "ko": "기공"},
#     ...
#   }
# }

# 전체 제품 목록
products = manager.list_products()
# ['prod1', 'grid', 'carpet', 'leather']

# 전체 요약 출력
manager.print_summary()
```

---

### 6. 불량 개별 관리

```python
# 불량 추가
manager.add_defect(
    product_id="prod1",
    defect_en="crack",
    defect_ko="균열"
)

# 불량 수정 (한글명만 변경)
manager.update_defect(
    product_id="prod1",
    defect_en="crack",
    defect_ko="크랙"
)

# 불량 삭제
manager.delete_defect(
    product_id="prod1",
    defect_en="crack"
)
```

---

## 🎯 실전 시나리오

### 시나리오 1: 신규 제품 등록 (관리자 페이지)

```python
# 사용자가 제품명 + DOCX 파일 업로드
async def admin_create_product(
    product_id: str,
    product_name_ko: str,
    uploaded_file: UploadFile
):
    # 1. 임시 파일 저장
    temp_path = Path(f"/tmp/{uploaded_file.filename}")
    with open(temp_path, "wb") as f:
        f.write(await uploaded_file.read())
    
    # 2. 제품 생성
    success = manager.create_product(
        product_id=product_id,
        product_name_ko=product_name_ko,
        manual_docx_path=temp_path
    )
    
    # 3. 임시 파일 삭제
    temp_path.unlink()
    
    return {"success": success}
```

---

### 시나리오 2: 매뉴얼 업데이트

```python
# 사용자가 새 매뉴얼 업로드
async def admin_update_manual(
    product_id: str,
    uploaded_file: UploadFile,
    replace_all: bool = False  # True=교체, False=추가
):
    temp_path = Path(f"/tmp/{uploaded_file.filename}")
    with open(temp_path, "wb") as f:
        f.write(await uploaded_file.read())
    
    success = manager.update_product(
        product_id=product_id,
        manual_docx_path=temp_path,
        merge_defects=not replace_all  # replace_all=True면 merge=False
    )
    
    temp_path.unlink()
    
    return {"success": success}
```

---

### 시나리오 3: 제품 삭제

```python
async def admin_delete_product(product_id: str):
    # 1. 제품 정보 백업 (선택)
    product_info = manager.get_product(product_id)
    
    # 2. 삭제
    success = manager.delete_product(product_id)
    
    return {"success": success}
```

---

## 📋 DOCX 매뉴얼 형식 요구사항

자동 추출이 동작하려면 매뉴얼이 다음 형식이어야 합니다:

```
1️⃣ hole (기공)
발생 원인
...

2️⃣ burr (날개 버)
발생 원인
...

3️⃣ Bent Defect (휨·압흔 불량)
발생 원인
...
```

**패턴**:
- `1️⃣ 영문명 (한글명)`
- `1️⃣ 영문명 Defect (한글명 불량)`

**지원 구분자**: `·`, `/`, `,` (한글명에서 자동 분리)

---

## ⚠️ 주의사항

1. **product_id는 고유해야 함**: 중복 생성 시 `False` 반환
2. **en(영문명)이 키로 사용됨**: 같은 제품 내에서 중복 불가
3. **DOCX 파일 필수**: 자동 추출을 사용하려면 올바른 형식의 DOCX 필요
4. **파일 저장 자동**: 모든 변경은 즉시 JSON 파일에 저장됨

---

## 🔧 트러블슈팅

### Q: 불량이 자동 추출되지 않아요
A: DOCX 형식 확인
```python
# 수동으로 추출 결과 확인
defects = manager.extract_defects_from_docx(Path("manual.docx"))
print(defects)  # [(en, ko), ...]
```

### Q: 기존 불량을 유지하고 싶어요
A: `merge_defects=True` 사용
```python
manager.update_product(..., merge_defects=True)
```

### Q: 전체 불량을 교체하고 싶어요
A: `merge_defects=False` 사용
```python
manager.update_product(..., merge_defects=False)
```

---

## 📊 JSON 구조 (참고)

```json
{
  "products": {
    "prod1": {
      "name_ko": "주조제품",
      "defects": {
        "hole": {
          "en": "hole",
          "ko": "기공"
        },
        "burr": {
          "en": "burr",
          "ko": "날개 버"
        }
      }
    }
  }
}
```

**간소화**: keywords, full_name_ko 제거됨 (코드에서 자동 처리)

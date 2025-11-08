# PatchCore Memory Bank – Quick Guide

이 문서는 두 스크립트 **`build_bank.py`** 와 **`infer.py`** 의 목적, 사용법, 입출력, 튜닝 팁을 정리한 README입니다.  
(환경: Rocky Linux 8.10, Python 3.9.x 기준)

---

## 개요

- **문제**: 제품·뷰·조명 별로 “정상(OK)” 분포가 달라서, 동일한 모델 한 개로 모든 경우를 잘 잡기 어렵다.
- **해결**: 제품/세팅별 **정상 이미지로 메모리뱅크(정상 패치 임베딩 분포)** 를 구축하고, 테스트 이미지를 해당 뱅크와 비교해 **이상 히트맵/마스크**를 생성한다.

### 구성 요약
- **`build_bank.py`**  
  정상 이미지 폴더(예: `./data/patchCore/prod1/ok`)에서 패치 임베딩을 추출 → **coreset 압축** →  
  OK 데이터로 **임계치(τ)** 를 캘리브레이트 → `memory_bank.pt`, `bank_config.json`, `tau.json` 저장
- **`infer.py`**  
  저장된 메모리뱅크를 로드 → 테스트 이미지의 패치 임베딩과 **최근접 거리(이상도)** 계산 →  
  **히트맵/마스크/오버레이** 파일로 저장 (+ 옵션으로 기준 이미지와 2×2 비교)

---

## 폴더 구조(권장)

```text
./data/
  patchCore/
    prod1/
      ok/                      # 정상 이미지(30~50장 이상 권장)
      memory_bank.pt           # build_bank.py 산출물
      bank_config.json         # build 시 사용한 파라미터 기록
      tau.json                 # 임계치(픽셀/이미지 레벨)
    prod2/
      ok/
      ...
  def_front/                   # (예) 테스트 이미지들
  ok_front/                    # (예) 기준 비교용 이미지들
```

> 원칙: **제품/뷰/세팅별로 폴더 분리** (카메라·조명·거리·면(front/back) 등 다르면 따로)

---

## 요구 사항

- Python 3.9.x
- PyTorch(쿠다 내장 휠 예: `+cu121` 또는 CPU), torchvision
- NumPy, OpenCV, scikit-learn(있으면 coreset KMeans 사용, 없으면 랜덤 샘플로 폴백)
- (선택) FAISS(있으면 KNN 가속, 없으면 NumPy 폴백)

예시 설치:
```bash
python -m pip install --upgrade pip
# GPU (CUDA 12.1 내장 휠 예시)
pip install "torch==2.7.1+cu121" "torchvision==0.22.1+cu121" --index-url https://download.pytorch.org/whl/cu121
# 또는 CPU 전용
# pip install "torch==2.7.1" "torchvision==0.22.1"

pip install numpy opencv-python scikit-learn
# 선택: pip install faiss-gpu  # 또는 conda로 설치 권장
```

---

## 1) 메모리뱅크 생성 — `build_bank.py`

### 실행 예시
```bash
python build_bank.py \
  --ok_dir ./data/patchCore/prod1/ok \
  --out_dir ./data/patchCore/prod1 \
  --shorter 512 --stride 2 --layers layer2 layer3 layer4 --coreset 0.02
```

### 주요 역할
1. 정상 이미지들을 짧은 변 기준 `--shorter` 크기로 리사이즈
2. ResNet50 **layer2/3/4** 멀티스케일 특징맵에서 **stride 간격**으로 패치 임베딩 추출
3. 전체 패치(수십~수백만 가능)를 **reservoir cap** 및 **coreset(압축)** 로 축소
4. OK 데이터에서 **픽셀/이미지 레벨 임계치(τ)** 를 **퍼센타일 기반**으로 산출
5. 결과 저장:
   - `memory_bank.pt` (float32 텐서 [M, D])
   - `bank_config.json` (리사이즈/stride/layers/coreset 등)
   - `tau.json` (`{"pixel": …, "image": …}`)

### 주요 인자
| 인자 | 설명 | 기본 |
|---|---|---|
| `--ok_dir` | 정상 이미지 폴더 | (필수) |
| `--out_dir` | 산출물 저장 폴더 | (필수) |
| `--shorter` | 짧은 변 리사이즈 크기 | 512 |
| `--stride` | 패치 샘플 간격(작을수록 촘촘) | 2 |
| `--layers` | 사용 레이어(복수) | `layer2 layer3 layer4` |
| `--coreset` | 압축 비율(0~1, 0은 무압축) | 0.02 |
| `--reservoir_max` | 스트리밍 수집 최대 패치 수 | 100000 |
| `--pixel_pct` | 픽셀 스코어 퍼센타일 | 0.995 (99.5%) |
| `--image_pct` | 이미지 스코어 퍼센타일 | 0.995 (99.5%) |

> ⚠️ 이미지 수가 많거나 `--stride 1`이면 패치가 급증 → **coreset 0.01~0.02** 권장

---

## 2) 단일 이미지 추론 — `infer.py`

### 실행 예시
```bash
python infer.py \
  --bank_dir ./data/patchCore/prod1 \
  --img ./data/def_front/cast_def_test.jpeg \
  --out_dir ./out_prod1 \
  --fuse_ref ./data/ok_front/cast_ok_test.jpeg   # (옵션) 비교용 2x2 타일 생성
```

### 주요 역할
1. `bank_dir`에서 `memory_bank.pt`, `bank_config.json`, `tau.json` 로드
2. 테스트 이미지에서 **패치 임베딩 추출** → 메모리뱅크와 **최근접 거리** 계산
3. 거리 맵을 원본 크기에 맞게 리사이즈 → **heatmap(0~1)** 정규화
4. `tau.json`의 픽셀 τ를 스케일에 맞게 변환해 **이진 마스크** 생성
5. 결과 저장:
   - `heat_gray.png` (0~255)
   - `mask.png` (이진 마스크)
   - `overlay.png` (원본 위 빨간색 오버레이)
   - `side_by_side.png` (옵션: 기준/테스트/마스크/오버레이 2×2)
   - `scores.json` (`{"image_score": …, "pixel_tau": …, "image_tau": …}`)

### 주요 인자
| 인자 | 설명 | 기본 |
|---|---|---|
| `--bank_dir` | 메모리뱅크 폴더 | (필수) |
| `--img` | 테스트 이미지 경로 | (필수) |
| `--out_dir` | 출력 폴더 | (필수) |
| `--fuse_ref` | (옵션) 기준 이미지 경로(2×2 비교) | None |

---

## 동작 원리(간단 설명)

1. **특징 추출**  
   ResNet50의 중간 레이어(layer2/3/4)를 upsample하여 같은 해상도로 맞춘 뒤 **채널 concat** →  
   stride 간격으로 **[N, D] 패치 임베딩** 생성

2. **정상 분포(메모리뱅크)**  
   여러 정상 이미지의 패치 임베딩을 모아 **정상 분포**를 근사(coreset로 크기 축소)

3. **이상도 계산**  
   테스트 패치 임베딩과 뱅크 간 **최근접 거리(L2)** 를 계산 →  
   거리 클수록 **정상에서 멀다 = 이상 가능성↑**

4. **캘리브레이션 및 마스크**  
   OK 데이터에서 얻은 거리 분포의 **퍼센타일(예: 99.5%)** 를 τ로 사용 →  
   테스트 거리 맵의 **τ 초과** 부분만 마스크로 표시

---

## 튜닝 팁

- **내부 미세 결함(크레이터) 민감도 올리기**
  - `--shorter` 640~768 ← 해상도 상향
  - `--stride` 1 ← 촘촘한 패치(속도·메모리↑)
  - 레이어에 **`layer2`** 가 포함되어야 텍스처 민감도 상승
- **오검출 줄이기**
  - `--pixel_pct` / `--image_pct` 를 **상향**(예: 0.997~0.999)
  - 마스크 후처리에 **모폴로지(열림/닫힘) + 작은영역 제거** 추가(필요 시 코드 보강)
- **메모리/속도 절충**
  - `--coreset 0.01~0.02` 유지
  - `--stride` 2 또는 3
  - reservoir cap(`--reservoir_max`) 조절

---

## 멀티-뱅크 운영 가이드

- 제품/뷰/세팅별로 **폴더 분리** 후 각 폴더에서 뱅크 생성
- 추론 시에는 **필요한 뱅크만 로드** (전부 상주 X)  
  서버 구성 시 **LRU 캐시(최근 2~3개 뱅크만 메모리 유지)** 권장
- (향후) 전역 DINO/CLIP 임베딩으로 **자동 라우팅**도 가능하나,  
  도입 초반엔 **명시적 bank 지정**이 운영·추적에 유리

---

## 자주 만나는 오류 & 해결

- **`permute ... input.dim() != len(dims)`**  
  → `feat_to_patches` 인덱싱을 **[H,W,C] → [N,C]** 로 평탄화하는 현재 버전으로 수정(이미 반영됨).
- **메모리 부족**  
  → `--stride` 증가, `--shorter` 감소, `--coreset`↑, reservoir cap↓
- **FAISS 없음**  
  → 자동으로 NumPy KNN 폴백(속도↓). 속도 필요 시 `faiss-gpu`(conda 권장) 설치.

---

## 빠른 시작(요약)

1) **메모리뱅크 생성**
```bash
python build_bank.py \
  --ok_dir ./data/patchCore/prod1/ok \
  --out_dir ./data/patchCore/prod1 \
  --shorter 512 --stride 2 --layers layer2 layer3 layer4 --coreset 0.02
```

2) **추론**
```bash
python infer.py \
  --bank_dir ./data/patchCore/prod1 \
  --img ./data/def_front/cast_def_test.jpeg \
  --out_dir ./out_prod1 \
  --fuse_ref ./data/ok_front/cast_ok_test.jpeg
```

출력: `heat_gray.png`, `mask.png`, `overlay.png`, (옵션) `side_by_side.png`, `scores.json`

---
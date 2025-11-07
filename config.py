from __future__ import annotations
from pathlib import Path

# 프로젝트 루트 기준으로 data/ 결과/ 등 경로 세팅
PROJECT_ROOT = Path(__file__).parent.resolve()

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

# 🔁 현재 파이프라인 요구
#   - 왼쪽(정상): ok_front
#   - 오른쪽(후보/불량): def_front
OK_DIR  = DATA_DIR / "ok_front"
DEF_DIR = DATA_DIR / "def_front"

# 좌우 캡션에 찍어줄 라벨(출력용)
OK_LABEL  = "ok_front"
DEF_LABEL = "def_front"

# LLaVA 계열 모델 경로/이름
# 로컬 HF 가중치 사용 형태 (예: llava-hf/llava-v1.6-mistral-7b-hf)
LLAVA_MODEL = "llava-hf/llava-v1.6-mistral-7b-hf"

# CLIP backbone
CLIP_MODEL = "openai/clip-vit-large-patch14"

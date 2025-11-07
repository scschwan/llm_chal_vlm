from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# (SAM/CLIPSeg 통합 전의 베이스 버전)
# fixed_objects.json 예시:
# {
#   "objects": [
#     { "name": "메인부품", "polygon": [[0.2,0.2],[0.8,0.2],[0.8,0.8],[0.2,0.8]], "weight": 1.5 }
#   ]
# }


def load_object_json(json_path: str | Path) -> Dict:
    p = Path(json_path)
    if not p.exists():
        return {"objects": []}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"objects": []}


def _poly_to_mask(shape_hw: Tuple[int, int], polygon_xy: List[List[float]]) -> np.ndarray:
    """
    polygon_xy: [[x_norm, y_norm], ...] 가정 (0~1)
    """
    h, w = shape_hw
    pts = [(int(x * w), int(y * h)) for (x, y) in polygon_xy]
    img = Image.new("L", (w, h), 0)
    ImageDraw.Draw(img).polygon(pts, fill=255)
    return (np.array(img, dtype=np.uint8) > 0)


def build_weight_map(img: Image.Image, cfg: Dict) -> np.ndarray:
    """
    polygon 기반 ROI에 가중치 부여.
    SAM/CLIPSeg 통합 시엔 여기서 mask union/blur 해서 반영해주면 됨.
    반환 shape: (H, W) float32
    """
    arr = np.asarray(img.convert("RGB"))
    h, w = arr.shape[:2]

    weights = np.ones((h, w), dtype=np.float32)

    for obj in cfg.get("objects", []):
        poly = obj.get("polygon")
        wt = float(obj.get("weight", 1.0))
        if not poly or wt <= 0:
            continue
        try:
            mask = _poly_to_mask((h, w), poly)
            weights[mask] *= wt
        except Exception:
            continue

    # clip to a reasonable range
    return np.clip(weights, 0.5, 5.0).astype("float32")


def build_prompt_hint(cfg: Dict) -> str:
    """
    ROI 힌트를 프롬프트에 주입해서
    "이런 파트를 잘 봐줘" 라고 VLM한테 살짝 알려주는 용도.
    """
    objs = cfg.get("objects", [])
    if not objs:
        return ""
    parts = []
    for o in objs:
        name = str(o.get("name", "")).strip()
        wt = o.get("weight", 1.0)
        if name:
            if wt and wt != 1.0:
                parts.append(f"{name}(가중치 {wt:g})")
            else:
                parts.append(name)

    return "ROI 우선 영역: " + ", ".join(parts) if parts else ""

# modules/region_detect.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Literal, Dict
import numpy as np
import cv2
from PIL import Image, ImageDraw


@dataclass
class Box:
    x: int
    y: int
    w: int
    h: int
    score: float  # 면적 기반 점수(간단화)

    def area(self) -> int:
        return int(self.w * self.h)


def _to_gray_u8(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)


def _clamp_box(x: int, y: int, w: int, h: int, W: int, H: int) -> Tuple[int, int, int, int]:
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    return x, y, w, h


def _expand_box(b: Box, W: int, H: int, pad_ratio: float) -> Box:
    if pad_ratio <= 0:
        return b
    pad_w = int(round(b.w * pad_ratio))
    pad_h = int(round(b.h * pad_ratio))
    x, y, w, h = _clamp_box(b.x - pad_w, b.y - pad_h, b.w + 2 * pad_w, b.h + 2 * pad_h, W, H)
    return Box(x, y, w, h, b.score)


def _iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a.x, a.y, a.x + a.w, a.y + a.h
    bx1, by1, bx2, by2 = b.x, b.y, b.x + b.w, b.y + b.h
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = a.area() + b.area() - inter
    return inter / union if union > 0 else 0.0


def _nms(boxes: List[Box], iou_thr: float) -> List[Box]:
    boxes = sorted(boxes, key=lambda b: b.score, reverse=True)
    keep: List[Box] = []
    for b in boxes:
        if all(_iou(b, k) < iou_thr for k in keep):
            keep.append(b)
    return keep


def diff_mask(
    left: Image.Image,
    right: Image.Image,
    blur_ksize: int = 3,
    thresh_mode: Literal["auto", "fixed", "percentile"] = "auto",
    thresh_val: int = 25,
    perc: float = 0.85,
) -> np.ndarray:
    """
    좌/우 이미지를 비교해 0/255 바이너리 마스크 생성
    """
    g1 = _to_gray_u8(left)
    g2 = _to_gray_u8(right)
    d = cv2.absdiff(g1, g2)
    if blur_ksize > 0:
        d = cv2.GaussianBlur(d, (blur_ksize, blur_ksize), 0)

    if thresh_mode == "auto":
        # ✅ Otsu는 바로 마스크를 반환 (임계값 따로 안 씀)
        _, mask = cv2.threshold(d, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        if thresh_mode == "percentile":
            t = int(np.clip(np.quantile(d, perc), 0, 255))
        else:  # "fixed"
            t = int(thresh_val)
        _, mask = cv2.threshold(d, t, 255, cv2.THRESH_BINARY)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8), iterations=1)
    return mask


def extract_rois(
    left: Image.Image,
    right: Image.Image,
    *,
    topk: int = 5,
    min_area_ratio: float = 0.001,     # 전체의 0.1% 미만 무시
    max_area_ratio: float = 0.20,      # 단일 ROI 20% 초과면 컷
    pad_ratio: float = 0.06,           # 박스 확장 비율
    iou_thr: float = 0.5,              # NMS 임계
    max_coverage_ratio: float = 0.25,  # 누적 커버리지 상한(25%)
    thresh_mode: Literal["auto", "fixed", "percentile"] = "auto",
    thresh_val: int = 25,
    perc: float = 0.85,
) -> Tuple[List[Box], Dict]:
    """
    차분 마스크 → 컨투어 → Box 추출 → NMS → 누적커버리지 제한 → 상위 k 반환
    stats: 총 픽셀수, 마스크 픽셀수, 커버리지 비율, ROI별 면적 비율 등
    """
    W, H = left.size
    total_px = W * H

    mask = diff_mask(left, right, blur_ksize=3, thresh_mode=thresh_mode, thresh_val=thresh_val, perc=perc)
    mask_px = int(np.count_nonzero(mask))
    mask_ratio = float(mask_px) / float(total_px)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes_raw: List[Box] = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        # 면적 제한
        if area < int(total_px * min_area_ratio) or area > int(total_px * max_area_ratio):
            continue
        boxes_raw.append(Box(x, y, w, h, float(area)))

    # 패딩 & NMS
    boxes_raw = [_expand_box(b, W, H, pad_ratio) for b in boxes_raw]
    boxes_nms = _nms(boxes_raw, iou_thr=iou_thr)
    boxes_nms.sort(key=lambda b: b.score, reverse=True)

    # 누적 커버리지 제한 + TopK
    selected: List[Box] = []
    cover_mask = np.zeros((H, W), dtype=np.uint8)
    cover_px = 0
    for b in boxes_nms:
        x1, y1, x2, y2 = b.x, b.y, b.x + b.w, b.y + b.h
        cover_mask[y1:y2, x1:x2] = 1
        new_cover_px = int(cover_mask.sum())
        new_ratio = new_cover_px / float(total_px)
        if new_ratio > max_coverage_ratio:
            cover_mask[y1:y2, x1:x2] = 0
            continue
        selected.append(b)
        cover_px = new_cover_px
        if len(selected) >= topk:
            break

    coverage_ratio = cover_px / float(total_px)
    stats = {
        "W": W, "H": H, "total_px": total_px,
        "mask_px": mask_px, "mask_ratio": mask_ratio,  # 차분 마스크 비율
        "roi_count": len(selected),
        "coverage_px": cover_px, "coverage_ratio": coverage_ratio,  # 선택 ROI 누적 커버리지
        "per_roi_ratio": [b.area() / float(total_px) for b in selected],
        "params": {
            "topk": topk, "min_area_ratio": min_area_ratio, "max_area_ratio": max_area_ratio,
            "pad_ratio": pad_ratio, "iou_thr": iou_thr, "max_coverage_ratio": max_coverage_ratio,
            "thresh_mode": thresh_mode, "thresh_val": thresh_val, "perc": perc,
        }
    }
    return selected, stats


def overlay_boxes(img: Image.Image, boxes: List[Box], color=(255, 0, 0), width=3) -> Image.Image:
    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    for i, b in enumerate(boxes, 1):
        draw.rectangle([b.x, b.y, b.x + b.w, b.y + b.h], outline=color, width=width)
        draw.text((b.x + 3, b.y + 3), f"ROI{i}", fill=color)
    return out


def boxes_to_prompt_hints(boxes: List[Box], tag: str, total_px: int) -> List[str]:
    hints: List[str] = []
    for i, b in enumerate(boxes, 1):
        pct = 100.0 * (b.area() / float(total_px))
        hints.append(f"{tag}{i}: x={b.x}, y={b.y}, w={b.w}, h={b.h}, area={b.area()}px (~{pct:.2f}%)")
        return hints

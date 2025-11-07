from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


@dataclass
class PreprocessCfg:
    brightness: float
    contrast: float
    gamma: float
    normalize_encoder: bool
    weight_map: np.ndarray
    handle_alpha: bool
    alpha_bg: Tuple[int,int,int]
    use_local_contrast: bool
    unsharp_amount: float
    unsharp_radius: float
    unsharp_threshold: float


def apply_preprocess(img: Image.Image, cfg: PreprocessCfg) -> Image.Image:
    """
    - 밝기/대비 약간 보정
    - 알파 있을 경우 BG merge
    - 샤픈(언샵) 살짝
    - weight_map은 (나중에 encoder stage에서 쓰거나 attention bias 용)
      여기서는 이미지 픽셀 자체는 그대로 두고, cfg에만 담겨 있는 상태로 넘어간다고 보면 됨.
    """
    im = img.convert("RGBA") if cfg.handle_alpha else img.convert("RGB")

    if cfg.handle_alpha:
        bg = Image.new("RGB", im.size, cfg.alpha_bg)
        bg.paste(im, mask=im.split()[-1])  # alpha merge
        im = bg

    # 밝기/대비
    if cfg.brightness != 1.0:
        im = ImageEnhance.Brightness(im).enhance(cfg.brightness)
    if cfg.contrast != 1.0:
        im = ImageEnhance.Contrast(im).enhance(cfg.contrast)

    # (선택) 샤픈 느낌 비슷하게
    if cfg.unsharp_amount > 0:
        im = im.filter(ImageFilter.UnsharpMask(
            radius=cfg.unsharp_radius,
            percent=int(cfg.unsharp_amount * 100),
            threshold=int(cfg.unsharp_threshold),
        ))

    # gamma 보정은 여기선 생략(원하면 np.array로 감싸서 pow 써도 됨)

    # normalize_encoder / weight_map은 실제 encoder에서 활용.
    # 지금 단계에서는 이미지 자체만 반환.
    return im

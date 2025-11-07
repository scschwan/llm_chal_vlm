from __future__ import annotations
from typing import Dict, List
from PIL import Image
import numpy as np

def compare_deformation(left_img: Image.Image, right_img: Image.Image) -> Dict:
    """
    두 이미지의 외곽형/변형 관련 정량 힌트를 리턴.
    실제 구현은 contour 추출 후 solidity, dent amount 등 계산.
    여기서는 placeholder 리턴.
    """

    # placeholder 계산
    solidity_L = 0.8
    solidity_R = 0.75
    dent_delta = +0.05
    edge_delta = 0.00

    # hotspot 후보 (ssim_grid_hints랑 유사)
    hotspots: List[str] = ["중단 좌측", "하단 중앙"]

    return {
        "solidity_L": solidity_L,
        "solidity_R": solidity_R,
        "delta_dent": dent_delta,
        "delta_edge": edge_delta,
        "hotspots": hotspots,
    }

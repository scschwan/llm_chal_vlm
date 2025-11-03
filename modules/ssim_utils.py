from __future__ import annotations
from typing import List, Tuple
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

def _to_gray_np(im: Image.Image) -> np.ndarray:
    return np.asarray(im.convert("L"), dtype=np.float32) / 255.0


def ssim_global(left_img: Image.Image, right_img: Image.Image) -> float:
    """
    전체 이미지 단위 SSIM (흑백 변환 후)
    """
    a = _to_gray_np(left_img)
    b = _to_gray_np(right_img)

    # resize to min common?
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    a = a[:h,:w]
    b = b[:h,:w]

    val, _ = ssim(a, b, full=True)
    return float(val)


def ssim_grid_hints(
    left_img: Image.Image,
    right_img: Image.Image,
    grid: Tuple[int,int] = (3,3),
    topk: int = 5,
) -> List[str]:
    """
    이미지를 (gh x gw) 격자로 나누고, 각 격자 블록별 SSIM을 비교해서
    SSIM이 낮은 (=차이가 큰) 구역 이름을 힌트로 뽑는다.
    예: "하단 좌측", "중단 중앙" 이런 식.
    """
    gh, gw = grid
    a = _to_gray_np(left_img)
    b = _to_gray_np(right_img)

    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    a = a[:h,:w]
    b = b[:h,:w]

    Hs = np.linspace(0, h, gh+1).astype(int)
    Ws = np.linspace(0, w, gw+1).astype(int)

    # 위치 이름 매핑용
    row_names = ["상단","중단","하단"] if gh == 3 else [f"{i}행" for i in range(gh)]
    col_names = ["좌측","중앙","우측"] if gw == 3 else [f"{j}열" for j in range(gw)]

    diffs = []
    for i in range(gh):
        for j in range(gw):
            ys, ye = Hs[i], Hs[i+1]
            xs, xe = Ws[j], Ws[j+1]

            blk_a = a[ys:ye, xs:xe]
            blk_b = b[ys:ye, xs:xe]
            if blk_a.size == 0 or blk_b.size == 0:
                continue

            val, _ = ssim(blk_a, blk_b, full=True)
            diffs.append(((i,j), float(val)))

    # 낮은 SSIM 순 (=차이 큰 순)
    diffs.sort(key=lambda x: x[1])  # ascending
    hints: List[str] = []
    for (i,j), val in diffs[:topk]:
        rn = row_names[min(i, len(row_names)-1)]
        cn = col_names[min(j, len(col_names)-1)]
        hints.append(f"{rn} {cn}")
    return hints

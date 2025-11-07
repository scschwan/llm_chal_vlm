from __future__ import annotations
from pathlib import Path
from typing import List, Optional
from PIL import Image
import matplotlib.pyplot as plt


class ImageProcessor:
    @staticmethod
    def show_pairs(
        query_path: str,
        candidate_paths: List[str],
        sims: List[float],
        suptitle: str,
        thumb_px: int = 640,
        pad_px: int = 8,
        save_path: Optional[str] = None,
        show: bool = True,   # ← 추가: 화면표시 on/off
    ):
        """
        좌측=query_path / 우측=candidate_paths[0] 이미지를
        matplotlib figure에 나란히 표시하고, sim/파일명까지 표기 후
        저장/표시하는 유틸.
        """
        left_img  = Image.open(query_path).convert("RGB")
        right_img = Image.open(candidate_paths[0]).convert("RGB")

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(suptitle, fontsize=10)

        axs[0].imshow(left_img)
        axs[0].set_title(f"Query: {Path(query_path).name}")
        axs[0].axis("off")

        axs[1].imshow(right_img)
        axs[1].set_title(f"Match: {Path(candidate_paths[0]).name}\nSim={sims[0]:.4f}")
        axs[1].axis("off")

        plt.tight_layout(pad=pad_px / 72.0)

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150)

        if show:
            plt.show()

        plt.close(fig)

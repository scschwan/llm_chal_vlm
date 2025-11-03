# modules/split_train_to_test.py
from __future__ import annotations
import random, shutil
from pathlib import Path
from typing import List, Tuple

SEED = 42
SPLIT_RATIO = 0.2  # test 20%

DATASET_ROOT = Path(r"C:\llm_challenge\datasets\defects")
SRC_IMAGES = DATASET_ROOT / "train" / "images"
SRC_LABELS = DATASET_ROOT / "train" / "labels"
DST_TEST_IMAGES = DATASET_ROOT / "test" / "images"
DST_TEST_LABELS = DATASET_ROOT / "test" / "labels"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def list_pairs() -> List[Tuple[Path, Path]]:
    imgs = sorted([p for p in SRC_IMAGES.rglob("*") if p.suffix.lower() in IMG_EXTS])
    pairs = []
    for ip in imgs:
        lp = SRC_LABELS / (ip.stem + ".txt")
        if lp.exists():
            pairs.append((ip, lp))
    return pairs

def main():
    random.seed(SEED)
    pairs = list_pairs()
    n_total = len(pairs)
    assert n_total > 0, f"이미지/라벨 페어가 없습니다: {SRC_IMAGES}"

    # test로 보낼 인덱스 선택(무작위, 재현성 고정)
    idx = list(range(n_total))
    random.shuffle(idx)
    k = int(round(n_total * SPLIT_RATIO))
    test_idx = set(idx[:k])

    # 폴더 준비
    DST_TEST_IMAGES.mkdir(parents=True, exist_ok=True)
    DST_TEST_LABELS.mkdir(parents=True, exist_ok=True)

    moved = 0
    for i, (img, lab) in enumerate(pairs):
        if i in test_idx:
            shutil.move(str(img), str(DST_TEST_IMAGES / img.name))
            shutil.move(str(lab), str(DST_TEST_LABELS / lab.name))
            moved += 1

    print(f"✅ 총 {n_total}장 중 test로 이동: {moved}장 ({moved/n_total*100:.1f}%)")
    print(f"남은 train 수: {n_total - moved}장")
    print(f"test images: {DST_TEST_IMAGES}")
    print(f"test labels: {DST_TEST_LABELS}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image augmentation for folder tree:
- Walks input_dir recursively (e.g., add_image/)
- For each image, generates 6 variants:
  1) rot90, 2) rot180, 3) rot270, 4) flip_h, 5) flip_v, 6) rot90+flip_h
- Saves to output_dir preserving subfolder structure
Python 3.9 / Pillow 10.2.0 compatible
"""

import argparse
import concurrent.futures as futures
import os
from pathlib import Path
from PIL import Image

VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def list_images(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            yield p

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def augment_one(src: Path, in_root: Path, out_root: Path):
    rel = src.relative_to(in_root).with_suffix("")   # e.g. leather/ok/001
    out_dir = (out_root / rel).parent               # .../leather/ok
    ensure_dir(out_dir)

    base = src.stem  # e.g. 001
    ext  = src.suffix.lower()

    try:
        with Image.open(src) as im:
            im.load()  # Read fully

            # 6 variants
            variants = {
                f"{base}_r90{ext}":   im.rotate(90, expand=True),
                f"{base}_r180{ext}":  im.rotate(180, expand=True),
                f"{base}_r270{ext}":  im.rotate(270, expand=True),
                f"{base}_fh{ext}":    im.transpose(Image.FLIP_LEFT_RIGHT),
                f"{base}_fv{ext}":    im.transpose(Image.FLIP_TOP_BOTTOM),
                f"{base}_r90_fh{ext}": im.rotate(90, expand=True).transpose(Image.FLIP_LEFT_RIGHT),
            }

            # Save (Pillow keeps PNG/JPEG by ext)
            for name, img in variants.items():
                out_path = out_dir / name
                # Avoid accidental overwrite if re-run
                if out_path.exists():
                    # add numeric suffix
                    i = 1
                    stem = out_path.stem
                    while True:
                        cand = out_dir / f"{stem}_{i}{ext}"
                        if not cand.exists():
                            out_path = cand
                            break
                        i += 1
                # For PNG keep default, for JPEG ensure quality
                save_kwargs = {}
                if ext in {".jpg", ".jpeg"}:
                    save_kwargs.update(dict(quality=95, subsampling=1, optimize=True))
                img.save(out_path, **save_kwargs)

        return True, str(src)
    except Exception as e:
        return False, f"{src}: {e}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", "-i", default="./add_image",
                    help="입력 폴더 루트 (기본: ./add_image)")
    ap.add_argument("--output_dir", "-o", default="./add_image_aug",
                    help="출력 폴더 루트 (기본: ./add_image_aug)")
    ap.add_argument("--workers", "-w", type=int, default=8,
                    help="병렬 처리 워커 수 (기본: 8)")
    args = ap.parse_args()

    in_root = Path(args.input_dir).resolve()
    out_root = Path(args.output_dir).resolve()

    if not in_root.exists():
        raise SystemExit(f"[ERROR] 입력 폴더가 없습니다: {in_root}")

    print(f"[INFO] input : {in_root}")
    print(f"[INFO] output: {out_root}")
    ensure_dir(out_root)

    imgs = list(list_images(in_root))
    if not imgs:
        raise SystemExit("[ERROR] 처리할 이미지가 없습니다.")

    print(f"[INFO] 총 {len(imgs)}개 원본 → 변형 6종씩 생성 예정")

    ok = 0
    fail = 0
    with futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for success, msg in ex.map(lambda p: augment_one(p, in_root, out_root), imgs):
            if success:
                ok += 1
            else:
                fail += 1
                print("[FAIL]", msg)

    print(f"[DONE] 성공 {ok} / 실패 {fail}")

if __name__ == "__main__":
    main()

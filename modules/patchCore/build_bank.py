#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PatchCore MemoryBank Builder
- 정상(OK) 이미지 폴더를 입력받아 메모리뱅크(정상 패치 임베딩)와 임계치(τ)를 저장합니다.
- τ는 OK 검증(동일 데이터)으로 대략적인 95~99퍼센타일을 계산해 저장합니다.

Usage:
  python build_bank.py --ok_dir ./data/patchCore/prod1/ok --out_dir ./data/patchCore/prod1 \
    --shorter 512 --stride 2 --layers layer2 layer3 layer4 --coreset 0.02

산출물( out_dir ):
  - memory_bank.pt        : torch Tensor [M, D], float32
  - bank_config.json      : 구성 파라미터(JSON)
  - tau.json              : {"pixel": float, "image": float} 임계치
"""

import os
import json
import math
import time
import glob
import argparse
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import cv2

# ============ Utils ============

def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def list_images(d: str) -> List[str]:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    files = []
    for e in exts:
        files += glob.glob(os.path.join(d, e))
    files.sort()
    return files

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

# ============ Feature Extractor (ResNet50 tap layers) ============

class ResNet50Multi(nn.Module):
    """
    ResNet50에서 layer2, layer3, layer4 출력을 훅으로 받아
    layer2 해상도에 맞게 upsample 후 concat(C-dim)한 feature map을 반환.
    """
    def __init__(self, layers=("layer2", "layer3", "layer4")):
        super().__init__()
        self.backbone = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        self.layers = layers

        self._feats = {}
        # forward hook 등록
        def hook(module, input, output, name):
            self._feats[name] = output

        if "layer2" in layers:
            self.backbone.layer2.register_forward_hook(lambda m, i, o: hook(m, i, o, "layer2"))
        if "layer3" in layers:
            self.backbone.layer3.register_forward_hook(lambda m, i, o: hook(m, i, o, "layer3"))
        if "layer4" in layers:
            self.backbone.layer4.register_forward_hook(lambda m, i, o: hook(m, i, o, "layer4"))

        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        # ImageNet 정규화
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    @torch.inference_mode()
    def forward(self, img: torch.Tensor):
        """
        img: [1,3,H,W] in [0,1]
        return: feat_map [1, Csum, H2, W2]  (layer2 해상도로 맞춤)
        """
        x = (img - self.mean) / self.std
        _ = self.backbone(x)  # hooks로 _feats 채워짐

        fmaps = []
        base = self._feats[self.layers[0]]  # 첫 레이어를 기준 해상도로
        H2, W2 = base.shape[-2:]
        for name in self.layers:
            f = self._feats[name]
            if f.shape[-2:] != (H2, W2):
                f = F.interpolate(f, size=(H2, W2), mode="bilinear", align_corners=False)
            f = F.normalize(f, p=2, dim=1)  # 채널 정규화
            fmaps.append(f)
        feat = torch.cat(fmaps, dim=1)  # [1, Csum, H2, W2]
        return feat

def preprocess_bgr_to_tensor(img_bgr: np.ndarray, shorter=512) -> torch.Tensor:
    """
    짧은 변을 shorter로 리사이즈 (비율 유지), BGR->RGB, [0,1] float tensor.
    """
    h, w = img_bgr.shape[:2]
    if min(h, w) != shorter:
        if h < w:
            new_h = shorter
            new_w = int(w * shorter / h)
        else:
            new_w = shorter
            new_h = int(h * shorter / w)
        img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(img_rgb).permute(2,0,1).unsqueeze(0)  # [1,3,H,W]
    return t

def feat_to_patches(feat: torch.Tensor, stride: int = 2) -> torch.Tensor:
    """
    feat: [1, C, H, W]
    return: [N, C] (stride 간격으로 샘플된 위치의 패치 벡터)
    """
    _, C, H, W = feat.shape

    # 위치 그리드 준비 (feat과 같은 device)
    ys = torch.arange(0, H, stride, device=feat.device)
    xs = torch.arange(0, W, stride, device=feat.device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # [Hy, Wx]

    # [1,C,H,W] -> [H,W,C] 로 바꿔서 위치인덱싱 후 평탄화
    f = feat[0].permute(1, 2, 0).contiguous()  # [H, W, C]
    samples = f[grid_y, grid_x]                # [Hy, Wx, C]
    patches = samples.reshape(-1, C)           # [N, C]

    patches = torch.nn.functional.normalize(patches, p=2, dim=1)
    return patches

# ============ Coreset (optional KMeans -> centers), fallback to random ============
def build_coreset(X: np.ndarray, ratio: float = 0.02, seed: int = 0) -> np.ndarray:
    """
    X: [N, D] float32
    return C: [k, D] coreset(centers) or random subset
    """
    k = max(1, int(len(X) * ratio))
    if k >= len(X):
        return X

    # 우선 MiniBatchKMeans 시도
    try:
        from sklearn.cluster import MiniBatchKMeans
        km = MiniBatchKMeans(n_clusters=k, batch_size=2048, n_init="auto", max_iter=50, verbose=0, random_state=seed)
        km.fit(X)
        C = km.cluster_centers_.astype(np.float32, copy=False)
        return C
    except Exception:
        # 폴백: 랜덤 서브셋
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(X), size=k, replace=False)
        return X[idx]

# ============ Distance (faiss → numpy fallback) ============
def knn_min_dist(query: np.ndarray, ref: np.ndarray, chunk: int = 50000) -> np.ndarray:
    """
    각 query 벡터의 최근접(ref) 거리(L2)를 반환 (NumPy 폴백).
    메모리 보호를 위해 chunk 단위로 처리.
    """
    Q, D = query.shape
    R = ref.shape[0]
    out = np.empty(Q, dtype=np.float32)
    for i in range(0, Q, chunk):
        q = query[i:i+chunk]  # [q, D]
        # (q,1,D) - (1,R,D) -> (q,R,D) -> (q,R)
        # (a-b)^2 = a^2 + b^2 - 2ab
        a2 = (q**2).sum(axis=1, keepdims=True)     # [q,1]
        b2 = (ref**2).sum(axis=1, keepdims=True).T # [1,R]
        ab = q @ ref.T                              # [q,R]
        dist = np.maximum(a2 + b2 - 2*ab, 0.0)      # 수치안정
        out[i:i+chunk] = np.sqrt(dist.min(axis=1))
    return out  # [Q]

def knn_min_dist_faiss(query: np.ndarray, ref: np.ndarray) -> np.ndarray:
    try:
        import faiss
        # CPU index (L2)
        index = faiss.IndexFlatL2(ref.shape[1])
        index.add(ref.astype(np.float32))
        D, I = index.search(query.astype(np.float32), 1)
        return np.sqrt(np.maximum(D[:,0], 0.0)).astype(np.float32)
    except Exception:
        return knn_min_dist(query, ref)

# ============ Build Memory Bank Pipeline ============

@torch.inference_mode()
def extract_ok_patches(model: ResNet50Multi,
                       ok_paths: List[str],
                       device: torch.device,
                       shorter=512, stride=2,
                       reservoir_max=100_000,
                       seed=0) -> np.ndarray:
    """
    OK 이미지 집합에서 패치 임베딩을 스트리밍으로 수집(저장소 cap 적용)
    return: np.ndarray [N, C]
    """
    rng = np.random.RandomState(seed)
    reservoir = []

    log(f"extract patches from OK images: {len(ok_paths)} imgs")
    for p in ok_paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            continue
        t = preprocess_bgr_to_tensor(img, shorter=shorter).to(device)
        feat = model(t)                    # [1, C, H2, W2]
        patches = feat_to_patches(feat, stride=stride)  # [N, C] torch
        arr = patches.cpu().numpy().astype(np.float32, copy=False)

        # reservoir cap per image (optional): limit overly large contribution
        per_cap = max(2000, int(1e9 / arr.shape[1]))  # crude guard
        if len(arr) > per_cap:
            idx = rng.choice(len(arr), size=per_cap, replace=False)
            arr = arr[idx]

        reservoir.append(arr)

        # global cap
        total = sum(x.shape[0] for x in reservoir)
        if total > reservoir_max:
            # trim by random
            concat = np.concatenate(reservoir, axis=0)
            idx = rng.choice(concat.shape[0], size=reservoir_max, replace=False)
            concat = concat[idx]
            reservoir = [concat]

    R = np.concatenate(reservoir, axis=0) if reservoir else np.empty((0, 1), np.float32)
    log(f"reservoir size: {R.shape}")
    return R

def calibrate_tau_pixel_image(model: ResNet50Multi,
                              bank: np.ndarray,
                              ok_paths: List[str],
                              device: torch.device,
                              shorter=512, stride=2,
                              pixel_pctl=0.995, image_pctl=0.995) -> Tuple[float, float]:
    """
    OK 이미지에 대해 anomaly distance를 계산하여 임계치(τ) 산출.
    - pixel-level τ: 픽셀/패치 거리의 상위 퍼센타일
    - image-level τ: 이미지 점수(최댓값 혹은 상위-k 평균)의 상위 퍼센타일
    """
    pixel_scores = []
    image_scores = []

    for p in ok_paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            continue
        t = preprocess_bgr_to_tensor(img, shorter=shorter).to(device)
        feat = model(t)
        patches = feat_to_patches(feat, stride=stride).cpu().numpy().astype(np.float32, copy=False)  # [N, C]

        d = knn_min_dist_faiss(patches, bank)  # [N]
        pixel_scores.extend(d.tolist())

        # 이미지 스코어: 최댓값(또는 상위 1% 평균 등)
        image_scores.append(float(np.percentile(d, 99)))

    pixel_tau = float(np.percentile(pixel_scores, pixel_pctl*100.0)) if pixel_scores else 0.0
    image_tau = float(np.percentile(image_scores, image_pctl*100.0)) if image_scores else 0.0
    return pixel_tau, image_tau

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ok_dir", required=True, help="정상 이미지 폴더 (예: ./data/patchCore/prod1/ok)")
    ap.add_argument("--out_dir", required=True, help="메모리뱅크/설정 저장 폴더 (예: ./data/patchCore/prod1)")
    ap.add_argument("--shorter", type=int, default=512)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--layers", nargs="+", default=["layer2","layer3","layer4"], choices=["layer2","layer3","layer4"])
    ap.add_argument("--coreset", type=float, default=0.02, help="0~1, 메모리뱅크 압축 비율")
    ap.add_argument("--reservoir_max", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pixel_pct", type=float, default=0.995, help="OK pixel score 상위 퍼센타일")
    ap.add_argument("--image_pct", type=float, default=0.995, help="OK image score 상위 퍼센타일")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    ok_paths = list_images(args.ok_dir)
    if len(ok_paths) == 0:
        raise RuntimeError(f"OK 이미지가 없습니다: {args.ok_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device: {device}")
    log("load backbone...")
    model = ResNet50Multi(layers=tuple(args.layers)).to(device).eval()

    # 1) OK patches 수집
    R = extract_ok_patches(model, ok_paths, device,
                           shorter=args.shorter, stride=args.stride,
                           reservoir_max=args.reservoir_max, seed=args.seed)
    if R.shape[0] == 0:
        raise RuntimeError("수집된 패치가 없습니다.")

    # 2) coreset 압축
    if args.coreset > 0:
        log(f"build coreset (ratio={args.coreset}) ...")
        bank = build_coreset(R, ratio=args.coreset, seed=args.seed)
    else:
        bank = R
    log(f"memory bank size: {bank.shape}")

    # 3) τ 캘리브레이션
    log("calibrate thresholds (τ) on OK set ...")
    pixel_tau, image_tau = calibrate_tau_pixel_image(model, bank, ok_paths, device,
                                                     shorter=args.shorter, stride=args.stride,
                                                     pixel_pctl=args.pixel_pct, image_pctl=args.image_pct)
    log(f"τ pixel: {pixel_tau:.4f}, τ image: {image_tau:.4f}")

    # 4) 저장
    torch.save(torch.from_numpy(bank), os.path.join(args.out_dir, "memory_bank.pt"))
    with open(os.path.join(args.out_dir, "bank_config.json"), "w") as f:
        json.dump({
            "shorter": args.shorter,
            "stride": args.stride,
            "layers": args.layers,
            "coreset": args.coreset,
            "reservoir_max": args.reservoir_max,
            "seed": args.seed
        }, f, indent=2)
    with open(os.path.join(args.out_dir, "tau.json"), "w") as f:
        json.dump({"pixel": pixel_tau, "image": image_tau}, f, indent=2)

    log(f"saved: {args.out_dir}/memory_bank.pt, bank_config.json, tau.json")
    log("done.")

if __name__ == "__main__":
    main()

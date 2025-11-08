#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PatchCore Inference (Single Image)
- 저장된 메모리뱅크를 로드하여 테스트 이미지의 이상 히트맵/마스크/오버레이를 생성.

Usage:
  python infer.py --bank_dir ./data/patchCore/prod1 \
    --img ./data/def_front/cast_def_test.jpeg \
    --out_dir ./out_prod1 \
    --fuse_ref ./data/ok_front/cast_ok_test.jpeg   # (옵션) 기준이미지와 나란히 비교 이미지 생성

출력(out_dir):
  - heat_gray.png      : 0~255 heatmap
  - mask.png           : 이진 마스크(τ_pixel로 threshold)
  - overlay.png        : 원본 위에 빨간색 오버레이
  - side_by_side.png   : (옵션) 기준/테스트/마스크/오버레이 2x2 배열
  - scores.json        : {"image_score": float, "pixel_tau": float, "image_tau": float}
"""

import os
import json
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import torchvision

from typing import Tuple

# --- 동일 모듈 (간단 복제) ---
class ResNet50Multi(torch.nn.Module):
    def __init__(self, layers=("layer2","layer3","layer4")):
        super().__init__()
        self.backbone = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        self.layers = layers
        self._feats = {}
        def hook(module, input, output, name):
            self._feats[name] = output
        if "layer2" in layers:
            self.backbone.layer2.register_forward_hook(lambda m,i,o: hook(m,i,o,"layer2"))
        if "layer3" in layers:
            self.backbone.layer3.register_forward_hook(lambda m,i,o: hook(m,i,o,"layer3"))
        if "layer4" in layers:
            self.backbone.layer4.register_forward_hook(lambda m,i,o: hook(m,i,o,"layer4"))
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485,0.456,0.406]).view(1,3,1,1))
        self.register_buffer("std",  torch.tensor([0.229,0.224,0.225]).view(1,3,1,1))

    @torch.inference_mode()
    def forward(self, img):
        x = (img - self.mean) / self.std
        _ = self.backbone(x)
        base = self._feats[self.layers[0]]
        H2, W2 = base.shape[-2:]
        fm = []
        for name in self.layers:
            f = self._feats[name]
            if f.shape[-2:] != (H2, W2):
                f = F.interpolate(f, size=(H2,W2), mode="bilinear", align_corners=False)
            f = F.normalize(f, p=2, dim=1)
            fm.append(f)
        return torch.cat(fm, dim=1)  # [1,Csum,H2,W2]

def preprocess_bgr_to_tensor(img_bgr: np.ndarray, shorter=512) -> torch.Tensor:
    h,w = img_bgr.shape[:2]
    if min(h,w) != shorter:
        if h < w:
            new_h = shorter
            new_w = int(w * shorter / h)
        else:
            new_w = shorter
            new_h = int(h * shorter / w)
        img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    t = torch.from_numpy(img_rgb).permute(2,0,1).unsqueeze(0)
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

def knn_min_dist(query: np.ndarray, ref: np.ndarray, chunk: int = 50000) -> np.ndarray:
    Q, D = query.shape
    out = np.empty(Q, dtype=np.float32)
    ref2 = (ref**2).sum(axis=1, keepdims=True).T  # [1,R]
    for i in range(0, Q, chunk):
        q = query[i:i+chunk]
        a2 = (q**2).sum(axis=1, keepdims=True)
        ab = q @ ref.T
        dist = np.maximum(a2 + ref2 - 2*ab, 0.0)
        out[i:i+chunk] = np.sqrt(dist.min(axis=1))
    return out

def knn_min_dist_faiss(query: np.ndarray, ref: np.ndarray) -> np.ndarray:
    try:
        import faiss
        index = faiss.IndexFlatL2(ref.shape[1])
        index.add(ref.astype(np.float32))
        D, I = index.search(query.astype(np.float32), 1)
        return np.sqrt(np.maximum(D[:,0], 0.0)).astype(np.float32)
    except Exception:
        return knn_min_dist(query, ref)

def upsample_to_img(heat_small: np.ndarray, img_hw: Tuple[int,int]) -> np.ndarray:
    # bilinear upsample to image size
    h, w = img_hw
    heat = cv2.resize(heat_small, (w, h), interpolation=cv2.INTER_CUBIC)
    # normalize 0~1
    if heat.max() > 0:
        heat = heat / heat.max()
    return heat

def save_mask_overlay(img_bgr: np.ndarray, heat01: np.ndarray, pixel_tau: float, out_dir: str):
    # heat01: 0~1
    heat_gray = (np.clip(heat01, 0, 1) * 255).astype(np.uint8)
    _, mask = cv2.threshold((heat01*255).astype(np.uint8), int(pixel_tau*255/np.max([pixel_tau,1e-6])), 255, cv2.THRESH_BINARY)

    # 시각화
    overlay = img_bgr.copy()
    overlay[mask > 0] = [0,0,255]
    overlay = cv2.addWeighted(img_bgr, 0.7, overlay, 0.3, 0)

    cv2.imwrite(os.path.join(out_dir, "heat_gray.png"), heat_gray)
    cv2.imwrite(os.path.join(out_dir, "mask.png"), mask)
    cv2.imwrite(os.path.join(out_dir, "overlay.png"), overlay)

def make_side_by_side(ok_bgr: np.ndarray, test_bgr: np.ndarray, mask: np.ndarray, overlay: np.ndarray, out_path: str):
    # 동일 높이로 리사이즈 후 2x2 타일
    h = 512
    def resize_h(x):
        hh, ww = x.shape[:2]
        new_w = int(ww * h / hh)
        return cv2.resize(x, (new_w, h), interpolation=cv2.INTER_AREA)

    a = resize_h(ok_bgr) if ok_bgr is not None else np.zeros((h, h, 3), np.uint8)
    b = resize_h(test_bgr)
    m = cv2.cvtColor(resize_h(mask), cv2.COLOR_GRAY2BGR)
    o = resize_h(overlay)

    # 폭 맞추기
    wmax = max(a.shape[1], b.shape[1], m.shape[1], o.shape[1])
    def pad_w(img):
        pad = wmax - img.shape[1]
        if pad <= 0: return img
        return cv2.copyMakeBorder(img, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=(0,0,0))
    A,B,M,O = map(pad_w, [a,b,m,o])

    top = np.hstack([A,B])
    bot = np.hstack([M,O])
    grid = np.vstack([top, bot])
    cv2.imwrite(out_path, grid)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank_dir", required=True, help="메모리뱅크 폴더 (memory_bank.pt, bank_config.json, tau.json 포함)")
    ap.add_argument("--img", required=True, help="테스트 이미지 경로")
    ap.add_argument("--out_dir", required=True, help="출력 폴더")
    ap.add_argument("--fuse_ref", default=None, help="(옵션) 기준(정상) 이미지 경로: side_by_side.png에 함께 표시")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 로드: bank, config, tau
    bank_path = os.path.join(args.bank_dir, "memory_bank.pt")
    cfg_path  = os.path.join(args.bank_dir, "bank_config.json")
    tau_path  = os.path.join(args.bank_dir, "tau.json")
    if not (os.path.isfile(bank_path) and os.path.isfile(cfg_path) and os.path.isfile(tau_path)):
        raise RuntimeError("bank_dir에 memory_bank.pt / bank_config.json / tau.json이 모두 존재해야 합니다.")

    bank = torch.load(bank_path, map_location="cpu").cpu().numpy().astype(np.float32, copy=False)  # [M,D]
    cfg  = json.load(open(cfg_path, "r"))
    tau  = json.load(open(tau_path, "r"))
    pixel_tau = float(tau.get("pixel", 0.0))  # 이 값은 '거리 스케일' 기준
    image_tau = float(tau.get("image", 0.0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 로드
    model = ResNet50Multi(layers=tuple(cfg.get("layers", ["layer2","layer3","layer4"]))).to(device).eval()

    # 이미지 로드
    img_bgr = cv2.imread(args.img, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"이미지를 열 수 없습니다: {args.img}")

    t = preprocess_bgr_to_tensor(img_bgr, shorter=int(cfg.get("shorter", 512))).to(device)
    feat = model(t)  # [1,C,H2,W2]
    patches = feat_to_patches(feat, stride=int(cfg.get("stride", 2))).cpu().numpy().astype(np.float32, copy=False)  # [N,C]

    # 최근접 거리(이상도)
    d = knn_min_dist_faiss(patches, bank)  # [N]
    # heatmap을 원본 해상도로
    # patch grid 크기:
    _, C, H2, W2 = feat.shape
    gy = list(range(0, H2, int(cfg.get("stride", 2))))
    gx = list(range(0, W2, int(cfg.get("stride", 2))))
    Hg, Wg = len(gy), len(gx)
    heat_small = d.reshape(Hg, Wg)  # (Hg, Wg)

    # 0~1 정규화된 heat
    # (거리 → 큰 값 = 더 이상)  →  0~1로 normalize
    maxv = float(heat_small.max()) if heat_small.size > 0 else 1.0
    heat01_small = heat_small / maxv if maxv > 0 else heat_small
    heat01 = upsample_to_img(heat01_small, img_bgr.shape[:2])

    # 저장 (pixel_tau는 '거리 값' 기준이므로 heat01 스케일로 변환해서 threshold)
    # heat01 = dist / dist_max  ⇒  dist = heat01 * dist_max
    dist_max_img = maxv
    pixel_tau_scaled = (pixel_tau / dist_max_img) if dist_max_img > 0 else 1.0
    pixel_tau_scaled = float(np.clip(pixel_tau_scaled, 0.0, 1.0))

    # 마스크/오버레이 저장
    heat_u8 = (np.clip(heat01, 0, 1) * 255).astype(np.uint8)
    _, mask = cv2.threshold(heat_u8, int(pixel_tau_scaled*255), 255, cv2.THRESH_BINARY)
    overlay = img_bgr.copy()
    overlay[mask > 0] = [0,0,255]
    overlay = cv2.addWeighted(img_bgr, 0.7, overlay, 0.3, 0)

    cv2.imwrite(os.path.join(args.out_dir, "heat_gray.png"), heat_u8)
    cv2.imwrite(os.path.join(args.out_dir, "mask.png"), mask)
    cv2.imwrite(os.path.join(args.out_dir, "overlay.png"), overlay)

    # (옵션) 기준 이미지와 2x2 배치
    if args.fuse_ref:
        ref_bgr = cv2.imread(args.fuse_ref, cv2.IMREAD_COLOR)
        if ref_bgr is None:
            ref_bgr = np.zeros_like(img_bgr)
    else:
        ref_bgr = None
    if ref_bgr is not None:
        # side-by-side 저장
        # (mask/overlay는 이미 동일 크기)
        side_path = os.path.join(args.out_dir, "side_by_side.png")
        def resize_h(x, h=512):
            hh, ww = x.shape[:2]
            new_w = int(ww * h / hh)
            return cv2.resize(x, (new_w, h), interpolation=cv2.INTER_AREA)
        a = resize_h(ref_bgr) if ref_bgr is not None else np.zeros((512,512,3), np.uint8)
        b = resize_h(img_bgr)
        m = cv2.cvtColor(resize_h(mask), cv2.COLOR_GRAY2BGR)
        o = resize_h(overlay)
        wmax = max(a.shape[1], b.shape[1], m.shape[1], o.shape[1])
        def pad_w(img):
            pad = wmax - img.shape[1]
            return cv2.copyMakeBorder(img, 0,0,0,max(0,pad), cv2.BORDER_CONSTANT, value=(0,0,0))
        grid = np.vstack([np.hstack([pad_w(a), pad_w(b)]), np.hstack([pad_w(m), pad_w(o)])])
        cv2.imwrite(side_path, grid)

    # 이미지 스코어(최댓값) 계산 및 저장
    image_score = float(np.percentile(d, 99)) if d.size else 0.0
    with open(os.path.join(args.out_dir, "scores.json"), "w") as f:
        json.dump({"image_score": image_score, "pixel_tau": float(pixel_tau), "image_tau": float(image_tau)}, f, indent=2)

    print(f"[done] saved to: {args.out_dir}")

if __name__ == "__main__":
    main()

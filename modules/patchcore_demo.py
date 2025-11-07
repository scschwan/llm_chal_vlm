import argparse, os, glob, math
import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

# -----------------------------
# 1) Feature Extractor (ResNet50 중간계층) 
# -----------------------------
class ResNet50Embed(torch.nn.Module):
    #def __init__(self, layers=("layer2","layer3","layer4")):
    def __init__(self, layers=("layer3","layer4")):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # stem = conv1 → bn1 → relu → maxpool
        self.stem = torch.nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        # ★ layer1을 반드시 포함해야 다음 블록 입력 채널이 맞습니다.
        self.layer1 = backbone.layer1  # 64ch → 256ch
        self.layer2 = backbone.layer2  # 256ch → 512ch
        self.layer3 = backbone.layer3  # 512ch → 1024ch
        self.layer4 = backbone.layer4  # 1024ch → 2048ch
        self.layers = layers
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, x):
        x = self.stem(x)       # [B, 64, 1/4H, 1/4W]
        l1 = self.layer1(x)    # [B, 256, 1/4H, 1/4W]
        f2 = self.layer2(l1)   # [B, 512, 1/8H, 1/8W]
        f3 = self.layer3(f2)   # [B,1024, 1/16H,1/16W]
        f4 = self.layer4(f3)   # [B,2048, 1/32H,1/32W]
        feats = []
        # 필요하면 layer1 특성도 쓰려면 "layer1"을 layers에 추가해 주세요.
        if "layer1" in self.layers: feats.append(l1)
        if "layer2" in self.layers: feats.append(f2)
        if "layer3" in self.layers: feats.append(f3)
        if "layer4" in self.layers: feats.append(f4)
        return feats

# -----------------------------
# 2) 이미지 전처리 & 보조 함수
# -----------------------------
#def make_transform(shorter=768):
def make_transform(shorter=512):
    return transforms.Compose([
        transforms.ToTensor(),                                 # [0,1]
        transforms.Resize(shorter, antialias=True),            # 짧은 변 기준
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225]),
    ])

def load_image_bgr(path):
    img = cv.imread(path, cv.IMREAD_COLOR)
    if img is None: raise FileNotFoundError(path)
    return img

def to_tensor_rgb(img_bgr, tfm):
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    x = tfm(img_rgb)  # [3,H,W] float
    return x

def up_to(img_like, fmap):
    # fmap: [1,C,h,w] -> upsample to img_like spatial size
    h, w = img_like.shape[-2], img_like.shape[-1]
    return F.interpolate(fmap, size=(h,w), mode="bilinear", align_corners=False)

def l2_normalize(feat, eps=1e-6):
    n = torch.norm(feat, p=2, dim=1, keepdim=True)
    return feat / (n + eps)

# -----------------------------
# 3) 메모리(coreset) 구성
# -----------------------------
@torch.no_grad()
def extract_patches(model, img_tensor, device, stride=1, use_half=False):
    # img_tensor: [1,3,H,W]
    with torch.no_grad():
        feats = model(img_tensor.to(device))
        # 모든 계층을 공통 해상도로 upsample & concat
        up = []
        for f in feats:
            u = F.interpolate(f, size=img_tensor.shape[-2:], mode="bilinear", align_corners=False)
            u = l2_normalize(u)
            up.append(u)
        emb = torch.cat(up, dim=1)  # [1, Csum, H, W]

        if stride > 1:
            # 평균 풀링으로 패치 개수 감소
            emb = F.avg_pool2d(emb, kernel_size=stride, stride=stride)  # [1,C,h',w']

        if use_half:
            emb = emb.half()

        # [H*W, C]
        emb = emb.squeeze(0).permute(1,2,0).reshape(-1, emb.size(1))
        return emb  # torch tensor (device 그대로)

    # img_tensor: [1,3,H,W]
    feats = model(img_tensor.to(device))
    # 모든 계층을 공통 공간으로 upsample & concat
    up = [l2_normalize(up_to(img_tensor, f)) for f in feats]
    emb = torch.cat(up, dim=1)  # [1, Csum, H, W]
    # [H*W, C] 패치 벡터로 변형
    emb = emb.squeeze(0).permute(1,2,0).reshape(-1, emb.size(1))
    emb = emb.half()  # extract_patches 끝 부분 
    return emb.cpu()  # [Npatch, Csum]

def kcenter_greedy(X, keep_ratio=0.05, seed=0):
    """
    간단한 k-center coreset. X: [N, D] (numpy)
    keep_ratio 0.01~0.10 권장
    """
    np.random.seed(seed)
    n = X.shape[0]
    k = max(1, int(n * keep_ratio))
    sel = np.random.choice(n, 1)  # 시작점 1개
    dist = np.full((n,), np.inf)
    for _ in range(k-1):
        # 현재 선택 집합까지의 최근접 거리 업데이트
        d = np.linalg.norm(X - X[sel[-1]], axis=1)
        dist = np.minimum(dist, d)
        sel = np.append(sel, np.argmax(dist))
    return np.unique(sel)

import random

def reservoir_add(reservoir, x_np, k_max, rng):
    """
    표준 reservoir sampling (유니폼).
    reservoir: list of np.ndarray chunks
    x_np: [n, d] np.ndarray
    k_max: 목표 총 패치 수 상한 (예: 100_000)
    """
    if x_np.size == 0:
        return
    # 초기에 빈 공간은 그냥 채움
    total = sum(arr.shape[0] for arr in reservoir)
    remain = max(0, k_max - total)
    if remain > 0:
        take = min(remain, x_np.shape[0])
        reservoir.append(x_np[:take])
        x_np = x_np[take:]
        total += take
    # 꽉 찼으면 확률적으로 교체
    i = 0
    while i < x_np.shape[0]:
        total += 1
        if rng.random() < (k_max / total):
            # 교체: 임의 위치의 행을 바꿈
            idx_chunk = rng.randrange(len(reservoir))
            if reservoir[idx_chunk].shape[0] == 0:
                idx_chunk = 0
            r = rng.randrange(reservoir[idx_chunk].shape[0])
            reservoir[idx_chunk][r] = x_np[i]
        i += 1

@torch.no_grad()
def build_memory_bank(model, img_paths, device, shorter=512, coreset_ratio=0.02,
                      stride=2, use_half=False, reservoir_max=100_000, seed=0):
    """
    - 전 이미지 특징을 cat하지 않고 reservoir에 최대 reservoir_max 패치만 유지
    - 마지막에만 coreset(k-center greedy)을 reservoir 위에서 수행
    """
    tfm = make_transform(shorter)
    rng = random.Random(seed)
    reservoir = []  # list of np arrays; 총합 <= reservoir_max

    for p in tqdm(img_paths, desc="Extract OK features (streaming)"):
        img = load_image_bgr(p)
        x = to_tensor_rgb(img, tfm).unsqueeze(0)
        # GPU에서 추출 후 CPU로 내리되(또는 곧장 np로) 절약
        emb = extract_patches(model, x, device, stride=stride, use_half=use_half)  # [N, C]
        emb_cpu = emb.detach().to('cpu').float().numpy()  # coreset 전에 float32 기준화
        # 이미지별로 너무 많으면 추가 추려서 reservoir에 넣기(이미지당 상한)
        per_img_cap = max(2000 // max(1, stride-1), 2000)  # 필요 시 조정
        if emb_cpu.shape[0] > per_img_cap:
            idx = rng.sample(range(emb_cpu.shape[0]), per_img_cap)
            emb_cpu = emb_cpu[idx]
        reservoir_add(reservoir, emb_cpu, reservoir_max, rng)

    if len(reservoir) == 0:
        raise RuntimeError("Reservoir is empty; check OK images or transforms.")

    R = np.concatenate(reservoir, axis=0)  # <= reservoir_max x C
    # coreset 대상 수 결정
    k = max(1, int(R.shape[0] * coreset_ratio))
    # k-center greedy on R
    sel_idx = kcenter_greedy(R, keep_ratio=k / R.shape[0], seed=seed)
    memory = torch.from_numpy(R[sel_idx]).to(device).float()
    return memory


# -----------------------------
# 4) 추론: 최근접 거리 히트맵 + 이미지 점수
# -----------------------------
@torch.no_grad()
def infer_anomaly(model, memory, img_bgr, device, shorter=768):
    tfm = make_transform(shorter)
    x = to_tensor_rgb(img_bgr, tfm).unsqueeze(0).to(device)
    feats = model(x)
    up = [l2_normalize(up_to(x, f)) for f in feats]
    emb_map = torch.cat(up, dim=1)         # [1,C,H,W]
    C, H, W = emb_map.size(1), emb_map.size(2), emb_map.size(3)
    patches = emb_map.permute(0,2,3,1).reshape(-1, C)  # [N, C]

    # L2 최근접 거리 (chunk로 메모리 절약)
    memory = memory.to(device)
    B = 4096
    dists = []
    for i in range(0, patches.size(0), B):
        q = patches[i:i+B]                  # [b,C]
        # ||q - m||^2 = ||q||^2 + ||m||^2 - 2 q m^T
        q2 = (q*q).sum(1, keepdim=True)     # [b,1]
        m2 = (memory*memory).sum(1).unsqueeze(0)  # [1,M]
        d = q2 + m2 - 2.0 * (q @ memory.t())     # [b,M]
        d_min, _ = torch.min(d, dim=1)           # [b]
        dists.append(torch.sqrt(torch.clamp(d_min, min=0)))
    dists = torch.cat(dists).reshape(H, W)       # [H,W]

    # 이미지 스코어: 상위 k% 패치 평균
    k = max(1, int(0.01 * H * W))
    img_score = torch.topk(dists.reshape(-1), k).values.mean().item()

    # 0-255 히트맵 (원본 크기로 리사이즈)
    heat = dists
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-6)
    heat = (heat * 255).byte().cpu().numpy()
    return heat, img_score

def overlay_heatmap(img_bgr, heat_gray, alpha=0.6):
    heat_color = cv.applyColorMap(heat_gray, cv.COLORMAP_JET)
    heat_color = cv.resize(heat_color, (img_bgr.shape[1], img_bgr.shape[0]),
                           interpolation=cv.INTER_LINEAR)
    over = cv.addWeighted(img_bgr, 1-alpha, heat_color, alpha, 0)
    return over

# -----------------------------
# 5) 메인
# -----------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet50Embed().to(device).eval()

    ok_imgs = sorted(sum([glob.glob(os.path.join(args.ok_dir, f"*{ext}"))
                          for ext in [".jpg",".jpeg",".png",".bmp",".tif",".tiff"]], []))
    assert len(ok_imgs) > 0, "OK 이미지가 없습니다."
    memory = build_memory_bank(model, ok_imgs, device,
                               shorter=args.shorter, coreset_ratio=args.coreset)

    test_bgr = load_image_bgr(args.test)
    heat, score = infer_anomaly(model, memory, test_bgr, device,
                            shorter=args.shorter,  # 512 또는 384
                            # infer 함수 내부에서도 extract와 동일한 처리 적용
                           )


    over = overlay_heatmap(test_bgr, heat)
    cv.imwrite(args.out, over)
    # 마스크도 함께 저장(원하면)
    cv.imwrite(os.path.splitext(args.out)[0] + "_gray.png", heat)

    print(f"[DONE] saved: {args.out}")
    print(f"Anomaly score (higher=worse): {score:.4f}")
    # 간단 임계치 제안: OK 검증셋의 99퍼센타일로 설정 권장
    # 여기서는 임시로 스코어>0.5*255 기준을 프린트만 해둠
    # (실 배포 시 OK만 여러 장 넣고 스코어의 퍼센타일을 계산하세요.)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ok_dir", type=str, required=True)
    ap.add_argument("--test", type=str, required=True)
    ap.add_argument("--out", type=str, default="out_heatmap.png")
    #ap.add_argument("--shorter", type=int, default=768)
    #ap.add_argument("--coreset", type=float, default=0.05)
    ap.add_argument("--shorter", type=int, default=512)
    ap.add_argument("--coreset", type=float, default=0.02)
    args = ap.parse_args()
    main(args)

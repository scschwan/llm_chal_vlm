# modules/train_yolo_seg.py
import os, sys, glob, cv2, numpy as np, yaml, torch
from ultralytics import YOLO

ROOT_THIS_FILE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT   = os.path.abspath(os.path.join(ROOT_THIS_FILE, os.pardir))
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)
if ROOT_THIS_FILE not in sys.path: sys.path.insert(0, ROOT_THIS_FILE)

# ---------- 유틸 ----------
def _delete_label_caches(dataset_path):
    for p in [os.path.join(dataset_path, s, "labels.cache") for s in ("train","valid","test")]:
        if os.path.isfile(p):
            try: os.remove(p); print(f"[INFO] removed cache: {p}")
            except Exception as e: print(f"[WARN] cache remove failed: {p} ({e})")

def _sanitize_labels_invalid_classes(labels_dir: str, nc: int):
    if not os.path.isdir(labels_dir): return (0, 0)
    changed = dropped = 0
    for p in glob.glob(os.path.join(labels_dir, "*.txt")):
        with open(p, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        new_lines = []
        for ln in lines:
            parts = ln.split()
            try: cid = int(float(parts[0]))
            except Exception: dropped += 1; continue
            if not (0 <= cid < nc): dropped += 1; continue
            new_lines.append(ln)
        if new_lines != lines:
            changed += 1
            with open(p, "w", encoding="utf-8") as f:
                f.write("\n".join(new_lines) + ("\n" if new_lines else ""))
    return (changed, dropped)

def _detect_dataset_kind(labels_dir: str) -> dict:
    """라벨 txt를 스캔해 세그/디텍 여부 판정"""
    files = glob.glob(os.path.join(labels_dir, "*.txt"))
    seg_ok = det_ok = empty = 0
    for p in files:
        with open(p, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if not lines: empty += 1; continue
        # 세그: 토큰 7개 이상( cls + (x,y)*>=3 )
        has_seg = any(len(ln.split()) >= 7 for ln in lines)
        # 디텍: 정확히 5개( cls cx cy w h )
        has_det = any(len(ln.split()) == 5 for ln in lines)
        if has_seg and not has_det: seg_ok += 1
        if has_det: det_ok += 1
    return {"total": len(files), "seg_ok": seg_ok, "det_ok": det_ok, "empty": empty}

# ---------- 공통 IoU ----------
def _iou_masks(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    uni   = np.logical_or(a, b).sum()
    return inter / uni if uni > 0 else 0.0

def _iou_boxes(ba, bb) -> float:
    # ba, bb: (x1,y1,x2,y2)
    xa1, ya1, xa2, ya2 = ba; xb1, yb1, xb2, yb2 = bb
    iw = max(0.0, min(xa2, xb2) - max(xa1, xb1))
    ih = max(0.0, min(ya2, yb2) - max(ya1, yb1))
    inter = iw * ih
    area_a = max(0.0, (xa2 - xa1)) * max(0.0, (ya2 - ya1))
    area_b = max(0.0, (xb2 - xb1)) * max(0.0, (yb2 - yb1))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

# ---------- 평가(세그) ----------
def _eval_seg(yaml_path, model_path, split="val", imgsz=640, conf=0.25, iou_thr=0.5, device=None):
    data = yaml.safe_load(open(yaml_path, "r", encoding="utf-8"))
    root = data.get("path", ".")
    img_dir = os.path.join(root, data.get(split, f"{split}/images"))
    lbl_dir = img_dir.replace("/images","/labels").replace("\\images","\\labels")
    model = YOLO(model_path)

    total_tp = total_fp = total_fn = 0
    for ip in sorted(glob.glob(os.path.join(img_dir, "*.*"))):
        if os.path.splitext(ip)[1].lower() not in [".jpg",".jpeg",".png",".bmp"]: continue
        img = cv2.imread(ip); h,w = img.shape[:2]
        stem = os.path.splitext(os.path.basename(ip))[0]
        lp = os.path.join(lbl_dir, stem + ".txt")

        gts = []
        if os.path.isfile(lp):
            for ln in open(lp,"r",encoding="utf-8"):
                vs = ln.strip().split()
                if not vs: continue
                coords = np.array(list(map(float, vs[1:])), np.float32).reshape(-1,2)
                coords[:,0]*=w; coords[:,1]*=h
                mask = np.zeros((h,w), np.uint8)
                if len(coords)>=3: cv2.fillPoly(mask,[coords.astype(np.int32)],1)
                gts.append(mask.astype(bool))

        rs = model.predict(ip, imgsz=imgsz, conf=conf, device=device or (0 if torch.cuda.is_available() else "cpu"), verbose=False)
        preds=[]
        if rs and rs[0].masks is not None:
            for poly in rs[0].masks.xyn:
                pts = np.array(poly, np.float32); pts[:,0]*=w; pts[:,1]*=h
                m = np.zeros((h,w), np.uint8)
                if len(pts)>=3: cv2.fillPoly(m,[pts.astype(np.int32)],1)
                preds.append(m.astype(bool))

        used=set(); tp=fp=0
        for pm in preds:
            best=-1; bj=-1
            for j,gm in enumerate(gts):
                if j in used: continue
                i=_iou_masks(pm,gm)
                if i>best: best=i; bj=j
            if bj!=-1 and best>=iou_thr: tp+=1; used.add(bj)
            else: fp+=1
        fn = len(gts)-len(used)
        total_tp += tp; total_fp += fp; total_fn += fn

    P = total_tp/(total_tp+total_fp) if total_tp+total_fp>0 else 0.0
    R = total_tp/(total_tp+total_fn) if total_tp+total_fn>0 else 0.0
    F1 = 2*P*R/(P+R) if P+R>0 else 0.0
    Acc= total_tp/(total_tp+total_fp+total_fn) if total_tp+total_fp+total_fn>0 else 0.0
    print("\n==== Object-level (SEG) ====")
    print(f"TP={total_tp} FP={total_fp} FN={total_fn}")
    print(f"Precision: {P:.4f}  Recall: {R:.4f}  F1: {F1:.4f}  Accuracy: {Acc:.4f}")

# ---------- 평가(디텍) ----------
def _eval_det(yaml_path, model_path, split="val", imgsz=640, conf=0.25, iou_thr=0.5, device=None):
    data = yaml.safe_load(open(yaml_path, "r", encoding="utf-8"))
    root = data.get("path", ".")
    img_dir = os.path.join(root, data.get(split, f"{split}/images"))
    lbl_dir = img_dir.replace("/images","/labels").replace("\\images","\\labels")
    model = YOLO(model_path)

    total_tp = total_fp = total_fn = 0
    for ip in sorted(glob.glob(os.path.join(img_dir, "*.*"))):
        if os.path.splitext(ip)[1].lower() not in [".jpg",".jpeg",".png",".bmp"]: continue
        img = cv2.imread(ip); h,w = img.shape[:2]
        stem = os.path.splitext(os.path.basename(ip))[0]
        lp = os.path.join(lbl_dir, stem + ".txt")

        gts=[]
        if os.path.isfile(lp):
            for ln in open(lp,"r",encoding="utf-8"):
                vs = ln.strip().split()
                if len(vs)!=5: continue  # 세그 라인 제외
                _, cx, cy, bw, bh = map(float, vs)
                x1 = (cx-bw/2.0)*w; y1 = (cy-bh/2.0)*h
                x2 = (cx+bw/2.0)*w; y2 = (cy+bh/2.0)*h
                gts.append((x1,y1,x2,y2))

        rs = model.predict(ip, imgsz=imgsz, conf=conf, device=device or (0 if torch.cuda.is_available() else "cpu"), verbose=False)
        preds=[]
        if rs and rs[0].boxes is not None:
            for b in rs[0].boxes.xyxy.cpu().numpy():
                preds.append(tuple(map(float, b)))

        used=set(); tp=fp=0
        for pb in preds:
            best=-1; bj=-1
            for j,gb in enumerate(gts):
                if j in used: continue
                i=_iou_boxes(pb,gb)
                if i>best: best=i; bj=j
            if bj!=-1 and best>=iou_thr: tp+=1; used.add(bj)
            else: fp+=1
        fn = len(gts)-len(used)
        total_tp += tp; total_fp += fp; total_fn += fn

    P = total_tp/(total_tp+total_fp) if total_tp+total_fp>0 else 0.0
    R = total_tp/(total_tp+total_fn) if total_tp+total_fn>0 else 0.0
    F1 = 2*P*R/(P+R) if P+R>0 else 0.0
    Acc= total_tp/(total_tp+total_fp+total_fn) if total_tp+total_fp+total_fn>0 else 0.0
    print("\n==== Object-level (DET) ====")
    print(f"TP={total_tp} FP={total_fp} FN={total_fn}")
    print(f"Precision: {P:.4f}  Recall: {R:.4f}  F1: {F1:.4f}  Accuracy: {Acc:.4f}")

# ---------- 메인 ----------
def train_yolo():
    dataset_path = "./datasets"
    yaml_path    = os.path.join(dataset_path, "data.yaml")
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"data.yaml not found: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    data["path"]  = os.path.abspath(dataset_path)
    data["train"] = "train/images"; data["val"] = "valid/images"
    if os.path.isdir(os.path.join(dataset_path,"test","images")): data["test"] = "test/images"
    else: data.pop("test", None)

    # names → nc 동기화
    names = data.get("names", [])
    if isinstance(names, dict):
        try: names = [names[k] for k in sorted(names.keys(), key=lambda x:int(x))]
        except Exception: names = list(names.values())
        data["names"] = names
    if isinstance(names, list): data["nc"] = len(names)
    nc = int(data.get("nc", 0) or 0)

    # 캐시 삭제 + 잘못된 클래스 라인 제거
    _delete_label_caches(dataset_path)
    for sp in ("train","valid","test"):
        ch, dr = _sanitize_labels_invalid_classes(os.path.join(dataset_path, sp, "labels"), nc)
        if ch or dr: print(f"[INFO] sanitize {sp}: changed_files={ch}, dropped_instances={dr}")

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    # 세그/디텍 자동 판별
    kind = _detect_dataset_kind(os.path.join(dataset_path, "train", "labels"))
    use_seg = (kind["total"]>0) and (kind["seg_ok"]>0) and (kind["det_ok"]==0)
    if use_seg:
        model_name = "yolo11s-seg.pt"
        print("[INFO] Segmentation 라벨 감지 → 세그 모델 사용")
    else:
        model_name = "yolo11s.pt"
        print(f"[WARN] 세그 라벨이 부족/혼합 → Detection 모델로 전환 "
              f"(files={kind['total']}, seg_only={kind['seg_ok']}, det_any={kind['det_ok']})")

    device = "0" if torch.cuda.is_available() else "cpu"
    model  = YOLO(model_name)
    print("모델 로드 완료")

    # 학습
    results = model.train(
        data=yaml_path, epochs=50, imgsz=640, batch=8, device=device,
        project="./utils/model_training/model", name="bolt_model",
        save=True, val=True, plots=True, verbose=True
    )

    # 기본 mAP 출력
    metrics = model.val(data=yaml_path, split="val", imgsz=640, device=device)
    m = getattr(metrics, "seg", None) or getattr(metrics, "box", None)
    if m is not None:
        print(f"mAP50: {getattr(m,'map50', float('nan')):.4f}")
        print(f"mAP50-95: {getattr(m,'map',   float('nan')):.4f}")

    # F1/정확도 평가
    best_path = os.path.join(".", "utils", "model_training", "model", "bolt_model", "weights", "best.pt")
    if os.path.isfile(best_path):
        if use_seg:
            _eval_seg(yaml_path, best_path, split="val", imgsz=640, conf=0.25, iou_thr=0.5, device=device)
        else:
            _eval_det(yaml_path, best_path, split="val", imgsz=640, conf=0.25, iou_thr=0.5, device=device)
    else:
        print(f"[WARN] best.pt 없음: {best_path}")

if __name__ == "__main__":
    train_yolo()

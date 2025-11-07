# modules/evaluate_yolo_seg.py
import os, glob, cv2, numpy as np
from ultralytics import YOLO
import torch
import yaml

def _load_yaml(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _yolo_seg_txt_to_instances(lbl_path, img_w, img_h):
    """YOLO-Seg 라벨(txt) -> [(cls, mask_bool), ...]"""
    instances = []
    if not os.path.isfile(lbl_path):
        return instances
    with open(lbl_path, "r", encoding="utf-8") as f:
        for line in f.read().strip().splitlines():
            if not line:
                continue
            vals = line.strip().split()
            cls = int(float(vals[0]))
            coords = list(map(float, vals[1:]))
            pts = np.array(coords, dtype=np.float32).reshape(-1, 2)
            pts[:, 0] = np.clip(pts[:, 0] * img_w, 0, img_w - 1)
            pts[:, 1] = np.clip(pts[:, 1] * img_h, 0, img_h - 1)
            pts = pts.astype(np.int32)
            mask = np.zeros((img_h, img_w), dtype=np.uint8)
            if len(pts) >= 3:
                cv2.fillPoly(mask, [pts], 1)
            instances.append((cls, mask.astype(bool)))
    return instances

def _pred_results_to_instances(ultra_result, img_w, img_h):
    """Ultralytics 결과 -> [(cls, mask_bool), ...]"""
    instances = []
    if ultra_result.masks is None:
        return instances
    classes = ultra_result.boxes.cls.cpu().numpy().astype(int) if ultra_result.boxes is not None else []
    for i, poly in enumerate(ultra_result.masks.xyn):
        cls = int(classes[i]) if i < len(classes) else 0
        pts = np.array(poly, dtype=np.float32)
        pts[:, 0] = np.clip(pts[:, 0] * img_w, 0, img_w - 1)
        pts[:, 1] = np.clip(pts[:, 1] * img_h, 0, img_h - 1)
        pts = pts.astype(np.int32)
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        if len(pts) >= 3:
            cv2.fillPoly(mask, [pts], 1)
        instances.append((cls, mask.astype(bool)))
    return instances

def _iou(mask_a, mask_b):
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return inter / union

def _greedy_match(preds, gts, iou_thr=0.5):
    """같은 클래스끼리 IoU 최대 매칭(그리디). 반환: tp, fp, fn"""
    used_gt = set()
    tp = 0
    by_cls_pred = {}
    by_cls_gt = {}
    for i, (c, m) in enumerate(preds):
        by_cls_pred.setdefault(c, []).append((i, m))
    for j, (c, m) in enumerate(gts):
        by_cls_gt.setdefault(c, []).append((j, m))
    for cls, pred_list in by_cls_pred.items():
        gt_list = by_cls_gt.get(cls, [])
        gt_indices = [j for j, _ in gt_list]
        gt_masks   = [m for _, m in gt_list]
        for (pi, pmask) in pred_list:
            best_j = -1
            best_iou = 0.0
            for k, (gj, gmask) in enumerate(zip(gt_indices, gt_masks)):
                if gj in used_gt:
                    continue
                iou = _iou(pmask, gmask)
                if iou > best_iou:
                    best_iou = iou
                    best_j = gj
            if best_j != -1 and best_iou >= iou_thr:
                tp += 1
                used_gt.add(best_j)
    fp = len(preds) - tp
    fn = len(gts) - len(used_gt)
    return tp, fp, fn

def evaluate_yolo_seg(yaml_path, model_or_path, split="val", imgsz=640, conf=0.25, iou_thr=0.5, device=None):
    """
    YOLO 세그멘테이션용 객체 단위 F1/정확도 계산.
    """
    data = _load_yaml(yaml_path)
    root = data.get("path", ".")
    images_dir = os.path.join(root, data.get(split, f"{split}/images"))
    labels_dir = images_dir.replace("/images", "/labels").replace("\\images", "\\labels")
    model = model_or_path if isinstance(model_or_path, YOLO) else YOLO(model_or_path)

    image_paths = sorted(
        glob.glob(os.path.join(images_dir, "*.jpg")) +
        glob.glob(os.path.join(images_dir, "*.png")) +
        glob.glob(os.path.join(images_dir, "*.jpeg"))
    )

    total_tp = total_fp = total_fn = 0
    per_class = {}

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        stem = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(labels_dir, f"{stem}.txt")

        gts = _yolo_seg_txt_to_instances(lbl_path, w, h)
        results = model.predict(
            img_path,
            imgsz=imgsz,
            conf=conf,
            device=device if device is not None else (0 if torch.cuda.is_available() else "cpu"),
            verbose=False
        )
        preds = _pred_results_to_instances(results[0], w, h) if len(results) else []

        tp, fp, fn = _greedy_match(preds, gts, iou_thr=iou_thr)
        total_tp += tp; total_fp += fp; total_fn += fn

        # per-class 통계
        from collections import defaultdict
        pred_by_cls = defaultdict(list); gt_by_cls = defaultdict(list)
        for c, m in preds: pred_by_cls[c].append((c, m))
        for c, m in gts:   gt_by_cls[c].append((c, m))
        for c in set(list(pred_by_cls.keys()) + list(gt_by_cls.keys())):
            ctp, cfp, cfn = _greedy_match(pred_by_cls[c], gt_by_cls[c], iou_thr=iou_thr)
            pc = per_class.setdefault(c, [0,0,0])
            pc[0] += ctp; pc[1] += cfp; pc[2] += cfn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy  = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0.0

    print("==== Overall (object-level) ====")
    print(f"TP={total_tp}  FP={total_fp}  FN={total_fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"Accuracy : {accuracy:.4f}  # 정의: TP/(TP+FP+FN)")

    if per_class:
        print("\n==== Per-class (object-level) ====")
        for cls, (tp, fp, fn) in sorted(per_class.items()):
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f = (2*p*r/(p+r)) if (p+r) > 0 else 0.0
            a = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
            print(f"Class {cls:>2d} | TP={tp:>4d} FP={fp:>4d} FN={fn:>4d} | P={p:.4f} R={r:.4f} F1={f:.4f} Acc={a:.4f}")

    return {
        "overall": {"tp": total_tp, "fp": total_fp, "fn": total_fn,
                    "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy},
        "per_class": per_class
    }

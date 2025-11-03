import os, sys, glob
import yaml
from ultralytics import YOLO
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)
if THIS_DIR not in sys.path: sys.path.insert(0, THIS_DIR)

def _delete_label_caches(dataset_path: str):
    for sub in ("train", "valid", "test"):
        p = os.path.join(dataset_path, sub, "labels.cache")
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
    files = glob.glob(os.path.join(labels_dir, "*.txt"))
    seg_ok = det_only = empty = 0
    for p in files:
        with open(p, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if not lines: empty += 1; continue
        has_seg = any(len(ln.split()) >= 7 for ln in lines)
        has_det = any(len(ln.split()) == 5 for ln in lines)
        if has_seg and not has_det: seg_ok += 1
        else: det_only += 1
    return {"total": len(files), "seg_ok": seg_ok, "det_only": det_only, "empty": empty}

def train_yolo():    
    dataset_path = "./datasets"   # <- 현재 폴더 구조에 맞춤
    yaml_path = os.path.join(dataset_path, 'data.yaml')
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"data.yaml not found: {yaml_path}")
    
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    data['path']  = os.path.abspath(dataset_path)
    data['train'] = 'train/images'
    data['val']   = 'valid/images'
    test_images_dir = os.path.join(dataset_path, 'test', 'images')
    if os.path.isdir(test_images_dir): data['test'] = 'test/images'
    else: data.pop('test', None)

    names = data.get('names', [])
    if isinstance(names, dict):
        try: names = [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
        except Exception: names = list(names.values())
        data['names'] = names
    if isinstance(names, list): data['nc'] = len(names)
    nc = int(data.get('nc', 0) or 0)

    _delete_label_caches(dataset_path)
    for sp in ("train", "valid", "test"):
        ch, dr = _sanitize_labels_invalid_classes(os.path.join(dataset_path, sp, "labels"), nc)
        if ch or dr: print(f"[INFO] sanitize {sp}: changed_files={ch}, dropped_instances={dr}")

    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    
    print("데이터셋 정보:")
    print(f"  경로: {data['path']}")
    print(f"  클래스 수: {data.get('nc', 'Unknown')}")
    print(f"  클래스 이름: {data.get('names', 'Unknown')}")
    
    kind = _detect_dataset_kind(os.path.join(dataset_path, "train", "labels"))
    use_seg = (kind["total"] > 0) and (kind["seg_ok"] > 0) and (kind["det_only"] == 0)
    if not use_seg:
        print(f"[WARN] 세그 라벨이 부족/혼합 → 디텍션으로 전환 "
              f"(files={kind['total']}, seg_ok={kind['seg_ok']}, det_only={kind['det_only']})")
    
    device = '0' if torch.cuda.is_available() else 'cpu'
    model_name = 'yolo11s-seg.pt' if use_seg else 'yolo11s.pt'
    model = YOLO(model_name)
    print("모델 로드 완료")
    
    try:
        results = model.train(
            data=yaml_path,
            epochs=50,
            imgsz=640,
            batch=8,
            device=device,
            project='./utils/model_training/model',
            name='bolt_model',
            save=True,
            val=True,
            plots=True,
            verbose=True
        ) 

        metrics = model.val(data=yaml_path, split='val', imgsz=640, device=device)
        m = getattr(metrics, 'seg', None) or getattr(metrics, 'box', None)
        if m is not None:
            print(f"mAP50: {getattr(m,'map50', float('nan')):.4f}")
            print(f"mAP50-95: {getattr(m,'map',   float('nan')):.4f}")
        else:
            print("경고: metrics.seg/metrics.box를 찾지 못했습니다. Ultralytics 버전/모델을 확인하세요.")

        # (옵션) 세그 모델일 때만 객체단위 F1/Acc 평가
        if use_seg:
            try:
                from modules.evaluate_yolo_seg import evaluate_yolo_seg
                best_path = os.path.join('.', 'utils', 'model_training', 'model', 'bolt_model', 'weights', 'best.pt')
                if os.path.isfile(best_path):
                    print("\n[INFO] 학습 완료. F1/Precision/Recall/Accuracy 평가 시작...")
                    evaluate_yolo_seg(
                        yaml_path=yaml_path,
                        model_or_path=best_path,
                        split='val',
                        imgsz=640,
                        conf=0.25,
                        iou_thr=0.5,
                        device=device
                    )
                    print("[INFO] 평가 완료.\n")
                else:
                    print(f"[WARN] best.pt 없음: {best_path}")
            except ModuleNotFoundError:
                print("[WARN] modules.evaluate_yolo_seg 를 찾지 못했습니다. (modules에 파일 생성 필요)")
        else:
            print("[INFO] 디텍션 학습이므로 마스크 기반 F1 평가는 생략.")
        return results
        
    except Exception as e:
        print(f"학습 중 오류 발생: {e}")
        import traceback; traceback.print_exc()
        return None

if __name__ == "__main__":
    train_yolo()

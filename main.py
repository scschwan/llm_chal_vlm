from __future__ import annotations
import os, datetime, random, re
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image

# 한글 폰트 경고 제거 (Windows)
try:
    import matplotlib
    matplotlib.rcParams['font.family'] = 'Malgun Gothic'
    matplotlib.rcParams['axes.unicode_minus'] = False
except Exception:
    pass

from config import (
    OK_DIR, DEF_DIR,
    OK_LABEL, DEF_LABEL,
    DATA_DIR, RESULTS_DIR,
    LLAVA_MODEL, CLIP_MODEL,
)

from modules.clip_search import CLIPSearch  # 인덱스 구축 + 인덱스 검색
from modules.vlm_local import VLM
from modules.image_processor import ImageProcessor

from modules.prompts import (
    build_ok_def_pair_prompt,
    PromptConfig,
)

from modules.ssim_utils import ssim_global, ssim_grid_hints
from modules.preprocess import PreprocessCfg, apply_preprocess
from modules.object_guidance import (
    load_object_json,
    build_weight_map,
    build_prompt_hint,
)

# (선택) 형상/찌그러짐 증거
try:
    from modules.shape_diff import compare_deformation as _compare_deformation
except Exception:
    _compare_deformation = None  # type: ignore

# === ROI 추출 유틸 ===
from modules.region_detect import extract_rois, overlay_boxes, boxes_to_prompt_hints


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")


def _list_images(root: Path) -> List[Path]:
    return [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS]


def _pick_random(root: Path) -> Path:
    files = _list_images(root)
    if not files:
        raise FileNotFoundError(f"이미지 없음: {root}")
    return random.choice(files)


def _to_uint8_pil(img: Image.Image) -> Image.Image:
    """
    SSIM 모듈이 data_range를 받지 않는 경우 경고/에러 방지용:
    PIL → uint8 RGB로 강제 (내부 ssim_utils가 np.uint8로 변환된다고 가정)
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    print("------------------------------------------------------------")
    print("후보(def_front 랜덤) → 기준(ok_front 인덱스) 비교 리포트 파이프라인 시작")
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # 1) 쿼리: def_front에서 랜덤 1장
    left_path = _pick_random(DEF_DIR)   # 쿼리(불량 랜덤)
    print(f"[Query (불량 랜덤)] {left_path.name}")

    # 2) 갤러리 인덱스: ok_front 전체
    device_clip = "cuda" if torch.cuda.is_available() else "cpu"
    clip = CLIPSearch(
        model_id=CLIP_MODEL,
        device=device_clip,
        verbose=True,
    )
    clip.build_index(str(OK_DIR))  # ok_front 임베딩 인덱스 구축

    # 쿼리(def_front → 1장)로 검색
    hit = clip.search_with_index(str(left_path), top_k=1)[0]

    right_path = Path(hit.candidate_path)
    sim_clip = hit.similarity
    print(f"[Top-1({OK_LABEL})] {right_path.name} | CLIP cosine={sim_clip:.4f}")

    # 3) 비교 썸네일 저장 (좌=DEF(쿼리), 우=OK(매칭)) + 화면 표시
    try:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_file = RESULTS_DIR / f"compare_{left_path.stem}.png"
        ImageProcessor.show_pairs(
            str(left_path),
            [str(right_path)],
            sims=[sim_clip],
            suptitle=(
                f"Left=불량({DEF_LABEL}): {left_path.name} | "
                f"Right=정상({OK_LABEL}): {right_path.name} | CLIP={sim_clip:.4f}"
            ),
            thumb_px=640,
            pad_px=8,
            save_path=str(out_file),
            show=True,
        )
        print(f"[시각화 저장] {out_file}")
    except Exception as e:
        print(f"(시각화 실패/건너뜀) {e}")

    # 4) 이미지 로드
    left_img  = Image.open(left_path).convert("RGB")
    right_img = Image.open(right_path).convert("RGB")

    # === 4-1) ROI 생성/통계/시각화 ===
    rois, roi_stats = extract_rois(
        left_img, right_img,
        topk=5,
        min_area_ratio=0.001,     # 0.1%
        max_area_ratio=0.20,      # 20%
        pad_ratio=0.06,
        iou_thr=0.5,
        max_coverage_ratio=0.25,  # 전체 픽셀의 25%까지
        thresh_mode="auto",       # 'fixed'|'percentile'도 가능
        thresh_val=25,
        perc=0.85,
    )

    (RESULTS_DIR / "roi").mkdir(parents=True, exist_ok=True)
    roiL_path = RESULTS_DIR / "roi" / f"{left_path.stem}_roiL.png"
    roiR_path = RESULTS_DIR / "roi" / f"{right_path.stem}_roiR.png"
    overlay_boxes(left_img,  rois, color=(0, 255, 0), width=3).save(roiL_path)
    overlay_boxes(right_img, rois, color=(255, 0, 0), width=3).save(roiR_path)
    print(f"[ROI] count={roi_stats['roi_count']} | coverage={roi_stats['coverage_ratio']*100:.2f}% "
          f"(mask~{roi_stats['mask_ratio']*100:.2f}%)")
    print(f"[ROI] saved: {roiL_path.name}, {roiR_path.name}")

    # 프롬프트용 ROI 힌트(좌표 + 면적%)
    roi_hints = boxes_to_prompt_hints(rois, tag="ROI", total_px=roi_stats["total_px"])

    # 5) SSIM / SSIM 그리드 힌트  (data_range=255로 고정해 경고 제거)
    ssim_score: Optional[float] = None
    grid_hints: List[str] = []
    try:
        ssim_score = ssim_global(_to_uint8_pil(left_img), _to_uint8_pil(right_img), data_range=255)
        grid_hints = ssim_grid_hints(_to_uint8_pil(left_img), _to_uint8_pil(right_img),
                                     topk=5, data_range=255)
        print(f"[Similarity] SSIM(global)={ssim_score:.4f}")
        if grid_hints:
            print(f"[Diff grid hints] {', '.join(grid_hints)}")
    except Exception as e:
        print(f"(SSIM 계산 건너뜀: {e})")

    # 6) shape_diff (찌그러짐/솔리디티 등 정량 로그)
    shape_evi = None
    hotspots: List[str] = []
    try:
        if _compare_deformation is not None:
            shape_evi = _compare_deformation(left_img, right_img)
            print(
                "[Shape] solidity L/R="
                f"{shape_evi['solidity_L']:.3f}/{shape_evi['solidity_R']:.3f}, "
                f"dentΔ={shape_evi['delta_dent']:+.3f}, "
                f"edgeΔ={shape_evi['delta_edge']:+.3f}"
            )
            if shape_evi.get("hotspots"):
                hotspots = list(shape_evi["hotspots"])
                print(f"[Shape hotspots] {', '.join(hotspots)}")
    except Exception as e:
        print(f"(형태 증거 계산 건너뜀: {e})")

    # 7) ROI 가이드 (고정/사전 정의된 관심 객체 목록) + diff ROI 결합
    obj_json = load_object_json(DATA_DIR / "fixed_objects.json")
    hint_fixed = build_prompt_hint(obj_json)

    hint_lines = []
    if hint_fixed:
        hint_lines.append(hint_fixed)
    if roi_hints:
        hint_lines.append("변화 감지 ROI(면적% 포함): " + "; ".join(roi_hints))
    prompt_hint = "\n".join(hint_lines).strip()

    # 8) 전처리 config (좌/우 동일 파이프)
    def _pp_cfg(img: Image.Image):
        weight_map = build_weight_map(img, obj_json)
        return PreprocessCfg(
            brightness=1.03,
            contrast=1.05,
            gamma=1.0,
            normalize_encoder=True,
            weight_map=weight_map,
            handle_alpha=True,
            alpha_bg=(128, 128, 128),
            use_local_contrast=True,
            unsharp_amount=0.5,
            unsharp_radius=1.1,
            unsharp_threshold=2,
        )

    cfg_L = _pp_cfg(left_img)
    cfg_R = _pp_cfg(right_img)

    used = {"n": 0}
    def preprocess_pair(img: Image.Image):
        # 첫 호출 → left cfg, 그다음 호출 → right cfg
        if used["n"] == 0:
            used["n"] += 1
            return apply_preprocess(img, cfg_L)
        else:
            return apply_preprocess(img, cfg_R)

    # 9) Evidence 문자열(LLM 프롬프트 참고 정보)
    parts = [f"CLIP={sim_clip:.3f}"]
    if ssim_score is not None:
        parts.append(f"SSIM={ssim_score:.3f}")
    if shape_evi is not None:
        parts.append(
            "shape("
            f"solL/R={shape_evi['solidity_L']:.2f}/{shape_evi['solidity_R']:.2f}, "
            f"dentΔ={shape_evi['delta_dent']:+.2f}, "
            f"edgeΔ={shape_evi['delta_edge']:+.2f}"
            ")"
        )
        if hotspots:
            parts.append("변형 위치: " + ", ".join(hotspots))
    if grid_hints:
        parts.append("차이 격자 후보: " + ", ".join(grid_hints))
    # ROI 통계 추가
    parts.append(f"ROI_count={roi_stats['roi_count']}, ROI_coverage={roi_stats['coverage_ratio']*100:.2f}%")
    evidence_summary = ". ".join(parts) + "."

    # 10) VLM(LLaVA 등 멀티모달 LLM) 준비
    device_vlm = "cuda" if torch.cuda.is_available() else "cpu"
    vlm = VLM(
        model_id=str(LLAVA_MODEL),
        device=device_vlm,
        persist=False,
        use_bf16=True,
        max_edge=640,
        verbose=True,
    )

    # 11) 프롬프트 구성 (DEF vs OK) — roi_hint에 우리가 만든 ROI 힌트 전달
    prompt = build_ok_def_pair_prompt(
        evidence_summary=evidence_summary,
        roi_hint=prompt_hint,
        grid_hints=grid_hints,
        hotspots=hotspots,
        defect_level="불명확",
        cfg=PromptConfig(
            headings=("INFO","SCENE","DETAIL","추론","STATUS"),
            min_left_points=3,
            min_right_points=3,
            max_points_per_side=6,
            language="ko",
            ignore_text_reading=True,
            status_text="이미지 분석 완료.",
        ),
    )

    # 12) 해상도 튜닝
    hires = 896 if (ssim_score is not None and ssim_score > 0.90) else 768

    # 13) VLM 호출 (좌/우 이미지 비교 설명 생성)
    text = vlm.compare_regions_text(
        str(left_path),
        str(right_path),
        prompt=prompt,
        max_new_tokens=360,
        do_sample=False,         # 필요 시 True/temperature/top_p 조정
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.1,
        preprocess=preprocess_pair,
        max_edge_override=hires,
    )

    raw_report = text.strip()

    # 14) 후처리 (가이드/규칙 제거)
    ban_patterns = [
        r'^\s*\[보조 정보',
        r'^\s*\[규칙',
        r'^\s*\[출력 섹션',
        r'^\s*\[DETAIL 작성 규칙',
        r'^\s*우선순위',
        r'^\s*섹션 제목은',
        r'^\s*모든 출력은',
        r'^\s*형식 유지',
        r'^\s*템플릿',
        r'^\s*예:\s*',
        r'^\s*서식 예시',
    ]

    def is_ban_line(s: str) -> bool:
        for pat in ban_patterns:
            if re.search(pat, s):
                return True
        return False

    cleaned_lines = []
    for ln in raw_report.splitlines():
        s = ln.rstrip()
        if not s.strip():
            continue
        if is_ban_line(s):
            continue
        for token in [
            "(객관적 사실만)", "(한 문장)", "(1문장)",
            "형식으로 서술", "동일 형식으로 서술",
            "최대 2개", "최소 3개 불릿",
        ]:
            s = s.replace(token, "")
        cleaned_lines.append(s)

    cleaned_report = "\n".join(cleaned_lines)

    # fallback: 너무 비면 원본 사용
    header_count = sum(
        1 for l in cleaned_report.splitlines()
        if l.startswith("[INFO]")
        or l.startswith("[SCENE]")
        or l.startswith("[DETAIL]")
        or l.startswith("[추론]")
        or l.startswith("[STATUS]")
    )
    final_report = cleaned_report if (header_count >= 2 and len(cleaned_report) >= 40) else raw_report

    # 15) 출력
    print("\n=== Similarity ===")
    print(f"CLIP cosine: {sim_clip:.4f}")
    if ssim_score is not None:
        print(f"SSIM (global): {ssim_score:.4f}")

    print("\n=== Difference (staged report) ===\n")
    print(final_report)
    print("\n✅ 완료")


if __name__ == "__main__":
    main()

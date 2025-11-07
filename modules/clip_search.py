from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Set
from pathlib import Path

import torch
from PIL import Image
import open_clip

try:
    import faiss  # faiss-cpu
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False


@dataclass
class ClipHit:
    candidate_path: str
    similarity: float


class CLIPSearch:
    """
    - open_clip 기반 실제 임베딩
    - OK 갤러리 인덱싱(build_index) → DEF 쿼리로 검색(search_with_index)
    - FAISS 있으면 Inner Product(IP) 인덱스, 없으면 torch.mv
    - 모델 ID 입력 형식(유연):
        * "ViT-B-32/openai"
        * "ViT-B-32/laion2b_s34b_b79k"
        * "openai/ViT-B-32"   (순서 뒤집힘도 허용)
        * "ViT-B-32"          (pretrained 자동 유추)
        * "openai"            (모델은 ViT-B-32로 매핑)
    """
    def __init__(
        self,
        model_id: str = "ViT-B-32/openai",
        device: str = "cpu",
        verbose: bool = False,
        use_fp16: bool = False,
    ):
        self.device = device
        self.verbose = verbose
        self.model, self.preproc = self._load_clip(model_id, device, use_fp16)

        self.gallery_paths: List[str] = []
        self.gallery_embs: Optional[torch.Tensor] = None  # [N,D] cpu float32 (L2-normed)
        self.faiss_index = None

    # ---------- 유틸: pretrained 목록을 dict[model] -> set(tags) 로 통일 ----------
    def _list_pretrained_map(self) -> Dict[str, Set[str]]:
        """open_clip.list_pretrained()가 dict이든 list이든 모두 흡수해 dict[model]->set(tags)로 반환"""
        m2tags: Dict[str, Set[str]] = {}
        try:
            raw = open_clip.list_pretrained()
            # 케이스 A: dict 형태 {model: [tags...]}
            if isinstance(raw, dict):
                for m, v in raw.items():
                    if isinstance(v, (list, tuple, set)):
                        m2tags[m] = set(v)
                    else:
                        m2tags[m] = {str(v)}
            else:
                # 케이스 B: list of tuples [(model, tag), ...] 혹은 비슷한 구조
                for item in raw:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        m, p = str(item[0]), str(item[1])
                        m2tags.setdefault(m, set()).add(p)
                    else:
                        # 혹시 예외적인 포맷이면 건너뜀
                        continue
        except Exception:
            pass
        return m2tags

    # ---------- 모델/프리트레인 해석 ----------
    def _resolve_model_pretrained(self, model_id: str) -> Tuple[str, str]:
        """
        open_clip 모델/프리트레인 조합을 유연하게 해석.
        실패 시 안전 폴백: ("ViT-B-32", "openai") → 안되면 ("ViT-B-32", "laion2b_s34b_b79k")
        """
        try:
            all_models = set(open_clip.list_models())
        except Exception:
            all_models = set()

        m2tags = self._list_pretrained_map()  # dict[model] -> set(tags)

        def _first_valid_tag(m: str) -> Optional[str]:
            tags = m2tags.get(m)
            if tags:
                # openai가 있으면 우선
                if "openai" in tags:
                    return "openai"
                # laion이 있으면 다음
                for t in ("laion2b_s34b_b79k", "laion2b_s13b_b90k"):
                    if t in tags:
                        return t
                # 아무거나 하나
                return next(iter(tags))
            return None

        model_id = model_id.strip()

        # 1) "A/B" 형태 처리
        if "/" in model_id:
            a, b = [x.strip() for x in model_id.split("/", 1)]
            # (a=model, b=tag) 우선 시도
            if a in all_models:
                tag = b if (b in (m2tags.get(a) or {b})) else (_first_valid_tag(a) or b)
                return a, tag
            # (a=tag, b=model) 뒤집힌 경우
            if b in all_models:
                tag = a if (a in (m2tags.get(b) or {a})) else (_first_valid_tag(b) or a)
                return b, tag

        # 2) 단일 토큰이 모델명인 경우
        if model_id in all_models:
            return model_id, (_first_valid_tag(model_id) or "openai")

        # 3) 단일 토큰이 프리트레인(tag)인 경우 → ViT-B-32에 매핑
        for m, tags in m2tags.items():
            if model_id in tags:
                return "ViT-B-32", model_id

        # 4) 안전 폴백
        return "ViT-B-32", "openai"

    def _load_clip(self, model_id: str, device: str, use_fp16: bool):
        model_name, pretrained = self._resolve_model_pretrained(model_id)

        # 첫 시도
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name=model_name, pretrained=pretrained, device=device
            )
        except Exception:
            # 폴백 1: openai 태그로 재시도
            try:
                model, _, preprocess = open_clip.create_model_and_transforms(
                    model_name=model_name, pretrained="openai", device=device
                )
            except Exception:
                # 폴백 2: ViT-B-32 + laion
                model, _, preprocess = open_clip.create_model_and_transforms(
                    model_name="ViT-B-32", pretrained="laion2b_s34b_b79k", device=device
                )

        model.eval()
        if use_fp16 and device.startswith("cuda"):
            model.half()
        return model, preprocess

    @torch.no_grad()
    def _embed_image(self, path: str) -> torch.Tensor:
        """open-clip 비주얼 임베딩 (L2 정규화, CPU float32 반환)"""
        img = Image.open(path).convert("RGB")
        x = self.preproc(img).unsqueeze(0).to(self.device)
        feat = self.model.encode_image(x)  # [1, D]
        feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-6)
        return feat.squeeze(0).detach().to("cpu", dtype=torch.float32)  # [D]

    def _gather_paths(self, gallery_dir: str) -> List[str]:
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
        return [str(p) for p in Path(gallery_dir).rglob("*") if p.suffix.lower() in exts]

    # ---------- 인덱스(벡터DB) 구축 ----------
    def build_index(self, gallery_dir: str):
        paths = self._gather_paths(gallery_dir)
        if self.verbose:
            print(f"[CLIP] 인덱스 구축: {len(paths)}개 (gallery={Path(gallery_dir).name})")
        if not paths:
            raise FileNotFoundError(f"갤러리 이미지가 없습니다: {gallery_dir}")

        embs = [self._embed_image(p) for p in paths]  # List[[D]]
        embs = torch.stack(embs, dim=0)               # [N, D] cpu float32 (L2-normed)

        self.gallery_paths = paths
        self.gallery_embs = embs

        if _HAS_FAISS:
            d = int(embs.shape[1])
            index = faiss.IndexFlatIP(d)              # 내적 = 코사인(정규화 가정)
            index.add(embs.numpy().astype("float32"))
            self.faiss_index = index
            if self.verbose:
                print(f"[FAISS] IndexFlatIP built (dim={d}, N={embs.shape[0]})")
        else:
            self.faiss_index = None
            if self.verbose:
                print("[FAISS] 미설치 → torch.mv로 검색합니다.")

    # ---------- 인덱스 검색 ----------
    @torch.no_grad()
    def search_with_index(self, query_img_path: str, top_k: int = 1) -> List[ClipHit]:
        assert self.gallery_embs is not None and len(self.gallery_paths) > 0, \
            "인덱스가 없습니다. build_index()를 먼저 호출하세요."

        q = self._embed_image(query_img_path)  # [D]

        if self.faiss_index is not None:
            sims, idx = self.faiss_index.search(q.unsqueeze(0).numpy(), k=min(top_k, len(self.gallery_paths)))
            idx_list = idx[0].tolist()
            sim_list = sims[0].tolist()
            hits = [ClipHit(candidate_path=self.gallery_paths[i], similarity=float(s)) for i, s in zip(idx_list, sim_list)]
        else:
            sims = torch.mv(self.gallery_embs, q)    # [N]
            k = min(top_k, sims.numel())
            top_idx = torch.topk(sims, k=k).indices.tolist()
            hits = [ClipHit(candidate_path=self.gallery_paths[i], similarity=float(sims[i])) for i in top_idx]

        if hits and self.verbose:
            print(f"[Top-1({Path(self.gallery_paths[0]).parent.name})] "
                  f"{Path(hits[0].candidate_path).name} | CLIP cosine={hits[0].similarity:.4f}")
        return hits

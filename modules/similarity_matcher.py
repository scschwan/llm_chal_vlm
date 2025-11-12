"""
TOP-K 유사도 매칭 모듈
기존 modules/clip_search.py를 활용하여 웹 API에서 호출 가능한 형태로 구성
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
from pathlib import Path
import json
import pickle
import torch
from PIL import Image
import open_clip

try:
    import faiss
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False


@dataclass
class SimilarityResult:
    """유사도 검색 결과"""
    query_image: str
    top_k_results: List[Dict[str, Any]]
    total_gallery_size: int
    model_info: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


@dataclass
class ImageMatch:
    """개별 매칭 결과"""
    image_path: str
    image_name: str
    similarity_score: float
    rank: int


class TopKSimilarityMatcher:
    """
    CLIP 기반 TOP-K 유사이미지 검색 모듈
    
    Features:
    - CLIP 모델 기반 이미지 임베딩
    - FAISS 인덱스 활용 고속 검색
    - 인덱스 저장/로드 기능
    - JSON 형태 결과 반환
    """
    
    def __init__(
        self,
        model_id: str = "ViT-B-32/openai",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_fp16: bool = True,
        verbose: bool = False
    ):
        """
        Args:
            model_id: CLIP 모델 ID (예: "ViT-B-32/openai")
            device: 연산 디바이스 ("cuda" or "cpu")
            use_fp16: FP16 사용 여부 (GPU 시 권장)
            verbose: 로그 출력 여부
        """
        self.device = device
        self.verbose = verbose
        self.model_id = model_id
        
        # CLIP 모델 로드
        self.model, self.preprocess = self._load_clip(model_id, device, use_fp16)
        
        # 갤러리 정보
        self.gallery_paths: List[str] = []
        self.gallery_embs: Optional[torch.Tensor] = None
        self.faiss_index = None
        self.index_built = False
        
        if self.verbose:
            print(f"[TopKMatcher] 초기화 완료 - Model: {model_id}, Device: {device}")
    
    def _resolve_model_pretrained(self, model_id: str) -> tuple[str, str]:
        """모델명과 pretrained 태그 파싱"""
        try:
            all_models = set(open_clip.list_models())
        except Exception:
            all_models = set()
        
        model_id = model_id.strip()
        
        # "A/B" 형태 처리
        if "/" in model_id:
            a, b = [x.strip() for x in model_id.split("/", 1)]
            if a in all_models:
                return a, b
            if b in all_models:
                return b, a
        
        # 단일 토큰이 모델명인 경우
        if model_id in all_models:
            return model_id, "openai"
        
        # 안전 폴백
        return "ViT-B-32", "openai"
    
    def _load_clip(self, model_id: str, device: str, use_fp16: bool):
        """CLIP 모델 로드"""
        model_name, pretrained = self._resolve_model_pretrained(model_id)
        
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name=model_name, 
                pretrained=pretrained, 
                device=device
            )
        except Exception:
            # 폴백: ViT-B-32 + openai
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name="ViT-B-32", 
                pretrained="openai", 
                device=device
            )
        
        model.eval()
        if use_fp16 and device.startswith("cuda"):
            model.half()
        
        return model, preprocess
    
    @torch.no_grad()
    def _embed_image(self, image_path: str) -> torch.Tensor:
        """이미지를 임베딩 벡터로 변환 (L2 정규화)"""
        img = Image.open(image_path).convert("RGB")
        x = self.preprocess(img).unsqueeze(0).to(self.device)
        feat = self.model.encode_image(x)
        feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-6)
        return feat.squeeze(0).detach().to("cpu", dtype=torch.float32)
    
    def _gather_image_paths(self, gallery_dir: str) -> List[str]:
        """디렉토리에서 이미지 파일 수집"""
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
        paths = [str(p) for p in Path(gallery_dir).rglob("*") if p.suffix.lower() in exts]
        return sorted(paths)
    
    def build_index(self, gallery_dir: str, recursive: bool = True) -> Dict[str, Any]:
        """
        갤러리 이미지 인덱스 구축
        
        Args:
            gallery_dir: 갤러리 이미지 디렉토리 경로
            
        Returns:
            인덱스 구축 정보 딕셔너리
        """
        paths = self._gather_image_paths(gallery_dir)
        
        if not paths:
            raise FileNotFoundError(f"갤러리 이미지가 없습니다: {gallery_dir}")
        
        if self.verbose:
            print(f"[TopKMatcher] 인덱스 구축 시작: {len(paths)}개 이미지")
        
        # 임베딩 생성
        embs = []
        for i, path in enumerate(paths):
            if self.verbose and (i + 1) % 100 == 0:
                print(f"  진행: {i+1}/{len(paths)}")
            embs.append(self._embed_image(path))
        
        embs = torch.stack(embs, dim=0)  # [N, D]
        
        self.gallery_paths = paths
        self.gallery_embs = embs
        
        # FAISS 인덱스 구축
        if _HAS_FAISS:
            d = int(embs.shape[1])
            index = faiss.IndexFlatIP(d)  # Inner Product (코사인 유사도)
            index.add(embs.numpy().astype("float32"))
            self.faiss_index = index
            if self.verbose:
                print(f"[TopKMatcher] FAISS 인덱스 구축 완료 (dim={d})")
        else:
            self.faiss_index = None
            if self.verbose:
                print("[TopKMatcher] FAISS 미설치 - Torch 기반 검색 사용")
        
        self.index_built = True
        
        return {
            "status": "success",
            "gallery_dir": gallery_dir,
            "num_images": len(paths),
            "embedding_dim": int(embs.shape[1]),
            "faiss_enabled": _HAS_FAISS
        }
    
    def save_index(self, save_dir: str):
        """
        인덱스를 파일로 저장 (재사용)
        
        Args:
            save_dir: 저장 디렉토리 경로
        """
        if not self.index_built:
            raise RuntimeError("인덱스가 구축되지 않았습니다. build_index()를 먼저 호출하세요.")
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 경로 및 임베딩 저장
        data = {
            "gallery_paths": self.gallery_paths,
            "gallery_embs": self.gallery_embs,
            "model_id": self.model_id
        }
        torch.save(data, save_path / "index_data.pt")
        
        # FAISS 인덱스 저장
        if self.faiss_index is not None:
            faiss.write_index(self.faiss_index, str(save_path / "faiss_index.bin"))
        
        if self.verbose:
            print(f"[TopKMatcher] 인덱스 저장 완료: {save_path}")
    
    def load_index(self, load_dir: str):
        """
        저장된 인덱스 로드
        
        Args:
            load_dir: 인덱스 디렉토리 경로
        """
        load_path = Path(load_dir)
        
        # 데이터 로드
        data = torch.load(load_path / "index_data.pt")
        self.gallery_paths = data["gallery_paths"]
        self.gallery_embs = data["gallery_embs"]
        
        # FAISS 인덱스 로드
        if _HAS_FAISS and (load_path / "faiss_index.bin").exists():
            self.faiss_index = faiss.read_index(str(load_path / "faiss_index.bin"))
        
        self.index_built = True
        
        if self.verbose:
            print(f"[TopKMatcher] 인덱스 로드 완료: {len(self.gallery_paths)}개 이미지")
    
    def search(
        self, 
        query_image_path: str, 
        top_k: int = 5
    ) -> SimilarityResult:
        """
        쿼리 이미지와 유사한 TOP-K 이미지 검색
        
        Args:
            query_image_path: 쿼리 이미지 경로
            top_k: 반환할 상위 K개 결과 수
            
        Returns:
            SimilarityResult 객체 (JSON 변환 가능)
        """
        if not self.index_built:
            raise RuntimeError("인덱스가 구축되지 않았습니다. build_index() 또는 load_index()를 먼저 호출하세요.")
        
        # 쿼리 임베딩
        q_emb = self._embed_image(query_image_path)
        
        # 검색
        if self.faiss_index is not None:
            # FAISS 검색
            k = min(top_k, len(self.gallery_paths))
            sims, indices = self.faiss_index.search(
                q_emb.unsqueeze(0).numpy(), 
                k=k
            )
            idx_list = indices[0].tolist()
            sim_list = sims[0].tolist()
        else:
            # Torch 검색
            sims = torch.mv(self.gallery_embs, q_emb)
            k = min(top_k, sims.numel())
            top_values, top_indices = torch.topk(sims, k=k)
            idx_list = top_indices.tolist()
            sim_list = top_values.tolist()
        
        # 결과 구성
        matches = []
        for rank, (idx, sim) in enumerate(zip(idx_list, sim_list), start=1):
            path = self.gallery_paths[idx]
            matches.append({
                "rank": rank,
                "image_path": path,
                "image_name": Path(path).name,
                "similarity_score": float(sim)
            })
        
        result = SimilarityResult(
            query_image=query_image_path,
            top_k_results=matches,
            total_gallery_size=len(self.gallery_paths),
            model_info=self.model_id
        )
        
        if self.verbose:
            print(f"[TopKMatcher] 검색 완료 - Top-1: {matches[0]['image_name']} "
                  f"(similarity: {matches[0]['similarity_score']:.4f})")
        
        return result


# API 용 헬퍼 함수들
def create_matcher(
    model_id: str = "ViT-B-32/openai",
    device: str = "auto",
    use_fp16: bool = True,
    verbose: bool = True
) -> TopKSimilarityMatcher:
    """
    매처 인스턴스 생성 헬퍼
    
    Args:
        model_id: CLIP 모델 ID
        device: "auto", "cuda", "cpu"
        use_fp16: FP16 사용 여부
        verbose: 로그 출력 여부
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    return TopKSimilarityMatcher(
        model_id=model_id,
        device=device,
        use_fp16=use_fp16,
        verbose=verbose
    )


def search_similar_images(
    matcher: TopKSimilarityMatcher,
    query_image: str,
    top_k: int = 5,
    return_json: bool = True
) -> str | SimilarityResult:
    """
    유사 이미지 검색 (단순 래퍼)
    
    Args:
        matcher: TopKSimilarityMatcher 인스턴스
        query_image: 쿼리 이미지 경로
        top_k: 상위 K개
        return_json: JSON 문자열 반환 여부
        
    Returns:
        JSON 문자열 또는 SimilarityResult 객체
    """
    result = matcher.search(query_image, top_k=top_k)
    
    if return_json:
        return result.to_json()
    return result


# 사용 예제
if __name__ == "__main__":
    # 1. 매처 생성
    matcher = create_matcher(
        model_id="ViT-B-32/openai",
        device="auto",
        verbose=True
    )
    
    # 2. 인덱스 구축 (최초 1회)
    gallery_dir = "./data/ok_front"
    info = matcher.build_index(gallery_dir)
    print(f"인덱스 구축 완료: {info}")
    
    # 3. 인덱스 저장 (선택)
    # matcher.save_index("./index_cache")
    
    # 4. 검색
    query_image = "./data/def_front/sample_defect.jpg"
    result = matcher.search(query_image, top_k=5)
    
    # 5. 결과 출력
    print("\n=== 검색 결과 ===")
    print(result.to_json())

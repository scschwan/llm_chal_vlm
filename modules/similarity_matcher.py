"""
CLIP 기반 TOP-K 유사도 매칭 (배치 처리 + 다중 프로세스 최적화)
"""
import torch
import open_clip
from PIL import Image
from pathlib import Path
from typing import List, Union, Optional, Dict
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False
    print("⚠️  FAISS 미설치: 검색 속도가 느릴 수 있습니다. 설치 권장: pip install faiss-gpu")


@dataclass
class SearchResult:
    """검색 결과"""
    top_k_results: List[Dict]
    total_gallery_size: int
    model_info: Dict


class ImageDataset(Dataset):
    """이미지 데이터셋 (배치 로딩용)"""
    
    def __init__(self, image_paths: List[Path], target_size: tuple = (224, 224)):
        self.image_paths = image_paths
        self.target_size = target_size
        
        # ✅ 전처리 파이프라인 (리사이즈 최적화)
        self.transform = transforms.Compose([
            transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        try:
            # 이미지 로드 (RGB 변환)
            image = Image.open(img_path).convert('RGB')
            
            # 전처리 적용
            image_tensor = self.transform(image)
            
            return image_tensor, str(img_path)
            
        except Exception as e:
            print(f"⚠️  이미지 로드 실패: {img_path} - {e}")
            # 빈 이미지 반환
            return torch.zeros(3, *self.target_size), str(img_path)


class TopKSimilarityMatcher:
    """
    CLIP 기반 TOP-K 유사도 매칭 (최적화 버전)
    
    개선 사항:
    - 배치 처리 (batch_size=32)
    - 다중 프로세스 이미지 로딩 (num_workers=4)
    - 이미지 리사이즈 최적화 (224x224)
    """
    
    def __init__(
        self,
        model_id: str = "ViT-B-32",
        pretrained: str = "openai",
        device: str = "auto",
        use_fp16: bool = False,
        batch_size: int = 32,
        num_workers: int = 4,
        verbose: bool = True
    ):
        self.model_id = model_id
        self.pretrained = pretrained
        self.use_fp16 = use_fp16
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.verbose = verbose
        
        # 디바이스 설정
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        if self.verbose:
            print(f"[TopKMatcher] 디바이스: {self.device}")
            print(f"[TopKMatcher] 배치 크기: {self.batch_size}")
            print(f"[TopKMatcher] 워커 수: {self.num_workers}")
        
        # CLIP 모델 로드
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_id,
            pretrained=pretrained,
            device=self.device
        )
        self.model.eval()
        
        # FP16 설정
        if self.use_fp16 and self.device == "cuda":
            self.model = self.model.half()
            if self.verbose:
                print("[TopKMatcher] FP16 모드 활성화")
        
        # 갤러리 데이터
        self.gallery_paths: Optional[List[str]] = None
        self.gallery_embs: Optional[torch.Tensor] = None
        self.faiss_index = None
        self.index_built = False
    
    def build_index(self, gallery_dir: Union[str, Path]) -> Dict:
        """
        갤러리 인덱스 구축 (배치 처리)
        
        Args:
            gallery_dir: 갤러리 이미지 디렉토리
        
        Returns:
            구축 정보 딕셔너리
        """
        gallery_dir = Path(gallery_dir)
        
        if not gallery_dir.exists():
            raise FileNotFoundError(f"디렉토리를 찾을 수 없습니다: {gallery_dir}")
        
        # 이미지 파일 수집
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
            image_paths.extend(gallery_dir.rglob(ext))
        
        image_paths = sorted(image_paths)
        
        if len(image_paths) == 0:
            raise ValueError(f"이미지를 찾을 수 없습니다: {gallery_dir}")
        
        if self.verbose:
            print(f"[TopKMatcher] 인덱스 구축 시작: {len(image_paths)}개 이미지")
        
        # ✅ 데이터셋 생성
        dataset = ImageDataset(image_paths, target_size=(224, 224))
        
        # ✅ DataLoader 생성 (다중 프로세스)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if self.device == "cuda" else False,
            prefetch_factor=2 if self.num_workers > 0 else None
        )
        
        # ✅ 배치 임베딩 생성
        all_embeddings = []
        all_paths = []
        
        with torch.no_grad():
            for batch_images, batch_paths in tqdm(
                dataloader,
                desc="임베딩 생성",
                disable=not self.verbose
            ):
                # GPU로 이동
                batch_images = batch_images.to(self.device)
                
                # FP16 변환
                if self.use_fp16 and self.device == "cuda":
                    batch_images = batch_images.half()
                
                # 배치 임베딩 생성
                batch_embs = self.model.encode_image(batch_images)
                batch_embs = batch_embs / batch_embs.norm(dim=-1, keepdim=True)
                
                # CPU로 이동
                all_embeddings.append(batch_embs.cpu().float())
                all_paths.extend(batch_paths)
        
        # 결합
        self.gallery_embs = torch.cat(all_embeddings, dim=0)
        self.gallery_paths = all_paths
        
        # FAISS 인덱스 구축
        if _HAS_FAISS:
            embeddings_np = self.gallery_embs.numpy()
            dim = embeddings_np.shape[1]
            
            self.faiss_index = faiss.IndexFlatIP(dim)  # Inner Product (코사인 유사도)
            self.faiss_index.add(embeddings_np)
            
            if self.verbose:
                print(f"[TopKMatcher] FAISS 인덱스 구축 완료 (dim={dim})")
        
        self.index_built = True
        
        return {
            "num_images": len(self.gallery_paths),
            "embedding_dim": self.gallery_embs.shape[1],
            "device": self.device,
            "has_faiss": _HAS_FAISS
        }
    
    def save_index(self, save_dir: Union[str, Path]):
        """인덱스 저장"""
        if not self.index_built:
            raise RuntimeError("인덱스가 구축되지 않았습니다")
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # PyTorch 데이터 저장
        torch.save({
            "gallery_paths": self.gallery_paths,
            "gallery_embs": self.gallery_embs,
            "model_id": self.model_id,
            "pretrained": self.pretrained
        }, save_dir / "index_data.pt")
        
        # FAISS 인덱스 저장
        if self.faiss_index is not None:
            faiss.write_index(self.faiss_index, str(save_dir / "faiss_index.bin"))
        
        if self.verbose:
            print(f"[TopKMatcher] 인덱스 저장 완료: {save_dir}")
    
    def load_index(self, load_dir: Union[str, Path]):
        """인덱스 로드"""
        load_dir = Path(load_dir)
        
        if not (load_dir / "index_data.pt").exists():
            raise FileNotFoundError(f"인덱스 파일을 찾을 수 없습니다: {load_dir}")
        
        # PyTorch 데이터 로드
        data = torch.load(load_dir / "index_data.pt", map_location="cpu")
        self.gallery_paths = data["gallery_paths"]
        self.gallery_embs = data["gallery_embs"]
        
        # FAISS 인덱스 로드
        faiss_path = load_dir / "faiss_index.bin"
        if faiss_path.exists() and _HAS_FAISS:
            self.faiss_index = faiss.read_index(str(faiss_path))
        
        self.index_built = True
        
        if self.verbose:
            print(f"[TopKMatcher] 인덱스 로드 완료: {len(self.gallery_paths)}개 이미지")
    
    def search(
        self,
        query_image_path: Union[str, Path],
        top_k: int = 5
    ) -> SearchResult:
        """
        유사 이미지 검색
        
        Args:
            query_image_path: 쿼리 이미지 경로
            top_k: 상위 K개 결과
        
        Returns:
            SearchResult 객체
        """
        if not self.index_built:
            raise RuntimeError("인덱스가 구축되지 않았습니다")
        
        # 쿼리 이미지 로드 및 전처리
        query_image = Image.open(query_image_path).convert('RGB')
        
        # ✅ 리사이즈 최적화 (CLIP 입력 크기)
        query_image = query_image.resize((224, 224), Image.BILINEAR)
        
        query_tensor = self.preprocess(query_image).unsqueeze(0).to(self.device)
        
        if self.use_fp16 and self.device == "cuda":
            query_tensor = query_tensor.half()
        
        # 쿼리 임베딩 생성
        with torch.no_grad():
            query_emb = self.model.encode_image(query_tensor)
            query_emb = query_emb / query_emb.norm(dim=-1, keepdim=True)
            query_emb = query_emb.cpu().float()
        
        # 검색
        if self.faiss_index is not None:
            # FAISS 검색
            query_np = query_emb.numpy()
            similarities, indices = self.faiss_index.search(query_np, top_k)
            similarities = similarities[0]
            indices = indices[0]
        else:
            # PyTorch 검색
            similarities = (query_emb @ self.gallery_embs.T).squeeze(0)
            similarities, indices = torch.topk(similarities, k=top_k)
            similarities = similarities.numpy()
            indices = indices.numpy()
        
        # 결과 구성
        results = []
        for idx, sim in zip(indices, similarities):
            results.append({
                "image_path": self.gallery_paths[idx],
                "similarity_score": float(sim)
            })
        
        return SearchResult(
            top_k_results=results,
            total_gallery_size=len(self.gallery_paths),
            model_info={
                "model_id": self.model_id,
                "pretrained": self.pretrained,
                "device": self.device
            }
        )


def create_matcher(
    model_id: str = "ViT-B-32/openai",
    device: str = "auto",
    use_fp16: bool = False,
    batch_size: int = 32,
    num_workers: int = 4,
    verbose: bool = True
) -> TopKSimilarityMatcher:
    """
    매처 생성 헬퍼 함수
    
    Args:
        model_id: CLIP 모델 ID (예: "ViT-B-32/openai")
        device: 디바이스 ("auto", "cuda", "cpu")
        use_fp16: FP16 사용 여부
        batch_size: 배치 크기 (기본 32)
        num_workers: 워커 수 (기본 4)
        verbose: 로그 출력 여부
    
    Returns:
        TopKSimilarityMatcher 인스턴스
    """
    # model_id 파싱 (ViT-B-32/openai → model=ViT-B-32, pretrained=openai)
    if "/" in model_id:
        model, pretrained = model_id.split("/", 1)
    else:
        model = model_id
        pretrained = "openai"
    
    return TopKSimilarityMatcher(
        model_id=model,
        pretrained=pretrained,
        device=device,
        use_fp16=use_fp16,
        batch_size=batch_size,
        num_workers=num_workers,
        verbose=verbose
    )
"""
CLIP 기반 TOP-K 유사도 매칭 V2 (DB 메타데이터 기반)

기존 similarity_matcher.py와의 차이점:
- 파일명 파싱 대신 DB에서 메타데이터 조회
- Object Storage URL 포함
- 제품명, 불량명 등 풍부한 메타데이터 제공
"""
import torch
import open_clip
from PIL import Image
from pathlib import Path
from typing import List, Union, Optional, Dict, Any
from dataclasses import dataclass, asdict
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import json
from sqlalchemy.orm import Session

try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False
    print("⚠️  FAISS 미설치: 검색 속도가 느릴 수 있습니다. 설치 권장: pip install faiss-gpu")


@dataclass
class ImageMetadata:
    """이미지 메타데이터 구조체"""
    image_id: int
    local_path: str
    storage_url: str
    product_id: int
    product_code: str
    product_name: str
    defect_type_id: Optional[int]
    defect_code: str
    defect_name: str
    image_type: str
    file_name: str
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return asdict(self)


@dataclass
class SearchResultV2:
    """검색 결과 V2"""
    results: List[Dict[str, Any]]
    total_gallery_size: int
    model_info: Dict[str, Any]
    index_type: str


class ImageDatasetV2(Dataset):
    """이미지 데이터셋 V2 (메타데이터 포함)"""
    
    def __init__(self, metadata_list: List[Dict], target_size: tuple = (224, 224)):
        self.metadata_list = metadata_list
        self.target_size = target_size
        
        # 전처리 파이프라인
        self.transform = transforms.Compose([
            transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])
    
    def __len__(self):
        return len(self.metadata_list)
    
    def __getitem__(self, idx):
        metadata = self.metadata_list[idx]
        local_path = metadata['local_path']
        
        try:
            # 이미지 로드
            image = Image.open(local_path).convert('RGB')
            
            # 전처리 적용
            image_tensor = self.transform(image)
            
            return image_tensor, idx
            
        except Exception as e:
            print(f"⚠️  이미지 로드 실패: {local_path} - {e}")
            # 빈 이미지 반환
            return torch.zeros(3, *self.target_size), idx


class TopKSimilarityMatcherV2:
    """
    CLIP 기반 TOP-K 유사도 매칭 V2 (DB 메타데이터 기반)
    
    주요 기능:
    - DB에서 이미지 메타데이터 조회
    - Object Storage URL 포함
    - 제품명, 불량명 등 풍부한 메타데이터 제공
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
            print(f"[MatcherV2] 디바이스: {self.device}")
            print(f"[MatcherV2] 배치 크기: {self.batch_size}")
            print(f"[MatcherV2] 워커 수: {self.num_workers}")
        
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
                print("[MatcherV2] FP16 모드 활성화")
        
        # 갤러리 데이터
        self.gallery_metadata: Optional[List[ImageMetadata]] = None
        self.gallery_embs: Optional[torch.Tensor] = None
        self.faiss_index = None
        self.index_built = False
        self.index_type = None  # 'normal' or 'defect'
    
    def build_index_from_db(
        self, 
        db_session: Session, 
        image_type: str
    ) -> Dict[str, Any]:
        """
        DB에서 이미지 메타데이터 조회 → 인덱스 구축
        
        Args:
            db_session: SQLAlchemy DB 세션
            image_type: 이미지 타입 ('normal' or 'defect')
        
        Returns:
            구축 정보 딕셔너리
        """
        from web.database.crud import fetch_image_metadata_for_index
        
        if self.verbose:
            print(f"[MatcherV2] DB에서 메타데이터 조회 중 (image_type={image_type})...")
        
        # DB에서 메타데이터 조회
        metadata_list = fetch_image_metadata_for_index(db_session, image_type)
        
        if len(metadata_list) == 0:
            raise ValueError(f"DB에 {image_type} 타입 이미지가 없습니다")
        
        if self.verbose:
            print(f"[MatcherV2] {len(metadata_list)}개 이미지 메타데이터 조회 완료")
        
        # 로컬 파일 존재 여부 확인
        valid_metadata = []
        for meta in metadata_list:
            local_path = Path(meta['local_path'])
            if local_path.exists():
                valid_metadata.append(meta)
            else:
                print(f"⚠️  로컬 파일 없음: {local_path}")
        
        if len(valid_metadata) == 0:
            raise ValueError("유효한 로컬 이미지 파일이 없습니다")
        
        if self.verbose:
            print(f"[MatcherV2] 유효한 이미지: {len(valid_metadata)}개")
        
        # 데이터셋 생성
        dataset = ImageDatasetV2(valid_metadata, target_size=(224, 224))
        
        # DataLoader 생성
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if self.device == "cuda" else False,
            prefetch_factor=2 if self.num_workers > 0 else None
        )
        
        # 배치 임베딩 생성
        all_embeddings = []
        
        with torch.no_grad():
            for batch_images, batch_indices in tqdm(
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
        
        # 결합
        self.gallery_embs = torch.cat(all_embeddings, dim=0)
        
        # 메타데이터 객체 변환
        self.gallery_metadata = [
            ImageMetadata(**meta) for meta in valid_metadata
        ]
        
        # FAISS 인덱스 구축
        if _HAS_FAISS:
            embeddings_np = self.gallery_embs.numpy()
            dim = embeddings_np.shape[1]
            
            self.faiss_index = faiss.IndexFlatIP(dim)
            self.faiss_index.add(embeddings_np)
            
            if self.verbose:
                print(f"[MatcherV2] FAISS 인덱스 구축 완료 (dim={dim})")
        
        self.index_built = True
        self.index_type = image_type
        
        return {
            "num_images": len(self.gallery_metadata),
            "embedding_dim": self.gallery_embs.shape[1],
            "device": self.device,
            "has_faiss": _HAS_FAISS,
            "index_type": image_type
        }
    
    def save_index(self, save_dir: Union[str, Path]):
        """인덱스 + 메타데이터 저장"""
        if not self.index_built:
            raise RuntimeError("인덱스가 구축되지 않았습니다")
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. FAISS 인덱스 저장
        if self.faiss_index is not None:
            faiss.write_index(
                self.faiss_index, 
                str(save_dir / "faiss_index.bin")
            )
        
        # 2. 임베딩 저장
        torch.save(
            self.gallery_embs, 
            save_dir / "embeddings.pt"
        )
        
        # 3. 메타데이터 저장 (JSON)
        metadata_list = [meta.to_dict() for meta in self.gallery_metadata]
        
        with open(save_dir / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump({
                "index_type": self.index_type,
                "num_images": len(metadata_list),
                "model_id": self.model_id,
                "pretrained": self.pretrained,
                "metadata": metadata_list
            }, f, ensure_ascii=False, indent=2)
        
        if self.verbose:
            print(f"[MatcherV2] 인덱스 저장 완료: {save_dir}")
    
    def load_index(self, load_dir: Union[str, Path]):
        """인덱스 + 메타데이터 로드"""
        load_dir = Path(load_dir)
        
        if not (load_dir / "metadata.json").exists():
            raise FileNotFoundError(f"메타데이터 파일 없음: {load_dir / 'metadata.json'}")
        
        # 1. 메타데이터 로드
        with open(load_dir / "metadata.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.index_type = data.get("index_type")
        
        # ImageMetadata 객체 변환
        self.gallery_metadata = [
            ImageMetadata(**meta) for meta in data["metadata"]
        ]
        
        # 2. 임베딩 로드
        self.gallery_embs = torch.load(
            load_dir / "embeddings.pt", 
            map_location="cpu"
        )
        
        # 3. FAISS 인덱스 로드
        faiss_path = load_dir / "faiss_index.bin"
        if faiss_path.exists() and _HAS_FAISS:
            self.faiss_index = faiss.read_index(str(faiss_path))
        
        self.index_built = True
        
        if self.verbose:
            print(f"[MatcherV2] 인덱스 로드 완료: {len(self.gallery_metadata)}개 이미지")
    
    def search(
        self,
        query_image_path: Union[str, Path],
        top_k: int = 5
    ) -> SearchResultV2:
        """
        유사 이미지 검색 (메타데이터 포함)
        
        Args:
            query_image_path: 쿼리 이미지 경로
            top_k: 상위 K개 결과
        
        Returns:
            SearchResultV2 객체 (메타데이터 포함)
        """
        if not self.index_built:
            raise RuntimeError("인덱스가 구축되지 않았습니다")
        
        # 쿼리 이미지 로드 및 전처리
        query_image = Image.open(query_image_path).convert('RGB')
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
        
        # 결과 구성 (메타데이터 포함)
        results = []
        for idx, sim in zip(indices, similarities):
            metadata = self.gallery_metadata[idx]
            
            results.append({
                "image_id": metadata.image_id,
                "similarity_score": float(sim),
                "local_path": metadata.local_path,
                "storage_url": metadata.storage_url,
                "product_id": metadata.product_id,
                "product_code": metadata.product_code,
                "product_name": metadata.product_name,
                "defect_type_id": metadata.defect_type_id,
                "defect_code": metadata.defect_code,
                "defect_name": metadata.defect_name,
                "image_type": metadata.image_type,
                "file_name": metadata.file_name
            })
        
        return SearchResultV2(
            results=results,
            total_gallery_size=len(self.gallery_metadata),
            model_info={
                "model_id": self.model_id,
                "pretrained": self.pretrained,
                "device": self.device
            },
            index_type=self.index_type
        )


def create_matcher_v2(
    model_id: str = "ViT-B-32/openai",
    device: str = "auto",
    use_fp16: bool = False,
    batch_size: int = 32,
    num_workers: int = 4,
    verbose: bool = True
) -> TopKSimilarityMatcherV2:
    """
    매처 V2 생성 헬퍼 함수
    
    Args:
        model_id: CLIP 모델 ID (예: "ViT-B-32/openai")
        device: 디바이스 ("auto", "cuda", "cpu")
        use_fp16: FP16 사용 여부
        batch_size: 배치 크기
        num_workers: 워커 수
        verbose: 로그 출력 여부
    
    Returns:
        TopKSimilarityMatcherV2 인스턴스
    """
    # model_id 파싱
    if "/" in model_id:
        model, pretrained = model_id.split("/", 1)
    else:
        model = model_id
        pretrained = "openai"
    
    return TopKSimilarityMatcherV2(
        model_id=model,
        pretrained=pretrained,
        device=device,
        use_fp16=use_fp16,
        batch_size=batch_size,
        num_workers=num_workers,
        verbose=verbose
    )
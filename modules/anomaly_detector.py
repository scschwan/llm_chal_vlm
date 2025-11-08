"""
PatchCore 기반 Anomaly Detection 모듈
modules/patchCore/infer.py를 API에서 호출 가능하도록 래핑
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
import cv2

# patchCore 모듈 import 를 위한 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from modules.patchCore.infer import (
    ResNet50Multi,
    preprocess_bgr_to_tensor,
    feat_to_patches,
    knn_min_dist_faiss,
    upsample_to_img
)


class AnomalyDetector:
    """
    PatchCore 메모리뱅크 기반 이상 검출기
    """
    
    def __init__(
        self,
        bank_base_dir: str = "../data/patchCore",
        device: str = "auto",
        verbose: bool = False
    ):
        """
        Args:
            bank_base_dir: 메모리뱅크 루트 디렉토리 (prod1, prod2 등 포함)
            device: "auto", "cuda", "cpu"
            verbose: 로그 출력 여부
        """
        self.bank_base_dir = Path(bank_base_dir)
        self.verbose = verbose
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # 캐시: 제품별 메모리뱅크 (최근 2개만 유지)
        self._bank_cache = {}
        self._cache_order = []
        self._max_cache = 2
        
        if self.verbose:
            print(f"[AnomalyDetector] 초기화 - Device: {self.device}")
    
    # 변경 후 (더 강건하게):
    def extract_product_name(self, filename: str) -> str:
        """
        파일명에서 제품명 추출
        예시: prod1_burr_10.jpeg → prod1
            cast_001.png → cast
        """
        name = Path(filename).stem
        
        # '_'로 split하여 첫 번째 토큰을 제품명으로 사용
        parts = name.split('_')
        if len(parts) > 0:
            product_name = parts[0]
        else:
            product_name = "unknown"
        
        # 실제 메모리뱅크 디렉토리에 존재하는지 확인
        available_products = [d.name for d in self.bank_base_dir.iterdir() if d.is_dir()]
        
        # 추출된 제품명이 없으면 기본값 사용
        if product_name not in available_products:
            if self.verbose:
                print(f"⚠️ 제품명 '{product_name}'이 메모리뱅크에 없음. 사용 가능: {available_products}")
            
            # 기본값으로 첫 번째 제품 사용
            if available_products:
                product_name = available_products[0]
                if self.verbose:
                    print(f"   → 기본 제품 '{product_name}' 사용")
        
        if self.verbose:
            print(f"[파일명 파싱] {filename} → 제품명: {product_name}")
        
        return product_name
    
    def load_memory_bank(self, product_name: str) -> Dict[str, Any]:
        """
        제품별 메모리뱅크 로드 (캐싱)
        
        Args:
            product_name: 제품명 (예: prod1)
            
        Returns:
            {
                "bank": np.ndarray,      # [M, D]
                "config": dict,          # bank_config.json
                "tau": dict              # tau.json
            }
        """
        # 캐시 확인
        if product_name in self._bank_cache:
            if self.verbose:
                print(f"[캐시 HIT] {product_name} 메모리뱅크")
            return self._bank_cache[product_name]
        
        # 디렉토리 확인
        bank_dir = self.bank_base_dir / product_name
        if not bank_dir.exists():
            raise FileNotFoundError(
                f"메모리뱅크 디렉토리를 찾을 수 없습니다: {bank_dir}\n"
                f"사용 가능한 제품: {[d.name for d in self.bank_base_dir.iterdir() if d.is_dir()]}"
            )
        
        # 파일 확인
        bank_path = bank_dir / "memory_bank.pt"
        config_path = bank_dir / "bank_config.json"
        tau_path = bank_dir / "tau.json"
        
        for path in [bank_path, config_path, tau_path]:
            if not path.exists():
                raise FileNotFoundError(f"필수 파일이 없습니다: {path}")
        
        # 로드
        bank = torch.load(bank_path, map_location="cpu").cpu().numpy().astype(np.float32, copy=False)
        config = json.load(open(config_path, "r"))
        tau = json.load(open(tau_path, "r"))
        
        result = {
            "bank": bank,
            "config": config,
            "tau": tau
        }
        
        # 캐시 저장 (LRU)
        self._bank_cache[product_name] = result
        self._cache_order.append(product_name)
        
        # 캐시 제한
        if len(self._cache_order) > self._max_cache:
            oldest = self._cache_order.pop(0)
            del self._bank_cache[oldest]
            if self.verbose:
                print(f"[캐시 제거] {oldest}")
        
        if self.verbose:
            print(f"[메모리뱅크 로드] {product_name} - Bank shape: {bank.shape}")
        
        return result
    
    def detect(
        self,
        test_image_path: str,
        product_name: Optional[str] = None,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        이상 검출 수행
        
        Args:
            test_image_path: 테스트 이미지 경로
            product_name: 제품명 (None이면 파일명에서 자동 추출)
            output_dir: 출력 디렉토리 (None이면 임시 디렉토리 사용)
            
        Returns:
            {
                "product_name": str,
                "image_score": float,
                "pixel_tau": float,
                "image_tau": float,
                "heatmap_path": str,
                "mask_path": str,
                "overlay_path": str
            }
        """
        # 제품명 자동 추출
        if product_name is None:
            filename = Path(test_image_path).name
            product_name = self.extract_product_name(filename)
        
        # 메모리뱅크 로드
        bank_data = self.load_memory_bank(product_name)
        bank = bank_data["bank"]
        config = bank_data["config"]
        tau = bank_data["tau"]
        
        # 모델 로드 (필요 시 캐싱 가능)
        layers = tuple(config.get("layers", ["layer2", "layer3", "layer4"]))
        model = ResNet50Multi(layers=layers).to(self.device).eval()
        
        # 이미지 로드
        img_bgr = cv2.imread(test_image_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f"이미지를 열 수 없습니다: {test_image_path}")
        
        # 전처리 및 특징 추출
        shorter = int(config.get("shorter", 512))
        t = preprocess_bgr_to_tensor(img_bgr, shorter=shorter).to(self.device)
        
        with torch.inference_mode():
            feat = model(t)  # [1, C, H2, W2]
        
        stride = int(config.get("stride", 2))
        patches = feat_to_patches(feat, stride=stride).cpu().numpy().astype(np.float32, copy=False)
        
        # 이상도 계산 (최근접 거리)
        distances = knn_min_dist_faiss(patches, bank)
        
        # Heatmap 생성
        _, C, H2, W2 = feat.shape
        gy = list(range(0, H2, stride))
        gx = list(range(0, W2, stride))
        Hg, Wg = len(gy), len(gx)
        heat_small = distances.reshape(Hg, Wg)
        
        # 정규화
        max_dist = float(heat_small.max()) if heat_small.size > 0 else 1.0
        heat01_small = heat_small / max_dist if max_dist > 0 else heat_small
        heat01 = upsample_to_img(heat01_small, img_bgr.shape[:2])
        
        # 임계값 적용
        pixel_tau = float(tau.get("pixel", 0.0))
        pixel_tau_scaled = (pixel_tau / max_dist) if max_dist > 0 else 1.0
        pixel_tau_scaled = float(np.clip(pixel_tau_scaled, 0.0, 1.0))
        
        # 마스크 생성
        heat_u8 = (np.clip(heat01, 0, 1) * 255).astype(np.uint8)
        _, mask = cv2.threshold(heat_u8, int(pixel_tau_scaled * 255), 255, cv2.THRESH_BINARY)
        
        # 오버레이 생성
        overlay = img_bgr.copy()
        overlay[mask > 0] = [0, 0, 255]  # 빨간색
        overlay = cv2.addWeighted(img_bgr, 0.7, overlay, 0.3, 0)
        
        # 출력 디렉토리 설정
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="anomaly_")
        else:
            os.makedirs(output_dir, exist_ok=True)
        
        # 저장
        heatmap_path = os.path.join(output_dir, "heatmap.png")
        mask_path = os.path.join(output_dir, "mask.png")
        overlay_path = os.path.join(output_dir, "overlay.png")
        
        cv2.imwrite(heatmap_path, heat_u8)
        cv2.imwrite(mask_path, mask)
        cv2.imwrite(overlay_path, overlay)
        
        # 이미지 레벨 스코어 (99 percentile)
        image_score = float(np.percentile(distances, 99)) if distances.size else 0.0
        
        result = {
            "product_name": product_name,
            "image_score": image_score,
            "pixel_tau": float(pixel_tau),
            "image_tau": float(tau.get("image", 0.0)),
            "heatmap_path": heatmap_path,
            "mask_path": mask_path,
            "overlay_path": overlay_path,
            "is_anomaly": image_score > float(tau.get("image", 0.0))
        }
        
        if self.verbose:
            print(f"[이상 검출 완료] {product_name} - Score: {image_score:.4f}, "
                  f"Anomaly: {result['is_anomaly']}")
        
        return result
    
    
    def detect_with_reference(
        self,
        test_image_path: str,
        reference_image_path: str,
        product_name: Optional[str] = None,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        기준 이미지와 함께 이상 검출 (Side-by-side 비교 이미지 생성)
        
        Args:
            test_image_path: 테스트 이미지 경로
            reference_image_path: 기준 이미지 경로 (TOP-1)
            product_name: 제품명
            output_dir: 출력 디렉토리
            
        Returns:
            detect() 결과 + {"comparison_path": str}
        """
        result = self.detect(test_image_path, product_name, output_dir)
        
         # Side-by-side 이미지 생성 (정상 이미지 + 오버레이만)
        ref_bgr = cv2.imread(reference_image_path, cv2.IMREAD_COLOR)
        overlay = cv2.imread(result["overlay_path"], cv2.IMREAD_COLOR)
        
        if ref_bgr is not None and overlay is not None:
            def resize_h(img, h=512):
                if img is None:
                    return np.zeros((h, h, 3), np.uint8)
                hh, ww = img.shape[:2]
                new_w = int(ww * h / hh)
                return cv2.resize(img, (new_w, h), interpolation=cv2.INTER_AREA)
            
            normal = resize_h(ref_bgr)
            overlay_resized = resize_h(overlay)
            
            # 폭 맞추기
            wmax = max(normal.shape[1], overlay_resized.shape[1])
            def pad_w(img):
                pad = wmax - img.shape[1]
                return cv2.copyMakeBorder(img, 0, 0, 0, max(0, pad), 
                                        cv2.BORDER_CONSTANT, value=(0, 0, 0))
            
            # 1x2 배치: [정상 이미지 | 오버레이]
            grid = np.hstack([pad_w(normal), pad_w(overlay_resized)])
            
            comparison_path = os.path.join(output_dir or result["overlay_path"].rsplit('/', 1)[0], 
                                        "comparison.png")
            cv2.imwrite(comparison_path, grid)
            result["comparison_path"] = comparison_path
        
        return result
    
    # detect_with_reference() 메서드 대신 새로운 메서드 추가
    def detect_with_normal_reference(
        self,
        test_image_path: str,
        product_name: Optional[str] = None,
        normal_gallery_dir: Optional[str] = None,
        similarity_matcher = None,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        정상 이미지 갤러리에서 유사한 정상 이미지를 찾아 기준으로 사용
        
        Args:
            test_image_path: 테스트 이미지 경로
            product_name: 제품명
            normal_gallery_dir: 정상 이미지 디렉토리 (None이면 자동 설정)
            similarity_matcher: CLIP 유사도 매처 인스턴스
            output_dir: 출력 디렉토리
            
        Returns:
            detect() 결과 + {"reference_image_path": str, "comparison_path": str}
        """
        # 제품명 자동 추출
        if product_name is None:
            filename = Path(test_image_path).name
            product_name = self.extract_product_name(filename)
        
        # 정상 이미지 디렉토리 설정
        if normal_gallery_dir is None:
            # 기본 경로: ../data/patchCore/{product_name}/ok
            normal_gallery_dir = self.bank_base_dir / product_name / "ok"
        
        normal_gallery_dir = Path(normal_gallery_dir)
        if not normal_gallery_dir.exists():
            raise FileNotFoundError(
                f"정상 이미지 디렉토리를 찾을 수 없습니다: {normal_gallery_dir}"
            )
        
        # CLIP으로 가장 유사한 정상 이미지 찾기
        if similarity_matcher is None:
            raise ValueError("similarity_matcher가 필요합니다")
        
        # 정상 이미지 갤러리로 임시 인덱스 구축
        temp_index_built = False
        if not similarity_matcher.index_built or \
        str(normal_gallery_dir) not in str(similarity_matcher.gallery_paths[0] if similarity_matcher.gallery_paths else ""):
            similarity_matcher.build_index(str(normal_gallery_dir))
            temp_index_built = True
        
        # 유사 정상 이미지 검색 (TOP-1)
        search_result = similarity_matcher.search(str(test_image_path), top_k=1)
        reference_image_path = search_result.top_k_results[0]["image_path"]
        
        if self.verbose:
            print(f"[정상 기준 이미지 선택] {reference_image_path}")
            print(f"  유사도: {search_result.top_k_results[0]['similarity_score']:.4f}")
        
        # 기존 detect_with_reference 호출
        result = self.detect_with_reference(
            test_image_path=test_image_path,
            reference_image_path=reference_image_path,
            product_name=product_name,
            output_dir=output_dir
        )
        
        result["reference_image_path"] = reference_image_path
        result["reference_similarity"] = search_result.top_k_results[0]['similarity_score']
        
        return result

# 헬퍼 함수
def create_detector(
    bank_base_dir: str = "../data/patchCore",
    device: str = "auto",
    verbose: bool = True
) -> AnomalyDetector:
    """
    Detector 인스턴스 생성 헬퍼
    """
    return AnomalyDetector(
        bank_base_dir=bank_base_dir,
        device=device,
        verbose=verbose
    )


# 사용 예제
if __name__ == "__main__":
    detector = create_detector(verbose=True)
    
    # 테스트
    result = detector.detect(
        test_image_path="../data/def_split/prod1_burr_10.jpeg",
        output_dir="./test_output"
    )
    
    print(f"\n=== 결과 ===")
    print(f"제품명: {result['product_name']}")
    print(f"이상 점수: {result['image_score']:.4f}")
    print(f"이상 여부: {result['is_anomaly']}")
    print(f"출력 파일:")
    print(f"  - Heatmap: {result['heatmap_path']}")
    print(f"  - Mask: {result['mask_path']}")
    print(f"  - Overlay: {result['overlay_path']}")

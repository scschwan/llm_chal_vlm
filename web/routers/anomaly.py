"""
이상 검출 관련 API 라우터
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Optional

router = APIRouter(prefix="/anomaly", tags=["anomaly"])

# 전역 변수
_detector_ref = None
_matcher_ref = None
_anomaly_output_dir_ref = None
_project_root_ref = None
_index_dir_ref = None


def init_anomaly_router(detector, matcher, anomaly_output_dir, proj_root, index_dir):
    """라우터 초기화"""
    global _detector_ref, _matcher_ref, _anomaly_output_dir_ref, _project_root_ref, _index_dir_ref
    _detector_ref = detector
    _matcher_ref = matcher
    _anomaly_output_dir_ref = anomaly_output_dir
    _project_root_ref = proj_root
    _index_dir_ref = index_dir
    print(f"[ANOMALY ROUTER] 초기화 완료: detector={_detector_ref is not None}")


def get_detector():
    return _detector_ref


def get_matcher():
    return _matcher_ref


def get_anomaly_output_dir():
    return _anomaly_output_dir_ref


def get_project_root():
    return _project_root_ref


def get_index_dir():
    return _index_dir_ref


class AnomalyDetectRequest(BaseModel):
    """이상 검출 요청"""
    test_image_path: str = Field(..., description="테스트 이미지 경로")
    product_name: str = Field(..., description="제품명")
    # ✅ TOP-1 불량 이미지는 표시용으로만 사용
    top1_defect_image: Optional[str] = Field(None, description="TOP-1 불량 이미지 (표시용)")


async def switch_to_normal_index():
    """정상 이미지 인덱스로 전환"""
    matcher = get_matcher()
    INDEX_DIR = get_index_dir()
    project_root = get_project_root()
    
    if matcher is None:
        raise HTTPException(500, "매처가 초기화되지 않았습니다")
    
    normal_index_path = INDEX_DIR / "normal"
    
    try:
        # 저장된 정상 인덱스 로드
        if (normal_index_path / "index_data.pt").exists():
            print(f"[ANOMALY] 정상 이미지 인덱스 로드: {normal_index_path}")
            matcher.load_index(str(normal_index_path))
            print(f"[ANOMALY] 정상 인덱스 로드 완료: {len(matcher.gallery_paths)}개 이미지")
            return {
                "status": "success",
                "index_type": "normal",
                "gallery_count": len(matcher.gallery_paths) if matcher.gallery_paths else 0
            }
        else:
            # 인덱스가 없으면 새로 구축
            normal_base_dir = project_root / "data" / "patchCore"
            
            if not normal_base_dir.exists():
                raise FileNotFoundError(f"정상 이미지 디렉토리가 없습니다: {normal_base_dir}")
            
            print(f"[ANOMALY] 정상 이미지 인덱스 구축 중: {normal_base_dir}")
            info = matcher.build_index(str(normal_base_dir))
            
            # 인덱스 저장
            normal_index_path.mkdir(parents=True, exist_ok=True)
            matcher.save_index(str(normal_index_path))
            
            print(f"[ANOMALY] 정상 인덱스 구축 완료: {info['num_images']}개 이미지")
            return {
                "status": "success",
                "index_type": "normal",
                "gallery_count": info["num_images"]
            }
    
    except Exception as e:
        print(f"[ANOMALY] 정상 인덱스 로드 실패: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"정상 인덱스 로드 실패: {str(e)}")


@router.post("/detect")
async def detect_anomaly(request: AnomalyDetectRequest):
    """
    이상 검출 수행
    
    Args:
        request: 이상 검출 요청
    
    Returns:
        이상 검출 결과 (점수, 마스크, 오버레이 이미지 등)
    """
    detector = get_detector()
    matcher = get_matcher()
    anomaly_output_dir = get_anomaly_output_dir()
    project_root = get_project_root()
    
    if detector is None:
        raise HTTPException(500, "이상 검출기가 초기화되지 않았습니다")
    
    if matcher is None:
        raise HTTPException(500, "유사도 매처가 초기화되지 않았습니다")
    
    # 테스트 이미지 경로 정규화
    test_path = Path(request.test_image_path)
    if not test_path.is_absolute():
        if str(test_path).startswith("uploads/"):
            test_path = project_root / "web" / test_path
        else:
            test_path = project_root / test_path
    
    if not test_path.exists():
        raise HTTPException(404, f"테스트 이미지를 찾을 수 없습니다: {test_path}")
    
    try:
        print(f"[ANOMALY] 이상 검출 시작: {test_path.name}")
        print(f"[ANOMALY] 제품명: {request.product_name}")
        
        # ✅ 정상 이미지 인덱스로 전환
        await switch_to_normal_index()
        
        # 출력 디렉토리 생성
        output_dir = anomaly_output_dir / test_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # ✅ 정상 이미지 기준으로 이상 검출 수행
        result = detector.detect_with_normal_reference(
            test_image_path=str(test_path),
            product_name=request.product_name,
            similarity_matcher=matcher,
            output_dir=str(output_dir)
        )
        
        print(f"[ANOMALY] 이상 검출 완료: score={result['image_score']:.4f}")
        print(f"[ANOMALY] 정상 기준 이미지: {result.get('reference_image_path', 'N/A')}")
        
        # 결과 반환
        return JSONResponse(content={
            "status": "success",
            "product_name": result["product_name"],
            "image_score": float(result["image_score"]),
            "pixel_tau": float(result["pixel_tau"]),
            "image_tau": float(result["image_tau"]),
            "is_anomaly": bool(result["is_anomaly"]),
            # ✅ 정상 기준 이미지 경로
            "reference_normal_path": result.get("reference_image_path", ""),
            # ✅ TOP-1 불량 이미지 경로 (표시용)
            "top1_defect_path": request.top1_defect_image,
            "mask_url": f"/anomaly/image/{test_path.stem}/mask.png",
            "overlay_url": f"/anomaly/image/{test_path.stem}/overlay.png",
            "comparison_url": f"/anomaly/image/{test_path.stem}/comparison.png"
        })
    
    except Exception as e:
        print(f"[ANOMALY] 이상 검출 실패: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"이상 검출 실패: {str(e)}")


@router.get("/image/{result_id}/{filename}")
async def serve_anomaly_image(result_id: str, filename: str):
    """이상 검출 결과 이미지 제공"""
    anomaly_output_dir = get_anomaly_output_dir()
    
    file_path = anomaly_output_dir / result_id / filename
    
    if not file_path.exists():
        raise HTTPException(404, f"이미지를 찾을 수 없습니다: {filename}")
    
    return FileResponse(file_path, media_type="image/png")
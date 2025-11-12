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


def init_anomaly_router(detector, matcher, anomaly_output_dir, proj_root):
    """라우터 초기화"""
    global _detector_ref, _matcher_ref, _anomaly_output_dir_ref, _project_root_ref
    _detector_ref = detector
    _matcher_ref = matcher
    _anomaly_output_dir_ref = anomaly_output_dir
    _project_root_ref = proj_root
    print(f"[ANOMALY ROUTER] 초기화 완료: detector={_detector_ref is not None}")


def get_detector():
    return _detector_ref


def get_matcher():
    return _matcher_ref


def get_anomaly_output_dir():
    return _anomaly_output_dir_ref


def get_project_root():
    return _project_root_ref


class AnomalyDetectRequest(BaseModel):
    """이상 검출 요청"""
    test_image_path: str = Field(..., description="테스트 이미지 경로")
    reference_image_path: Optional[str] = Field(None, description="기준 이미지 경로 (TOP-1)")
    product_name: str = Field(..., description="제품명")


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
        
        # 출력 디렉토리 생성
        output_dir = anomaly_output_dir / test_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 참조 이미지가 있으면 사용, 없으면 자동 선택
        if request.reference_image_path:
            # 사용자가 선택한 TOP-1 이미지 사용
            ref_path = Path(request.reference_image_path)
            if not ref_path.is_absolute():
                ref_path = project_root / ref_path
            
            if not ref_path.exists():
                raise HTTPException(404, f"참조 이미지를 찾을 수 없습니다: {ref_path}")
            
            result = detector.detect_with_reference(
                test_image_path=str(test_path),
                reference_image_path=str(ref_path),
                product_name=request.product_name,
                output_dir=str(output_dir)
            )
        else:
            # 자동으로 정상 이미지 선택
            if matcher is None:
                raise HTTPException(500, "유사도 매처가 초기화되지 않았습니다")
            
            result = detector.detect_with_normal_reference(
                test_image_path=str(test_path),
                product_name=request.product_name,
                similarity_matcher=matcher,
                output_dir=str(output_dir)
            )
        
        print(f"[ANOMALY] 이상 검출 완료: score={result['image_score']:.4f}")
        
        # 결과 반환
        return JSONResponse(content={
            "status": "success",
            "product_name": result["product_name"],
            "image_score": float(result["image_score"]),
            "pixel_tau": float(result["pixel_tau"]),
            "image_tau": float(result["image_tau"]),
            "is_anomaly": bool(result["is_anomaly"]),
            "reference_image_path": result.get("reference_image_path", ""),
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
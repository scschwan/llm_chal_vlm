"""
이상 검출 관련 API 라우터
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Optional
from web.database.models import ResponseHistory
import json
from sqlalchemy.orm import Session
from web.database.connection import get_db
from datetime import datetime

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
    defect_name: str = Field(None, description="불량명")
    search_id: Optional[int] = Field(None, description="검색 이력 ID")
    similarity_score: Optional[float] = Field(None, description="유사도 점수")

# ===== 개선된 anomaly.py =====

# 전역 변수에 현재 인덱스 타입 추가
_current_index_type = None  # "defect" or "normal"

async def switch_to_normal_index():
    """정상 이미지 인덱스로 전환 (V2 DB 기반)"""
    global _current_index_type
    
    matcher = get_matcher()
    INDEX_DIR = get_index_dir()  # V2라면 INDEX_DIR_V2가 들어옴
    
    if matcher is None:
        raise HTTPException(500, "매처가 초기화되지 않았습니다")
    
    # V2인지 확인
    is_v2 = hasattr(matcher, 'gallery_metadata')
    
    # ✅ 이미 정상 인덱스가 로드되어 있으면 스킵
    if _current_index_type == "normal" and matcher.index_built:
        print(f"[ANOMALY] 이미 정상 인덱스가 로드되어 있음 (스킵)")
        
        if is_v2:
            gallery_count = len(matcher.gallery_metadata)
        else:
            gallery_count = len(matcher.gallery_paths) if matcher.gallery_paths else 0
        
        return {
            "status": "success",
            "index_type": "normal",
            "gallery_count": gallery_count,
            "cached": True
        }
    
    normal_index_path = INDEX_DIR / "normal"
    
    try:
        if is_v2:
            # ========== V2: DB 기반 인덱스 ==========
            # 저장된 정상 인덱스 로드
            if (normal_index_path / "metadata.json").exists():
                print(f"[ANOMALY V2] 정상 이미지 인덱스 로드: {normal_index_path}")
                matcher.load_index(str(normal_index_path))
                _current_index_type = "normal"
                print(f"[ANOMALY V2] 정상 인덱스 로드 완료: {len(matcher.gallery_metadata)}개 이미지")
                return {
                    "status": "success",
                    "index_type": "normal",
                    "gallery_count": len(matcher.gallery_metadata)
                }
            else:
                # ✅ 인덱스가 없으면 DB에서 구축
                
                
                print(f"[ANOMALY V2] 정상 이미지 인덱스 구축 중 (DB 기반)...")
                
                # DB 세션 생성
                db = next(get_db())
                
                try:
                    # DB에서 normal 타입 이미지로 인덱스 구축
                    info = matcher.build_index_from_db(db, image_type="normal")
                    
                    # 인덱스 저장
                    normal_index_path.mkdir(parents=True, exist_ok=True)
                    matcher.save_index(str(normal_index_path))
                    _current_index_type = "normal"
                    
                    print(f"[ANOMALY V2] 정상 인덱스 구축 완료: {info['num_images']}개 이미지")
                    return {
                        "status": "success",
                        "index_type": "normal",
                        "gallery_count": info["num_images"]
                    }
                finally:
                    db.close()
        
        else:
            # ========== V1: 파일 기반 인덱스 ==========
            # 저장된 정상 인덱스 로드
            if (normal_index_path / "index_data.pt").exists():
                print(f"[ANOMALY V1] 정상 이미지 인덱스 로드: {normal_index_path}")
                matcher.load_index(str(normal_index_path))
                _current_index_type = "normal"
                print(f"[ANOMALY V1] 정상 인덱스 로드 완료: {len(matcher.gallery_paths)}개 이미지")
                return {
                    "status": "success",
                    "index_type": "normal",
                    "gallery_count": len(matcher.gallery_paths) if matcher.gallery_paths else 0
                }
            else:
                # 인덱스가 없으면 새로 구축
                project_root = get_project_root()
                normal_base_dir = project_root / "data" / "patchCore"
                
                if not normal_base_dir.exists():
                    raise FileNotFoundError(f"정상 이미지 디렉토리가 없습니다: {normal_base_dir}")
                
                print(f"[ANOMALY V1] 정상 이미지 인덱스 구축 중: {normal_base_dir}")
                info = matcher.build_index(str(normal_base_dir))
                
                # 인덱스 저장
                normal_index_path.mkdir(parents=True, exist_ok=True)
                matcher.save_index(str(normal_index_path))
                _current_index_type = "normal"
                
                print(f"[ANOMALY V1] 정상 인덱스 구축 완료: {info['num_images']}개 이미지")
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
        #await switch_to_normal_index()
        
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

        # TOP-1 유사 이미지 찾기 (정상 이미지에서)
        top1_match = None
        if matcher and matcher.index_built:
            try:
                print(f"[ANOMALY] 정상 이미지에서 유사 이미지 검색 중...")
                search_result = matcher.search(str(test_path), top_k=1)
                
                if search_result.results and len(search_result.results) > 0:
                    top1_match = search_result.results[0]
                    print(f"[ANOMALY] TOP-1 정상 이미지: {top1_match['file_name']}")
            except Exception as e:
                print(f"[ANOMALY] 유사 이미지 검색 실패: {e}")
        
        print(f"[ANOMALY] 이상 검출 완료: score={result['image_score']:.4f}")
        print(f"[ANOMALY] 정상 기준 이미지: {result.get('reference_image_path', 'N/A')}")

        # ========== DB 저장 ==========
        db: Session = next(get_db())
        response_id = None
        
        try:
            # response_history 테이블에 저장
            response_history = ResponseHistory(
                executed_at=datetime.now(),
                search_id=request.search_id,  # 이전 페이지에서 전달받은 값
                product_code=request.product_name,  # 한글 제품명
                defect_code=request.defect_name,    # 한글 불량명
                similarity_score=request.similarity_score,  # 이전 페이지에서 전달받은 값
                anomaly_score=result['image_score'],
                test_image_path=str(test_path)
            )
            
            db.add(response_history)
            db.commit()
            db.refresh(response_history)
            
            response_id = response_history.response_id
            print(f"[ANOMALY] DB 저장 완료: response_id={response_id}")
        
        except Exception as e:
            print(f"[ANOMALY] DB 저장 실패: {e}")
            db.rollback()
        finally:
            db.close()
        
        # 결과 반환
        '''
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
            "comparison_url": f"/anomaly/image/{test_path.stem}/comparison.png",
            "response_id": response_id  # 추가
        })
        '''
        response_data  = {
            "status": "success",
            "product": request.product_name,
            "test_image": str(test_path),
            "anomaly_score": result['image_score'],
            "is_anomaly": result['is_anomaly'],
            "threshold": result.get('threshold', 0.5),
            "overlay_url": result.get('overlay_url', ''),
            "heatmap_url": result.get('heatmap_url', ''),
            "top1_defect_path": request.top1_defect_image,
            "response_id": response_id  # 추가
        }
        # TOP-1 정상 이미지 정보 추가
        if top1_match:
            response_data.update({
                "top1_normal_image": top1_match.get('local_path', ''),
                "top1_similarity": top1_match.get('similarity_score', 0),
                "top1_product": top1_match.get('product_name', ''),
            })

        return JSONResponse(content=response_data)
    
    except Exception as e:
        print(f"[ANOMALY] 이상 검출 실패: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"이상 검출 실패: {str(e)}")

# 기존 import에 추가
from web.utils.session_helper import get_origin_image_path, upload_session_file
from web.utils.object_storage import get_obs_manager

class AnomalyDetectSessionRequest(BaseModel):
    """이상 검출 요청 (세션 기반)"""
    session_id: str = Field(..., description="세션 ID")
    product_name: str = Field(..., description="제품명")
    top1_defect_image: Optional[str] = Field(None, description="TOP-1 불량 이미지 (표시용)")



@router.post("/detect-session")
async def detect_anomaly_session(request: AnomalyDetectSessionRequest):
    """
    이상 검출 수행 (세션 기반)
    
    Args:
        request: 세션 기반 이상 검출 요청
    
    Returns:
        이상 검출 결과 (OBS URL 포함)
    """
    detector = get_detector()
    matcher = get_matcher()
  
    if detector is None:
        raise HTTPException(500, "이상 검출기가 초기화되지 않았습니다")
    
    if matcher is None:
        raise HTTPException(500, "유사도 매처가 초기화되지 않았습니다")
    
    try:
        # 1. 세션 폴더에서 origin 이미지 찾기
        session_dir = Path("/home/dmillion/llm_chal_vlm/uploads") / request.session_id
        
        if not session_dir.exists():
            raise HTTPException(404, f"세션을 찾을 수 없습니다: {request.session_id}")
        
        origin_path = get_origin_image_path(session_dir)
        
        print(f"[ANOMALY-SESSION] 세션 ID: {request.session_id}")
        print(f"[ANOMALY-SESSION] 세션 폴더: {session_dir}")
        print(f"[ANOMALY-SESSION] 원본 이미지: {origin_path}")
        print(f"[ANOMALY-SESSION] 제품명: {request.product_name}")
        
        # 2. 정상 이미지 인덱스로 전환
        await switch_to_normal_index()
        
        # 3. 출력 디렉토리 생성 (세션 폴더 내)
        output_dir = session_dir / "anomaly_results"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 4. 정상 이미지 기준으로 이상 검출 수행
        result = detector.detect_with_normal_reference(
            test_image_path=str(origin_path),
            product_name=request.product_name,
            similarity_matcher=matcher,
            output_dir=str(output_dir)
        )
        
        print(f"[ANOMALY-SESSION] 이상 검출 완료: score={result['image_score']:.4f}")
        print(f"[ANOMALY-SESSION] 정상 기준 이미지: {result.get('reference_image_path', 'N/A')}")
        
        # 5. 생성된 이미지들 OBS에 업로드
        obs = get_obs_manager()
        base_obs_url = obs.get_url(f"uploads/{request.session_id}/origin.png")
        
        mask_path = output_dir / "mask.png"
        overlay_path = output_dir / "overlay.png"
        comparison_path = output_dir / "comparison.png"
        
        mask_url = upload_session_file(mask_path, base_obs_url, "mask.png")
        overlay_url = upload_session_file(overlay_path, base_obs_url, "overlay.png")
        comparison_url = upload_session_file(comparison_path, base_obs_url, "comparison.png")
        
        print(f"[ANOMALY-SESSION] OBS 업로드 완료")
        
        # 6. 결과 반환
        return JSONResponse(content={
            "status": "success",
            "session_id": request.session_id,
            "product_name": result["product_name"],
            "image_score": float(result["image_score"]),
            "pixel_tau": float(result["pixel_tau"]),
            "image_tau": float(result["image_tau"]),
            "is_anomaly": bool(result["is_anomaly"]),
            # OBS URL 반환
            "reference_normal_url": result.get("reference_image_url", ""),
            "top1_defect_url": request.top1_defect_image,
            "mask_url": mask_url,
            "overlay_url": overlay_url,
            "comparison_url": comparison_url
        })
    
    except Exception as e:
        print(f"[ANOMALY-SESSION] 이상 검출 실패: {e}")
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
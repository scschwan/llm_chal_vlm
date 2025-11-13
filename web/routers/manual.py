"""
대응 매뉴얼 생성 관련 API 라우터
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Optional
import httpx
import time

router = APIRouter(prefix="/manual", tags=["manual"])

# 전역 변수
_mapper_ref = None
_rag_ref = None
_project_root_ref = None
_llm_server_url = "http://localhost:5001"


def init_manual_router(mapper, rag, proj_root, llm_url=None):
    """라우터 초기화"""
    global _mapper_ref, _rag_ref, _project_root_ref, _llm_server_url
    _mapper_ref = mapper
    _rag_ref = rag
    _project_root_ref = proj_root
    if llm_url:
        _llm_server_url = llm_url
    print(f"[MANUAL ROUTER] 초기화 완료: mapper={_mapper_ref is not None}, rag={_rag_ref is not None}")


def get_mapper():
    return _mapper_ref


def get_rag():
    return _rag_ref


def get_project_root():
    return _project_root_ref


class ManualGenerateRequest(BaseModel):
    """매뉴얼 생성 요청"""
    product: str = Field(..., description="제품명")
    defect: str = Field(..., description="불량 유형")
    anomaly_score: float = Field(..., description="이상 점수")
    is_anomaly: bool = Field(..., description="이상 판정 여부")
    model_type: str = Field("hyperclovax", description="모델 타입 (hyperclovax, exaone, llava)")
    image_path: Optional[str] = Field(None, description="이미지 경로 (VLM용)")


@router.post("/generate")
async def generate_manual(request: ManualGenerateRequest):
    """
    대응 매뉴얼 생성
    
    Args:
        request: 매뉴얼 생성 요청
    
    Returns:
        생성된 매뉴얼 (4개 섹션)
    """
    mapper = get_mapper()
    rag = get_rag()
    project_root = get_project_root()
    
    if mapper is None:
        raise HTTPException(500, "DefectMapper가 초기화되지 않았습니다")
    
    t0 = time.time()
    
    try:
        print(f"[MANUAL] 매뉴얼 생성 시작: {request.product}/{request.defect}")
        print(f"[MANUAL] 모델: {request.model_type}")
        
        # 1. 불량 정보 조회
        defect_info = mapper.get_defect_info(request.product, request.defect)
        
        if not defect_info:
            raise HTTPException(404, f"불량 정보를 찾을 수 없습니다: {request.product}/{request.defect}")
        
        # 2. RAG 매뉴얼 검색 (개선된 rag.py 사용)
        manual_context = {"원인": [], "조치": []}
        
        if rag:
            print(f"[MANUAL] RAG 검색 시작: 제품={request.product}, 불량={request.defect}")
            
            # RAG에서 직접 구조화된 결과 반환 (추가 필터링 불필요)
            manual_context = rag.search_defect_manual(
                product=request.product,
                defect=request.defect,
                #keywords=[defect_info.ko, defect_info.full_name_ko],
                keywords=[defect_info.ko],  # ko만 사용 (자동 확장됨)
                k=3
            )
            
            print(f"[MANUAL] RAG 검색 완료: 원인 {len(manual_context['원인'])}개, 조치 {len(manual_context['조치'])}개")
            
            # 디버깅: 검색된 내용 출력
            if manual_context["원인"]:
                print(f"[DEBUG] 원인 샘플: {manual_context['원인'][0][:100]}...")
            if manual_context["조치"]:
                print(f"[DEBUG] 조치 샘플: {manual_context['조치'][0][:100]}...")
        else:
            print("[MANUAL] RAG 비활성화")
        
        # 3. LLM/VLM 호출
        llm_analysis = None
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            if request.model_type == "hyperclovax":
                # HyperCLOVAX
                payload = {
                    "product": request.product,
                    "defect_en": defect_info.en,
                    "defect_ko": defect_info.ko,
                    "full_name_ko": defect_info.full_name_ko,
                    "anomaly_score": float(request.anomaly_score),
                    "is_anomaly": bool(request.is_anomaly),
                    "manual_context": manual_context,
                    #"max_new_tokens": 1024,
                    "max_new_tokens": 768,
                    "temperature": 0.3
                }
                
                response = await client.post(f"{_llm_server_url}/analyze", json=payload)
                
            elif request.model_type == "exaone":
                # EXAONE 3.5
                payload = {
                    "product": request.product,
                    "defect_en": defect_info.en,
                    "defect_ko": defect_info.ko,
                    "full_name_ko": defect_info.full_name_ko,
                    "anomaly_score": float(request.anomaly_score),
                    "is_anomaly": bool(request.is_anomaly),
                    "manual_context": manual_context,
                    "max_new_tokens": 1024,
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "repetition_penalty": 1.1
                }
                
                response = await client.post(f"{_llm_server_url}/analyze_exaone", json=payload)
                
            elif request.model_type == "llava":
                # LLaVA
                if not request.image_path:
                    raise HTTPException(400, "VLM 모델 사용 시 이미지 경로가 필요합니다")
                
                # 이미지 경로 정규화
                image_path = Path(request.image_path)
                if not image_path.is_absolute():
                    if str(image_path).startswith("uploads/"):
                        image_path = project_root / "web" / image_path
                    else:
                        image_path = project_root / image_path
                
                payload = {
                    "image_path": str(image_path),
                    "product": request.product,
                    "defect_en": defect_info.en,
                    "defect_ko": defect_info.ko,
                    "full_name_ko": defect_info.full_name_ko,
                    "anomaly_score": float(request.anomaly_score),
                    "is_anomaly": bool(request.is_anomaly),
                    "manual_context": manual_context,
                    "max_new_tokens": 1024,
                    "temperature": 0.3
                }
                
                response = await client.post(f"{_llm_server_url}/analyze_vlm", json=payload)
                
            else:
                raise HTTPException(400, f"지원하지 않는 모델 타입: {request.model_type}")
            
            if response.status_code == 200:
                result = response.json()
                llm_analysis = result.get("analysis", "")
                print(f"[MANUAL] LLM 분석 완료: {len(llm_analysis)} 문자")
            else:
                raise Exception(f"LLM 서버 오류: {response.status_code} - {response.text}")
        
        # 4. 결과 반환
        processing_time = round(time.time() - t0, 2)
        
        return JSONResponse(content={
            "status": "success",
            "product": request.product,
            "defect_en": defect_info.en,
            "defect_ko": defect_info.ko,
            "full_name_ko": defect_info.full_name_ko,
            "manual_context": manual_context,
            "llm_analysis": llm_analysis,
            "anomaly_score": float(request.anomaly_score),
            "is_anomaly": bool(request.is_anomaly),
            "model_type": request.model_type,
            "processing_time": processing_time
        })
        
    except httpx.ConnectError:
        raise HTTPException(503, "LLM 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.")
    except Exception as e:
        print(f"[MANUAL] 매뉴얼 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"매뉴얼 생성 실패: {str(e)}")
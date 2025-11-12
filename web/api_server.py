"""
TOP-K 유사도 매칭 + Anomaly Detection API 서버
FastAPI 기반으로 외부 웹서버에서 호출 가능한 REST API 제공
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict,List, Optional
import os
import sys
import shutil
from pathlib import Path
import uvicorn
from fastapi.staticfiles import StaticFiles
import httpx


import time  # 기존 import에 추가

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
uploads_dir = project_root / "web" / "uploads"
uploads_dir.mkdir(parents=True, exist_ok=True)  # 폴더가 없으면 생성



# modules 폴더의 모듈 import
from modules.similarity_matcher import TopKSimilarityMatcher, create_matcher
from modules.anomaly_detector import AnomalyDetector, create_detector
# VLM 모듈 import
from modules.vlm import RAGManager, VLMInference, PromptBuilder, DefectMapper
from modules.vlm.llm_inference import LLMInference





# ====================
# VLM 관련 실행 함수,컴포넌트
# ====================

# 전역 변수로 VLM 컴포넌트 초기화
vlm_components = {
    "rag": None,
    "vlm": None,
    "mapper": None,
    "prompt_builder": PromptBuilder()
}

current_index_type: Optional[str] = None  # 'defect' 또는 'normal'
normal_dir = project_root / "data" / "patchCore" / "prod1"


def init_vlm_components():
    """VLM 컴포넌트 초기화 (서버 시작 시 1회)"""
    global vlm_components
    
    try:
        print("\n" + "="*50)
        print("VLM 컴포넌트 초기화 중...")
        print("="*50)
        
        # 경로 설정
        #pdf_path = project_root / "prod1_menual.pdf"
        vector_store_path = project_root / "manual_store"
        #pdf_path = vector_store_path / "prod1_menual.pdf"

            # PDF 경로를 두 곳에서 확인
        pdf_candidates = [
            vector_store_path / "prod1_menual.pdf",  # 벡터 스토어 내부
            project_root / "prod1_menual.pdf"         # 프로젝트 루트
        ]

        pdf_path = None
        for candidate in pdf_candidates:
            if candidate.exists():
                pdf_path = candidate
                print(f"✅ PDF 파일 발견: {pdf_path}")
                break
        
        if not pdf_path:
            print("⚠️  prod1_menual.pdf를 찾을 수 없습니다")
            print(f"   확인 경로: {[str(p) for p in pdf_candidates]}")


        mapping_file = project_root / "web" / "defect_mapping.json"
        
        # 매핑 파일이 없으면 생성
        if not mapping_file.exists():
            print("⚠️  매핑 파일이 없습니다. 기본 파일을 생성합니다...")
            from modules.vlm.defect_mapper import create_default_mapping
            create_default_mapping(mapping_file)
        
        # DefectMapper 초기화
        print("\n1. DefectMapper 초기화...")
        vlm_components["mapper"] = DefectMapper(mapping_file)
        
        # RAGManager 초기화
        print("\n2. RAGManager 초기화...")
        if not pdf_path.exists():
            print(f"⚠️  PDF 파일을 찾을 수 없습니다: {pdf_path}")
            print("   VLM 기능이 제한됩니다.")
        else:
            vlm_components["rag"] = RAGManager(
                pdf_path=pdf_path,
                vector_store_path=vector_store_path,
                device="cuda",
                verbose=True
            )
        
        # VLMInference 초기화 (선택적 - 메모리 고려)
        print("\n3. VLMInference 초기화 (스킵 - 필요 시 동적 로드)...")
        # vlm_components["vlm"] = VLMInference(
        #     model_name="llava-hf/llava-v1.6-mistral-7b-hf",
        #     use_4bit=True,
        #     verbose=True
        # )
        print("   → VLM 모델은 첫 요청 시 동적으로 로드됩니다.")
        
        print("\n" + "="*50)
        print("✅ VLM 컴포넌트 초기화 완료")
        print("="*50 + "\n")
        
    except Exception as e:
        print(f"\n❌ VLM 초기화 오류: {e}")
        import traceback
        traceback.print_exc()

# ====================
# Pydantic 모델
# ====================

class BuildIndexRequest(BaseModel):
    """인덱스 구축 요청"""
    gallery_dir: str = Field(..., description="갤러리 이미지 디렉토리 경로")
    save_index: bool = Field(False, description="인덱스를 파일로 저장할지 여부")
    index_save_dir: Optional[str] = Field(None, description="인덱스 저장 경로")


class SearchRequest(BaseModel):
    """검색 요청 (이미지 경로 기반)"""
    query_image_path: str = Field(..., description="쿼리 이미지 경로")
    top_k: int = Field(5, ge=1, le=50, description="상위 K개 결과")


class SearchResponse(BaseModel):
    """검색 응답"""
    status: str
    query_image: str
    top_k_results: List[dict]
    total_gallery_size: int
    model_info: str


class AnomalyDetectRequest(BaseModel):
    """이상 검출 요청"""
    test_image_path: str = Field(..., description="테스트 이미지 경로")
    reference_image_path: Optional[str] = Field(None, description="기준 이미지 경로 (TOP-1)")
    product_name: Optional[str] = Field(None, description="제품명 (자동 추출 가능)")


class AnomalyDetectResponse(BaseModel):
    """이상 검출 응답"""
    status: str
    product_name: str
    image_score: float
    pixel_tau: float
    image_tau: float
    is_anomaly: bool
    reference_normal_url: str  # 추가: 정상 기준 이미지
    mask_url: str
    overlay_url: str
    comparison_url: Optional[str] = None


class HealthResponse(BaseModel):
    """헬스체크 응답"""
    status: str
    message: str
    index_built: bool
    gallery_size: int

# ====== 매뉴얼 생성 공용 모델 ======
class ManualGenRequest(BaseModel):
    image_path: str
    top1_image_path: Optional[str] = None
    product_name: Optional[str] = None
    defect_name: Optional[str] = None
    anomaly_score: Optional[float] = None
    is_anomaly: Optional[bool] = None
    #max_new_tokens: int = 512
    max_new_tokens: int = 1024
    #temperature: float = 0.7
    temperature: float = 0.3
    verbose: bool = False  # ✅ 추가: 디버그 로그 출력


async def switch_index(index_type: str):
    """
    인덱스 타입 전환
    
    Args:
        index_type: 'defect' (불량 이미지용) 또는 'normal' (정상 이미지용)
    
    Returns:
        dict: 전환 결과
    """
    global current_index_type , normal_dir
    
    # 이미 로드된 인덱스면 스킵
    if current_index_type == index_type:
        return {
            "status": "already_loaded",
            "index_type": index_type,
            "gallery_count": len(matcher.gallery_paths) if matcher and matcher.gallery_paths else 0
        }
    
    # 인덱스 타입 검증
    if index_type not in ["defect", "normal"]:
        raise ValueError(f"잘못된 index_type: {index_type} (defect 또는 normal만 가능)")
    
    # 갤러리 디렉토리 설정
    if index_type == "defect":
        gallery_dir = project_root / "data" / "def_split"
    else:  # normal
        #gallery_dir = project_root / "data" / "ok_split"
        gallery_dir = normal_dir
    
    # 디렉토리 존재 확인
    if not gallery_dir.exists():
        raise FileNotFoundError(f"갤러리 디렉토리가 없습니다: {gallery_dir}")
    
    print(f"\n{'='*60}")
    print(f"인덱스 전환: {current_index_type or 'None'} → {index_type}")
    print(f"갤러리: {gallery_dir}")
    print(f"{'='*60}")
    
    try:
        # 인덱스 구축
        info = matcher.build_index(str(gallery_dir))
        
        # 인덱스 저장
        index_path = INDEX_DIR / index_type
        index_path.mkdir(parents=True, exist_ok=True)
        matcher.save_index(str(index_path))
        
        # 현재 인덱스 타입 업데이트
        current_index_type = index_type
        
        print(f"✅ 인덱스 전환 완료: {info['num_images']}개 이미지")
        
        return {
            "status": "success",
            "index_type": index_type,
            "gallery_dir": str(gallery_dir),
            "gallery_count": info["num_images"],
            "message": f"{index_type} 인덱스로 전환 완료"
        }
    
    except Exception as e:
        print(f"❌ 인덱스 전환 실패: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"인덱스 전환 실패: {str(e)}")

# ====== 공용 코어 ======
# ====== 매뉴얼 생성 공용 코어 ======
async def _manual_core(mode: str, req: ManualGenRequest):
    """
    mode: 'llm' | 'vlm'
    1) 제품/불량 추출
    2) PatchCore 이상 검출 (추가!)
    3) mapper/RAG로 메뉴얼 추출
    4) LLM/VLM 호출
    """
    t0 = time.time()

    # 0) 제품/불량 보정 (TOP-1 파일명 규칙: {product}_{defect}_...)
    product = req.product_name
    defect  = req.defect_name
    
    if not product or not defect:
        name = (req.top1_image_path or '').split('/')[-1]
        parts = name.split('_')
        if not product and len(parts) >= 1:
            product = parts[0]
        if not defect and len(parts) >= 2:
            defect = parts[1]

    if not product or not defect:
        raise HTTPException(400, "product/defect 파악 실패: product_name, defect_name를 제공하거나 TOP-1 파일명 규칙을 확인하세요.")

    # ========================================
    # ✅ 1) PatchCore 이상 검출 (추가!)
    # ========================================
    anomaly_score = req.anomaly_score or 0.0
    is_anomaly = req.is_anomaly if req.is_anomaly is not None else False
    
    # req에 anomaly_score가 없으면 직접 검출
    if req.anomaly_score is None and detector is not None:
        try:
            if req.verbose:
                print(f"\n🔍 PatchCore 이상 검출 시작: {req.image_path}")
            
            # 출력 디렉토리
            image_stem = Path(req.image_path).stem
            output_dir = ANOMALY_OUTPUT_DIR / image_stem
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # ✅ PatchCore 실행
            if req.top1_image_path:
                # TOP-1 이미지를 기준으로 사용
                anomaly_result = detector.detect_with_reference(
                    test_image_path=req.image_path,
                    reference_image_path=req.top1_image_path,
                    product_name=product,
                    output_dir=str(output_dir)
                )
            else:
                # 자동 정상 이미지 선정
                anomaly_result = detector.detect_with_normal_reference(
                    test_image_path=req.image_path,
                    product_name=product,
                    similarity_matcher=matcher,
                    output_dir=str(output_dir)
                )
            
            anomaly_score = float(anomaly_result["image_score"])
            is_anomaly = bool(anomaly_result["is_anomaly"])
            
            if req.verbose:
                print(f"✅ 이상 검출 완료: score={anomaly_score:.4f}, anomaly={is_anomaly}")
        
        except Exception as e:
            print(f"⚠️ PatchCore 이상 검출 실패: {e}")
            import traceback
            traceback.print_exc()
            # 실패해도 계속 진행 (score=0.0)

    # ========================================
    # 2) 매핑 + RAG
    # ========================================
    mapper = vlm_components["mapper"]
    rag    = vlm_components["rag"]

    defect_info = mapper.get_defect_info(product, defect)
    if not defect_info:
        raise HTTPException(404, f"불량 정보를 찾을 수 없습니다: {product}/{defect}")

    manual_ctx = {"원인": [], "조치": []}
    if rag:
        keywords   = mapper.get_search_keywords(product, defect)
        manual_ctx = rag.search_defect_manual(product, defect, keywords)
    else:
        print("⚠️ RAG 미초기화 상태 - manual_ctx는 빈 값일 수 있음")

    # ========================================
    # 3) LLM/VLM 호출
    # ========================================
    llm_analysis = None
    vlm_analysis = None

    async with httpx.AsyncClient(timeout=60) as client:
        if mode == "llm":
            payload = {
                "product": product,
                "defect_en": defect_info.en,
                "defect_ko": defect_info.ko,
                "full_name_ko": defect_info.full_name_ko,
                "anomaly_score": float(anomaly_score),  # ✅ 실제 검출 값
                "is_anomaly": bool(is_anomaly),          # ✅ 실제 판정
                "manual_context": manual_ctx,
                "max_new_tokens": req.max_new_tokens,
                "temperature": req.temperature
            }
            r = await client.post(f"{LLM_SERVER_URL}/analyze", json=payload)
            r.raise_for_status()
            llm_analysis = r.json().get("analysis", "")
        elif mode == "llm_exaone":
            # ✅ EXAONE 3.5
            payload = {
                "product": product,
                "defect_en": defect_info.en,
                "defect_ko": defect_info.ko,
                "full_name_ko": defect_info.full_name_ko,
                "anomaly_score": float(anomaly_score),
                "is_anomaly": bool(is_anomaly),
                "manual_context": manual_ctx,
                "max_new_tokens": req.max_new_tokens,
                "temperature": req.temperature,
                "top_p": 0.9,
                "repetition_penalty": 1.1,
            }
            
            r = await client.post(f"{LLM_SERVER_URL}/analyze_exaone", json=payload)
            r.raise_for_status()
            llm_analysis = r.json().get("analysis", "")

        elif mode == "vlm":
            payload = {
                "image_path": req.image_path,
                "product": product,
                "defect_en": defect_info.en,
                "defect_ko": defect_info.ko,
                "full_name_ko": defect_info.full_name_ko,
                "anomaly_score": float(anomaly_score),
                "is_anomaly": bool(is_anomaly),
                "manual_context": manual_ctx,
                "max_new_tokens": req.max_new_tokens,
                "temperature": req.temperature,
            }
            
            print(f"[VLM] 요청 payload: {payload}")
            
            r = await client.post(f"{LLM_SERVER_URL}/analyze_vlm", json=payload)
            r.raise_for_status()
            vlm_analysis = r.json().get("analysis", "")

        else:
            raise HTTPException(400, f"mode 지원 안 함: {mode}")

    # ========================================
    # 4) 결과 반환
    # ========================================
    out = {
        "status": "success",
        "product": product,
        "defect_en": defect_info.en,
        "defect_ko": defect_info.ko,
        "full_name_ko": defect_info.full_name_ko,
        "manual": manual_ctx,
        "anomaly_score": float(anomaly_score),   # ✅ 실제 값
        "is_anomaly": bool(is_anomaly),          # ✅ 실제 판정
        "processing_time": round(time.time() - t0, 2)
    }
    if llm_analysis is not None:
        out["llm_analysis"] = llm_analysis
    if vlm_analysis is not None:
        out["vlm_analysis"] = vlm_analysis
    return out

# ====================
# FastAPI 앱 생성
# ====================

# HTML 파일 서빙
WEB_DIR = Path(__file__).parent



app = FastAPI(
    title="유사도 매칭 + Anomaly Detection API",
    description="CLIP 기반 이미지 유사도 검색 + PatchCore 이상 검출 서비스",
    version="2.0.0"
)

# Static 파일 마운트 (CORS 전에 위치)
STATIC_DIR = WEB_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)  # 폴더 없으면 생성
(STATIC_DIR / "css").mkdir(exist_ok=True)
(STATIC_DIR / "js").mkdir(exist_ok=True)

# static 폴더 마운트
app.mount("/static", StaticFiles(directory=str(WEB_DIR / "static")), name="static")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ====================
# 전역 변수
# ====================

# 매처 및 디텍터 인스턴스
matcher: Optional[TopKSimilarityMatcher] = None
detector: Optional[AnomalyDetector] = None

# 설정
UPLOAD_DIR = Path("./uploads")
INDEX_DIR = Path("./index_cache")
ANOMALY_OUTPUT_DIR = Path("./anomaly_outputs")

UPLOAD_DIR.mkdir(exist_ok=True)
INDEX_DIR.mkdir(exist_ok=True)
ANOMALY_OUTPUT_DIR.mkdir(exist_ok=True)


# ====================
# 라이프사이클 이벤트
# ====================

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 초기화"""
    global matcher, detector, current_index_type
    
    print("=" * 60)
    print("유사도 매칭 + Anomaly Detection API 서버 시작")
    print("=" * 60)
    
    # 1. 유사도 매처 생성
    matcher = create_matcher(
        model_id="ViT-B-32/openai",
        device="auto",
        use_fp16=False,  # 안정성 우선
        verbose=True
    )
    
    '''
    # 기존 인덱스 로드 시도
    if (INDEX_DIR / "index_data.pt").exists():
        try:
            matcher.load_index(str(INDEX_DIR))
            print(f"✅ 기존 인덱스 로드 완료: {len(matcher.gallery_paths)}개 이미지")
        except Exception as e:
            print(f"⚠️  기존 인덱스 로드 실패: {e}")
    else:
        print("ℹ️  저장된 인덱스 없음")
    
    # 인덱스가 없으면 자동 구축 시도
    if not matcher.index_built:
        default_gallery = Path("../data/def_split")  # 변경된 경로
        
        if default_gallery.exists():
            print(f"🔄 자동 인덱스 구축 시작: {default_gallery}")
            try:
                info = matcher.build_index(str(default_gallery))
                matcher.save_index(str(INDEX_DIR))
                print(f"✅ 자동 인덱스 구축 완료: {info['num_images']}개 이미지")
            except Exception as e:
                print(f"❌ 자동 인덱스 구축 실패: {e}")
        else:
            print(f"⚠️  기본 갤러리 디렉토리 없음: {default_gallery}")
    '''
    # 2. 두 인덱스 모두 미리 구축
    print("\n" + "="*60)
    print("인덱스 사전 구축 시작")
    print("="*60)
    
    # 2-1. 불량 이미지 인덱스 구축
    defect_dir = project_root / "data" / "def_split"
    defect_index_path = INDEX_DIR / "defect"
    defect_index_path.mkdir(parents=True, exist_ok=True)
    
    if defect_dir.exists():
        try:
            print(f"\n[1/2] 불량 이미지 인덱스 구축 중...")
            print(f"      경로: {defect_dir}")
            
            info = matcher.build_index(str(defect_dir))
            matcher.save_index(str(defect_index_path))
            
            print(f"      ✅ 완료: {info['num_images']}개 이미지")
        except Exception as e:
            print(f"      ❌ 실패: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n[1/2] ⚠️  불량 이미지 디렉토리 없음: {defect_dir}")
    
    # 2-2. 정상 이미지 인덱스 구축
    #normal_dir = project_root / "data" / "ok_split"
    #normal_dir = project_root / "data" / "patchCore" / "prod1"
    global normal_dir
    normal_index_path = INDEX_DIR / "normal"
    normal_index_path.mkdir(parents=True, exist_ok=True)
    
    if normal_dir.exists():
        try:
            print(f"\n[2/2] 정상 이미지 인덱스 구축 중...")
            print(f"      경로: {normal_dir}")
            
            info = matcher.build_index(str(normal_dir))
            matcher.save_index(str(normal_index_path))
            
            print(f"      ✅ 완료: {info['num_images']}개 이미지")
        except Exception as e:
            print(f"      ❌ 실패: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n[2/2] ⚠️  정상 이미지 디렉토리 없음: {normal_dir}")
    
    print("\n" + "="*60)
    print("인덱스 사전 구축 완료")
    print("="*60)
    
    # 3. 기본 인덱스를 불량 이미지로 설정
    try:
        print("\n🔄 기본 인덱스 로드 중 (불량 이미지)...")
        if (defect_index_path / "index_data.pt").exists():
            matcher.load_index(str(defect_index_path))
            current_index_type = "defect"
            print(f"✅ 불량 이미지 인덱스 로드 완료: {len(matcher.gallery_paths)}개")
        else:
            print("⚠️  저장된 불량 인덱스를 찾을 수 없습니다")
    except Exception as e:
        print(f"⚠️  기본 인덱스 로드 실패: {e}")


    # 2. Anomaly Detector 생성
    try:
        detector = create_detector(
            bank_base_dir=str(project_root / "data" / "patchCore"),
            device="auto",
            verbose=True
        )
        print("✅ Anomaly Detector 초기화 완료")
    except Exception as e:
        print(f"⚠️  Anomaly Detector 초기화 실패: {e}")
        detector = None
    

    # VLM 컴포넌트 초기화
    init_vlm_components()
    print("✅ VLM Component 초기화 완료")

    print("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """서버 종료 시 정리"""
    print("\n서버 종료 중...")


# ====================
# API 엔드포인트 - 유사도 검색
# ====================

@app.get("/health2", response_model=HealthResponse)
async def health_check():
    """헬스체크 엔드포인트"""
    return HealthResponse(
        status="healthy",
        message="API 서버가 정상 작동 중입니다",
        index_built=matcher.index_built if matcher else False,
        gallery_size=len(matcher.gallery_paths) if matcher and matcher.index_built else 0
    )


@app.post("/build_index")
async def build_index(request: BuildIndexRequest):
    """갤러리 이미지 인덱스 구축"""
    if matcher is None:
        raise HTTPException(status_code=500, detail="매처가 초기화되지 않았습니다")
    
    gallery_dir = Path(request.gallery_dir)
    if not gallery_dir.exists():
        raise HTTPException(status_code=404, detail=f"디렉토리를 찾을 수 없습니다: {gallery_dir}")
    
    try:
        info = matcher.build_index(str(gallery_dir))
        
        if request.save_index:
            save_dir = request.index_save_dir or str(INDEX_DIR)
            matcher.save_index(save_dir)
            info["index_saved"] = True
            info["index_save_path"] = save_dir
        else:
            info["index_saved"] = False
        
        return JSONResponse(content=info)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"인덱스 구축 실패: {str(e)}")


@app.post("/search", response_model=SearchResponse)
async def search_by_path(request: SearchRequest):
    """이미지 경로로 유사 이미지 검색"""
    if matcher is None:
        raise HTTPException(status_code=500, detail="매처가 초기화되지 않았습니다")
    
    if not matcher.index_built:
        raise HTTPException(status_code=400, detail="인덱스가 구축되지 않았습니다")
    
    query_path = Path(request.query_image_path)
    if not query_path.exists():
        raise HTTPException(status_code=404, detail=f"이미지를 찾을 수 없습니다: {query_path}")
    
    try:
        result = matcher.search(str(query_path), top_k=request.top_k)
        
        return SearchResponse(
            status="success",
            query_image=result.query_image,
            top_k_results=result.top_k_results,
            total_gallery_size=result.total_gallery_size,
            model_info=result.model_info
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"검색 실패: {str(e)}")


@app.post("/search/upload")
async def search_upload(
    file: UploadFile = File(...),
    top_k: int = 5
):
    """이미지 업로드 및 유사도 검색"""
    try:
        if matcher is None:
            raise HTTPException(status_code=500, detail="매처가 초기화되지 않았습니다")
        
        if not matcher.index_built:
            raise HTTPException(status_code=400, detail="인덱스가 구축되지 않았습니다")
        
        # 1. 파일 저장
        file_path = uploads_dir / file.filename
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"파일 저장 완료: {file_path}")
        
        # 2. 유사도 검색 수행
        result = matcher.search(
            str(file_path),
            top_k=top_k
        )
        
        # 3. 결과 반환 - result는 SimilarityResult 객체
        return {
            "status": "success",
            "uploaded_file": str(file_path),
            "top_k_results": result.top_k_results,  # 이미 리스트
            "total_gallery_size": result.total_gallery_size
        }
        
    except Exception as e:
        print(f"검색 오류: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/index/info")
async def get_index_info():
    """현재 인덱스 정보 조회"""
    if matcher is None:
        raise HTTPException(status_code=500, detail="매처가 초기화되지 않았습니다")
    
    if not matcher.index_built:
        return JSONResponse(content={
            "status": "no_index",
            "message": "인덱스가 구축되지 않았습니다"
        })
    
    return JSONResponse(content={
        "status": "index_built",
        "gallery_size": len(matcher.gallery_paths),
        "model_id": matcher.model_id,
        "device": matcher.device,
        "faiss_enabled": matcher.faiss_index is not None,
        "sample_paths": matcher.gallery_paths[:5]
    })




@app.post("/index/switch")
async def switch_index_endpoint(index_type: str):
    """
    인덱스 타입 전환 API
    
    Query Parameters:
        index_type: 'defect' 또는 'normal'
    """
    try:
        result = await switch_index(index_type)
        return result
    except ValueError as e:
        raise HTTPException(400, str(e))
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/index/status")
async def get_index_status():
    """현재 인덱스 상태 조회"""
    if matcher is None:
        return {
            "status": "error",
            "message": "매처가 초기화되지 않았습니다",
            "current_index_type": None,
            "gallery_count": 0
        }
    
    return {
        "status": "success",
        "current_index_type": current_index_type,
        "gallery_count": len(matcher.gallery_paths) if matcher.gallery_paths else 0,
        "index_built": matcher.index_built,
        "model_id": matcher.model_id if hasattr(matcher, 'model_id') else None
    }


# ====================
# API 엔드포인트 - Anomaly Detection
# ====================

# /detect_anomaly 엔드포인트 수정
@app.post("/detect_anomaly", response_model=AnomalyDetectResponse)
async def detect_anomaly(request: AnomalyDetectRequest):
    """이상 검출 수행"""
    if detector is None:
        raise HTTPException(status_code=500, detail="Anomaly Detector가 초기화되지 않았습니다")
    
    test_path = Path(request.test_image_path)
    if not test_path.exists():
        raise HTTPException(status_code=404, detail=f"테스트 이미지를 찾을 수 없습니다: {test_path}")
    
    try:
        # 출력 디렉토리 생성
        output_dir = ANOMALY_OUTPUT_DIR / test_path.stem
        output_dir.mkdir(exist_ok=True)
        
        # reference_image_path가 제공되지 않으면 정상 이미지에서 자동 검색
        if not request.reference_image_path:
            if matcher is None:
                raise HTTPException(status_code=500, detail="유사도 매처가 초기화되지 않았습니다")
            
            result = detector.detect_with_normal_reference(
                test_image_path=str(test_path),
                product_name=request.product_name,
                similarity_matcher=matcher,
                output_dir=str(output_dir)
            )
        else:
            # 사용자가 고른 TOP-1(참조 이미지)을 실제로 사용
            ref_path = Path(request.reference_image_path)
            if not ref_path.exists():
                raise HTTPException(status_code=404, detail=f"기준 이미지를 찾을 수 없습니다: {ref_path}")
            result = detector.detect_with_reference(
                test_image_path=str(test_path),
                reference_image_path=str(ref_path),
                product_name=request.product_name,
                output_dir=str(output_dir)
            )
        
        # URL 생성
        return AnomalyDetectResponse(
            status="success",
            product_name=result["product_name"],
            image_score=result["image_score"],
            pixel_tau=result["pixel_tau"],
            image_tau=result["image_tau"],
            is_anomaly=result["is_anomaly"],
            reference_normal_url=f"/api/image/{result['reference_image_path']}" if "reference_image_path" in result else "",
            mask_url=f"/anomaly/image/{test_path.stem}/mask.png",
            overlay_url=f"/anomaly/image/{test_path.stem}/overlay.png",
            comparison_url=f"/anomaly/image/{test_path.stem}/comparison.png" if "comparison_path" in result else None
        )
    
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"이상 검출 실패: {str(e)}")


@app.post("/detect_anomaly/upload")
async def detect_anomaly_upload(
    test_file: UploadFile = File(...),
    reference_file: Optional[UploadFile] = File(None),
    product_name: Optional[str] = None
):
    """업로드된 이미지로 이상 검출"""
    if detector is None:
        raise HTTPException(status_code=500, detail="Anomaly Detector가 초기화되지 않았습니다")
    
    try:
        # 테스트 이미지 저장
        test_path = UPLOAD_DIR / test_file.filename
        with test_path.open("wb") as buffer:
            shutil.copyfileobj(test_file.file, buffer)
        
        # 기준 이미지 저장 (있는 경우)
        ref_path = None
        if reference_file:
            ref_path = UPLOAD_DIR / reference_file.filename
            with ref_path.open("wb") as buffer:
                shutil.copyfileobj(reference_file.file, buffer)
        
        # 출력 디렉토리
        output_dir = ANOMALY_OUTPUT_DIR / test_path.stem
        output_dir.mkdir(exist_ok=True)
        
        # 이상 검출
        if ref_path:
           
            result = detector.detect_with_reference(
                test_image_path=str(test_path),
                reference_image_path=str(ref_path),
                product_name=product_name,
                output_dir=str(output_dir)
            )
           
           
        else:
            result = detector.detect(
                test_image_path=str(test_path),
                product_name=product_name,
                output_dir=str(output_dir)
            )
        
        return JSONResponse(content={
            "status": "success",
            "product_name": result["product_name"],
            "image_score": result["image_score"],
            "is_anomaly": result["is_anomaly"],
            "heatmap_url": f"/anomaly/image/{test_path.stem}/heatmap.png",
            "mask_url": f"/anomaly/image/{test_path.stem}/mask.png",
            "overlay_url": f"/anomaly/image/{test_path.stem}/overlay.png",
            "comparison_url": f"/anomaly/image/{test_path.stem}/comparison.png" if "comparison_path" in result else None
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"이상 검출 실패: {str(e)}")


@app.get("/anomaly/image/{result_id}/{filename}")
async def serve_anomaly_image(result_id: str, filename: str):
    """이상 검출 결과 이미지 제공"""
    file_path = ANOMALY_OUTPUT_DIR / result_id / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="이미지를 찾을 수 없습니다")
    
    return FileResponse(file_path, media_type="image/png")


def get_next_seqno(base_dir: Path, product_name: str, defect_name: str) -> int:
    """특정 제품/불량의 다음 seqno 반환"""
    pattern = f"{product_name}_{defect_name}_*"
    existing_files = list(base_dir.glob(pattern))
    
    max_seqno = 0
    for file_path in existing_files:
        stem = file_path.stem
        parts = stem.split('_')
        
        if len(parts) >= 3:
            try:
                seqno = int(parts[-1])
                max_seqno = max(max_seqno, seqno)
            except ValueError:
                continue
    
    return max_seqno + 1

@app.post("/register_defect")
async def register_defect(
    file: UploadFile = File(...),
    product_name: str = Form(...),  # Query → Form으로 변경
    defect_name: str = Form(...)    # Query → Form으로 변경
):
    """불량 이미지 등록"""
    # 저장 경로 설정
    defect_dir = Path(f"../data/def_split")
    defect_dir.mkdir(parents=True, exist_ok=True)
    
    # 현재 등록된 파일 중 최대 seqno 찾기
    pattern = f"{product_name}_{defect_name}_*.{Path(file.filename).suffix}"
    existing_files = list(defect_dir.glob(f"{product_name}_{defect_name}_*"))
    
    # seqno 추출 및 최대값 찾기
    max_seqno = 0
    for existing_file in existing_files:
        try:
            # 파일명 형식: prod1_hole_001.jpg
            stem = existing_file.stem  # prod1_hole_001
            parts = stem.split('_')
            if len(parts) >= 3:
                seqno_str = parts[-1]  # 001
                seqno = int(seqno_str)
                max_seqno = max(max_seqno, seqno)
        except (ValueError, IndexError):
            continue
    
    # 새로운 seqno
    new_seqno = max_seqno + 1
    
    # 파일명 생성
    ext = Path(file.filename).suffix
    new_filename = f"{product_name}_{defect_name}_{new_seqno:03d}{ext}"
    save_path = defect_dir / new_filename
    
    # 저장
    with save_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    


    # 인덱스 재구축 (선택사항)
    if matcher and matcher.index_built:
        try:
            matcher.build_index(str(defect_dir))
            matcher.save_index(str(INDEX_DIR))
            index_rebuilt = True
        except Exception as e:
            print(f"인덱스 재구축 실패: {e}")
            index_rebuilt = False
    else:
        index_rebuilt = False
    
    return JSONResponse(content={
        "status": "success",
        "saved_path": str(save_path),
        "filename": new_filename,
        "product_name": product_name,
        "defect_name": defect_name,
        "seqno": new_seqno,
        "index_rebuilt": index_rebuilt
    })

@app.get("/defect/stats/{product_name}/{defect_name}")
async def get_defect_stats(product_name: str, defect_name: str):
    """특정 불량의 통계 조회"""
    defect_dir = Path("../data/def_split")
    
    if not defect_dir.exists():
        return JSONResponse(content={
            "product_name": product_name,
            "defect_name": defect_name,
            "total_count": 0,
            "next_seqno": 1
        })
    
    pattern = f"{product_name}_{defect_name}_*"
    existing_files = list(defect_dir.glob(pattern))
    next_seqno = get_next_seqno(defect_dir, product_name, defect_name)
    
    return JSONResponse(content={
        "product_name": product_name,
        "defect_name": defect_name,
        "total_count": len(existing_files),
        "next_seqno": next_seqno,
        "files": [f.name for f in sorted(existing_files)]
    })

@app.get("/defect_config.json")
async def serve_defect_config():
    """불량 설정 파일 제공"""
    config_path = WEB_DIR / "defect_config.json"
    if not config_path.exists():
        # 기본 설정 반환
        return JSONResponse(content={
            "products": {
                "prod1": {"name": "제품1", "defects": ["hole", "burr", "scratch"]}
            }
        })
    return FileResponse(config_path)

# ====================
# 이미지 서빙 및 정적 파일
# ====================

@app.get("/api/image/{image_path:path}")
async def serve_image(image_path: str):
    """이미지 파일 제공 엔드포인트"""
    try:
        # 상대 경로 정규화
        if image_path.startswith("../"):
            image_path = image_path.replace("../", "")
        
        # 경로 처리
        if image_path.startswith("uploads/"):
            file_path = uploads_dir / image_path.replace("uploads/", "")
        elif image_path.startswith("data/"):
            file_path = project_root / image_path
        else:
            # 기본적으로 project_root 기준
            file_path = project_root / image_path
        
        print(f"이미지 서빙 시도: {file_path}")
        
        if not file_path.exists():
            raise HTTPException(404, f"이미지를 찾을 수 없습니다: {file_path}")
        
        return FileResponse(str(file_path))
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"이미지 서빙 오류: {e}")
        raise HTTPException(500, str(e))

@app.delete("/uploads/clean")
async def clean_uploads():
    """업로드 디렉토리 정리"""
    try:
        for file in UPLOAD_DIR.glob("*"):
            if file.is_file():
                file.unlink()
        
        return JSONResponse(content={
            "status": "success",
            "message": "업로드 디렉토리가 정리되었습니다"
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"정리 실패: {str(e)}")




@app.get("/matching.html")
async def serve_matching():
    """matching.html 서빙"""
    return FileResponse(WEB_DIR / "matching.html")

@app.get("/")
async def root():
    """루트 접근 시 matching.html로 리다이렉트"""
    return FileResponse(WEB_DIR / "matching.html")

# ====================
# VLM 관련 실행 코드
# ====================

@app.post("/generate_manual")
async def generate_manual(request: dict):
    """
    불량 매뉴얼 생성 (기본 버전 - RAG만)
    
    Request Body:
    {
        "image_path": "path/to/image.jpg",
        "product": "prod1",
        "defect": "burr"
    }
    
    Response:
    {
        "status": "success",
        "defect_info": {...},
        "manual": {"원인": [...], "조치": [...]},
        "message": "매뉴얼 검색 완료"
    }
    """
    try:
        image_path = request.get("image_path")
        product = request.get("product")
        defect = request.get("defect")
        
        if not all([image_path, product, defect]):
            raise HTTPException(400, "image_path, product, defect 필수")
        
        # DefectMapper로 정보 조회
        mapper = vlm_components["mapper"]
        defect_info = mapper.get_defect_info(product, defect)
        
        if not defect_info:
            raise HTTPException(404, f"불량 정보를 찾을 수 없습니다: {product}/{defect}")
        
        # RAG 검색
        rag = vlm_components["rag"]
        if not rag:
            raise HTTPException(503, "RAG 서비스가 초기화되지 않았습니다")
        
        keywords = mapper.get_search_keywords(product, defect)
        manual_context = rag.search_defect_manual(product, defect, keywords)
        
        return {
            "status": "success",
            "defect_info": {
                "en": defect_info.en,
                "ko": defect_info.ko,
                "full_name_ko": defect_info.full_name_ko
            },
            "manual": manual_context,
            "message": "매뉴얼 검색 완료"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"매뉴얼 생성 오류: {str(e)}")



# LLM 서버 URL
LLM_SERVER_URL = "http://localhost:5001"

# 전역 변수에서 llm_manager 제거 또는 None으로 설정
llm_manager = None  # 사용하지 않음

# LLM 서버 호출 함수 추가
async def call_llm_server(
    product: str,
    defect_en: str,
    defect_ko: str,
    full_name_ko: str,
    anomaly_score: float,
    is_anomaly: bool,
    manual_context: Dict[str, List[str]],
    #max_new_tokens=400,  # ✅ 줄임 (512 → 400)
    max_new_tokens=1024,  # ✅ 줄임 (512 → 400)
    temperature=0.2      # ✅ 더 낮춤 (0.3 → 0.2)
) -> str:
    """LLM 서버에 분석 요청"""
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{LLM_SERVER_URL}/analyze",
                json={
                    "product": product,
                    "defect_en": defect_en,
                    "defect_ko": defect_ko,
                    "full_name_ko": full_name_ko,
                    "anomaly_score": anomaly_score,
                    "is_anomaly": is_anomaly,
                    "manual_context": manual_context,
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "model_provider": "hyperclovax"  # ✅ 추가: 기본값 또는 req에서 받기
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["analysis"]
            else:
                raise Exception(f"LLM 서버 오류: {response.status_code} - {response.text}")
                
    except httpx.ConnectError:
        raise Exception("LLM 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.")
    except Exception as e:
        raise Exception(f"LLM 서버 호출 실패: {str(e)}")



@app.post("/generate_manual_advanced")
async def generate_manual_advanced(request: dict):
    """
    고급 불량 분석 (통합 파이프라인)
    - 유사도 검색 → PatchCore 이상 검출 → RAG → VLM
    """
    import time
    start_time = time.time()
    
    try:
        image_path = request.get("image_path")
        if not image_path:
            raise HTTPException(400, "image_path 필수")
        
        # 경로 정규화
        image_path_obj = Path(image_path)
        
        if not image_path_obj.is_absolute():
            if image_path.startswith("./uploads/"):
                filename = image_path.replace("./uploads/", "")
                image_path_obj = uploads_dir / filename
            elif image_path.startswith("uploads/"):
                filename = image_path.replace("uploads/", "")
                image_path_obj = uploads_dir / filename
            else:
                image_path_obj = project_root / image_path
        
        print(f"\n{'='*60}")
        print(f"고급 분석 시작: {image_path_obj.name}")
        print(f"{'='*60}")
        
        if not image_path_obj.exists():
            raise HTTPException(404, f"이미지를 찾을 수 없습니다: {image_path_obj}")

        result = {
            "status": "success",
            "steps": []
        }
        
        # Step 1: 유사도 검색으로 제품명 추출
        print("\n[Step 1] 유사도 검색...")
        result["steps"].append("1. 유사도 검색 중...")
        
        if not matcher or not matcher.index_built:
            raise HTTPException(503, "인덱스가 구축되지 않았습니다")
        
        search_result = matcher.search(str(image_path_obj), top_k=5)
        
        if not search_result.top_k_results:
            raise HTTPException(404, "유사한 이미지를 찾을 수 없습니다")
        
        # TOP-K 중에서 불량 이미지 찾기 (def가 포함된 것)
        product = None
        defect = None
        top_result = None
        
        for result_item in search_result.top_k_results:
            filename = Path(result_item["image_path"]).stem
            parts = filename.split("_")
            
            if len(parts) >= 3:  # product_def_숫자 형태
                temp_product = parts[0]
                temp_defect = parts[1]
                
                # 'ok', 'normal' 등이 아닌 불량명 찾기
                if temp_defect.lower() not in ['ok', 'normal', 'good']:
                    product = temp_product
                    defect = temp_defect
                    top_result = result_item
                    print(f"✅ 불량 매칭: {filename} → 제품:{product}, 불량:{defect}")
                    break
        
        if not product or not defect:
            raise HTTPException(400, "불량 이미지를 찾을 수 없습니다")
        
        result["similarity"] = {
            "top_match": top_result["image_path"],
            "similarity": float(top_result["similarity_score"]),
            "product": product,
            "defect": defect
        }
        
        # Step 2: 불량 정보 조회
        print(f"\n[Step 2] 불량 정보 조회: {product}/{defect}")
        result["steps"].append("2. 불량 정보 조회 중...")
        
        mapper = vlm_components["mapper"]
        if not mapper:
            raise HTTPException(503, "DefectMapper가 초기화되지 않았습니다")
            
        defect_info = mapper.get_defect_info(product, defect)
        
        if not defect_info:
            raise HTTPException(404, f"불량 정보를 찾을 수 없습니다: {product}/{defect}")
        
        result["defect_info"] = {
            "product": product,
            "en": defect_info.en,
            "ko": defect_info.ko,
            "full_name_ko": defect_info.full_name_ko
        }
        
        # Step 3: PatchCore 이상 검출 (정상 이미지 기준)
        print(f"\n[Step 3] 이상 영역 검출...")
        result["steps"].append("3. 이상 영역 검출 중...")
        
        if not detector:
            raise HTTPException(503, "AnomalyDetector가 초기화되지 않았습니다")
        
        # 출력 디렉토리
        output_dir = ANOMALY_OUTPUT_DIR / image_path_obj.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 이상 검출 - 정상 이미지 기준으로 검출
        anomaly_result = detector.detect_with_normal_reference(
            test_image_path=str(image_path_obj),
            product_name=product,  # 추출된 제품명 사용
            similarity_matcher=matcher,
            output_dir=str(output_dir)
        )
        
        print(f"✅ 이상 점수: {anomaly_result['image_score']:.4f}")
        print(f"   정상 기준: {anomaly_result.get('reference_image_path', 'N/A')}")
        
        result["anomaly"] = {
            "score": float(anomaly_result["image_score"]),
            "is_anomaly": anomaly_result["is_anomaly"],
            "normal_image_url": f"/api/image/{anomaly_result.get('reference_image_path', '')}",
            "overlay_image_url": f"/anomaly/image/{image_path_obj.stem}/overlay.png",
            "mask_image_url": f"/anomaly/image/{image_path_obj.stem}/mask.png"
        }
        
        # Step 4: RAG 매뉴얼 검색
        print(f"\n[Step 4] 매뉴얼 검색...")
        result["steps"].append("4. 매뉴얼 검색 중...")
        
        rag = vlm_components.get("rag")
        
        if rag is None:
            print("⚠️  RAG가 비활성화되어 있습니다 (PDF 파일 없음)")
            result["manual"] = {
                "원인": ["RAG 서비스가 비활성화되어 있습니다"],
                "조치": ["PDF 매뉴얼 파일을 추가하세요"]
            }
        else:
            keywords = mapper.get_search_keywords(product, defect)
            raw_manual_context = rag.search_defect_manual(product, defect, keywords)

            # ✅ 매뉴얼 정제: 해당 불량만 필터링
            manual_context = {
                "원인": [],
                "조치": []
            }

            for cause_text in raw_manual_context.get("원인", []):
                # defect 키워드 포함 여부 확인
                if defect.lower() in cause_text.lower():
                    # 해당 불량 섹션만 추출
                    lines = []
                    in_section = False
                    for line in cause_text.split('\n'):
                        line = line.strip()
                        if defect.lower() in line.lower():
                            in_section = True
                        elif line.startswith(('burr', 'hole', 'scratch')) and defect.lower() not in line.lower():
                            break  # 다른 불량 시작
                        elif in_section and line and line.startswith('•'):
                            lines.append(line)
                    
                    if lines:
                        manual_context["원인"].append('\n'.join(lines[:3]))

            # 조치도 동일하게 처리
            for action_text in raw_manual_context.get("조치", []):
                if defect.lower() in action_text.lower():
                    lines = []
                    in_section = False
                    for line in action_text.split('\n'):
                        line = line.strip()
                        if defect.lower() in line.lower():
                            in_section = True
                        elif line.startswith(('burr', 'hole', 'scratch')) and defect.lower() not in line.lower():
                            break
                        elif in_section and line and line.startswith('•'):
                            lines.append(line)
                    
                    if lines:
                        manual_context["조치"].append('\n'.join(lines[:3]))

            print(f"✅ 매뉴얼 정제 완료:")
            print(f"  원인: {len(manual_context['원인'])}개")
            print(f"  조치: {len(manual_context['조치'])}개")
        
        # Step 5: VLM 분석 (선택적)
        print("[Step 5] LLM 분석...")
        try:
            llm_analysis = await call_llm_server(
                product=product,
                defect_en=defect_info.en,
                defect_ko=defect_info.ko,
                full_name_ko=defect_info.full_name_ko,
                anomaly_score=float(result["anomaly"]["score"]),     # ✅ 여기!
                is_anomaly=bool(result["anomaly"]["is_anomaly"]),    # ✅ 여기!
                manual_context=result.get("manual", {}),
                #max_new_tokens=512,  # 충분히 길게
                max_new_tokens=1024,  # 충분히 길게
                temperature=0.3       # ✅ 낮춤 (0.7 → 0.3): 더 일관된 출력
            )
            print(f"✅ LLM 분석 완료 ({len(llm_analysis)} 문자)")
            
        except Exception as e:
            print(f"⚠️  LLM 분석 실패: {e}")
            llm_analysis = f"LLM 분석 실패: {str(e)}"

        result["vlm_analysis"] = llm_analysis
        # 처리 시간
        result["processing_time"] = round(time.time() - start_time, 2)
        
        print(f"\n{'='*60}")
        print(f"분석 완료: {result['processing_time']}초")
        print(f"{'='*60}\n")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"고급 분석 오류: {str(e)}")
   
# ====== 라우트: LLM 전용 ======
@app.post("/manual/generate/llm")
async def manual_generate_llm(req: ManualGenRequest):
    try:
        return await _manual_core("llm", req)
    except HTTPException:
        raise
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(500, f"LLM 생성 오류: {str(e)}")

# ✅ EXAONE 전용 엔드포인트 추가
@app.post("/manual/generate/llm_exaone")
async def manual_generate_llm_exaone(req: ManualGenRequest):
    """EXAONE 3.5 기반 대응 매뉴얼 생성"""
    try:
        return await _manual_core("llm_exaone", req)
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"EXAONE 생성 오류: {str(e)}")

# ====== 라우트: VLM 전용 ======
@app.post("/manual/generate/vlm")
async def manual_generate_vlm(req: ManualGenRequest):
    try:
        return await _manual_core("vlm", req)
    except HTTPException:
        raise
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(500, f"VLM 생성 오류: {str(e)}")

@app.get("/vlm/status")
async def vlm_status():
    """VLM 컴포넌트 상태 확인"""
    return {
        "mapper_loaded": vlm_components["mapper"] is not None,
        "rag_loaded": vlm_components["rag"] is not None,
        "vlm_loaded": vlm_components["vlm"] is not None,
        "prompt_builder_loaded": vlm_components["prompt_builder"] is not None
    }


@app.post("/analyze_defect_complete")
async def analyze_defect_complete(
    image_path: str,
    product: Optional[str] = None,
    defect: Optional[str] = None
):
    """완전한 불량 분석 파이프라인"""
    
    # 1. 유사도 검색
    search_result = matcher.search(image_path, top_k=5)
    top1 = search_result.top_k_results[0]
    
    # product/defect 추출
    if not product or not defect:
        filename = Path(top1["image_path"]).stem
        product, defect, _ = filename.split("_")
    
    # 2. PatchCore 이상 검출
    anomaly_result = detector.detect_with_normal_reference(
        test_image_path=image_path,
        product_name=product,
        similarity_matcher=matcher,
        output_dir=str(ANOMALY_OUTPUT_DIR / Path(image_path).stem)
    )
    
    # 3. RAG 매뉴얼 검색
    mapper = vlm_components["mapper"]
    rag = vlm_components["rag"]
    
    keywords = mapper.get_search_keywords(product, defect)
    manual_context = rag.search_defect_manual(product, defect, keywords)
    
    # 4. LLM 답변 생성
    defect_info = mapper.get_defect_info(product, defect)
    llm_analysis = await call_llm_server(
        product=product,
        defect_en=defect_info.en,
        defect_ko=defect_info.ko,
        full_name_ko=defect_info.full_name_ko,
        anomaly_score=float(anomaly_result["image_score"]),  # ✅ 실제 스코어
        is_anomaly=bool(anomaly_result["is_anomaly"]),        # ✅ 실제 판정
        manual_context=manual_context
    )
    
    return {
        "similarity": search_result,
        "anomaly": anomaly_result,
        "manual": manual_context,
        "llm_analysis": llm_analysis
    }

@app.post("/vlm/reload")
async def vlm_reload():
    """VLM 컴포넌트 재로드"""
    try:
        init_vlm_components()
        return {"status": "success", "message": "VLM 컴포넌트 재로드 완료"}
    except Exception as e:
        raise HTTPException(500, f"재로드 오류: {str(e)}")


# ====================
# 서버 실행
# ====================

if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=5000,  # 변경된 포트
        reload=True,
        log_level="info"
    )
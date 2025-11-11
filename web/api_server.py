"""
TOP-K 유사도 매칭 + Anomaly Detection + LLM 통합 API 서버
FastAPI 기반으로 외부 웹서버에서 호출 가능한 REST API 제공

주요 기능:
1. CLIP 기반 유사도 검색 (TOP-K)
2. PatchCore 이상 검출
3. RAG 기반 매뉴얼 검색
4. LLM 대응 방안 생성
"""

import os
import sys
import time
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import httpx
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ====================
# 프로젝트 루트 경로 설정
# ====================
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
uploads_dir = project_root / "web" / "uploads"
uploads_dir.mkdir(parents=True, exist_ok=True)

# ====================
# 모듈 import
# ====================
from modules.similarity_matcher import TopKSimilarityMatcher, create_matcher
from modules.anomaly_detector import AnomalyDetector, create_detector
from modules.vlm import RAGManager, DefectMapper

# ====================
# FastAPI 앱 생성
# ====================
WEB_DIR = Path(__file__).parent

app = FastAPI(
    title="유사도 매칭 + Anomaly Detection + LLM API",
    description="CLIP 기반 이미지 유사도 검색 + PatchCore 이상 검출 + LLM 대응 매뉴얼 생성",
    version="3.0.0"
)

# Static 파일 마운트
STATIC_DIR = WEB_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)
(STATIC_DIR / "css").mkdir(exist_ok=True)
(STATIC_DIR / "js").mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

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
# 유사도 매처 및 이상 검출기
matcher: Optional[TopKSimilarityMatcher] = None
detector: Optional[AnomalyDetector] = None

# VLM 컴포넌트
vlm_components = {
    "rag": None,
    "mapper": None,
}

# 디렉토리 설정
UPLOAD_DIR = Path("./uploads")
INDEX_DIR = Path("./index_cache")
ANOMALY_OUTPUT_DIR = Path("./anomaly_outputs")

UPLOAD_DIR.mkdir(exist_ok=True)
INDEX_DIR.mkdir(exist_ok=True)
ANOMALY_OUTPUT_DIR.mkdir(exist_ok=True)

# LLM 서버 URL
LLM_SERVER_URL = "http://localhost:5001"

# ====================
# Pydantic 모델
# ====================
class SearchResponse(BaseModel):
    """검색 응답"""
    status: str
    query_image: str
    top_k_results: List[dict]
    total_gallery_size: int
    model_info: str


class AnomalyDetectResponse(BaseModel):
    """이상 검출 응답"""
    status: str
    product_name: str
    image_score: float
    pixel_tau: float
    image_tau: float
    is_anomaly: bool
    reference_normal_url: str
    mask_url: str
    overlay_url: str
    comparison_url: Optional[str] = None


class HealthResponse(BaseModel):
    """헬스체크 응답"""
    status: str
    message: str
    index_built: bool
    gallery_size: int


class ManualGenRequest(BaseModel):
    """매뉴얼 생성 요청"""
    image_path: str
    top1_image_path: Optional[str] = None
    product_name: Optional[str] = None
    defect_name: Optional[str] = None
    anomaly_score: Optional[float] = None
    is_anomaly: Optional[bool] = None
    max_new_tokens: int = 512
    temperature: float = 0.7
    verbose: bool = False

# ====================
# VLM 컴포넌트 초기화
# ====================
def init_vlm_components():
    """VLM 컴포넌트 초기화 (서버 시작 시 1회)"""
    global vlm_components
    
    try:
        print("\n" + "="*50)
        print("VLM 컴포넌트 초기화 중...")
        print("="*50)
        
        # 경로 설정
        vector_store_path = project_root / "manual_store"
        mapping_file = project_root / "web" / "defect_mapping.json"
        
        # 1. DefectMapper 초기화
        print("\n1. DefectMapper 초기화...")
        if not mapping_file.exists():
            print("⚠️  매핑 파일이 없습니다. 기본 파일을 생성합니다...")
            from modules.vlm.defect_mapper import create_default_mapping
            create_default_mapping(mapping_file)
        
        vlm_components["mapper"] = DefectMapper(mapping_file)
        
        # 2. RAGManager 초기화
        print("\n2. RAGManager 초기화...")
        pdf_candidates = [
            vector_store_path / "prod1_menual.pdf",
            project_root / "prod1_menual.pdf"
        ]
        
        pdf_path = None
        for candidate in pdf_candidates:
            if candidate.exists():
                pdf_path = candidate
                print(f"✅ PDF 파일 발견: {pdf_path}")
                break
        
        if not pdf_path:
            print(f"⚠️  PDF 파일을 찾을 수 없습니다")
            print(f"   확인 경로: {[str(p) for p in pdf_candidates]}")
            print("   VLM 기능이 제한됩니다.")
        else:
            vlm_components["rag"] = RAGManager(
                pdf_path=pdf_path,
                vector_store_path=vector_store_path,
                device="cuda",
                verbose=True
            )
        
        print("\n" + "="*50)
        print("✅ VLM 컴포넌트 초기화 완료")
        print("="*50 + "\n")
        
    except Exception as e:
        print(f"\n❌ VLM 초기화 오류: {e}")
        import traceback
        traceback.print_exc()

# ====================
# 라이프사이클 이벤트
# ====================
@app.on_event("startup")
async def startup_event():
    """서버 시작 시 초기화"""
    global matcher, detector
    
    print("=" * 60)
    print("유사도 매칭 + Anomaly Detection + LLM API 서버 시작")
    print("=" * 60)
    
    # 1. 유사도 매처 생성
    print("\n[1/3] 유사도 매처 초기화...")
    matcher = create_matcher(
        model_id="ViT-B-32/openai",
        device="auto",
        use_fp16=False,
        verbose=True
    )
    
    # 기존 인덱스 로드 시도
    if (INDEX_DIR / "index_data.pt").exists():
        try:
            matcher.load_index(str(INDEX_DIR))
            print(f"✅ 기존 인덱스 로드 완료: {len(matcher.gallery_paths)}개 이미지")
        except Exception as e:
            print(f"⚠️  기존 인덱스 로드 실패: {e}")
    
    # 인덱스가 없으면 자동 구축
    if not matcher.index_built:
        default_gallery = project_root / "data" / "def_split"
        
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
    
    # 2. Anomaly Detector 생성
    print("\n[2/3] Anomaly Detector 초기화...")
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
    
    # 3. VLM 컴포넌트 초기화
    print("\n[3/3] VLM 컴포넌트 초기화...")
    init_vlm_components()
    
    print("\n" + "=" * 60)
    print("✅ 서버 초기화 완료")
    print("=" * 60 + "\n")


@app.on_event("shutdown")
async def shutdown_event():
    """서버 종료 시 정리"""
    print("\n서버 종료 중...")

# ====================
# 헬스체크 엔드포인트
# ====================
@app.get("/health2", response_model=HealthResponse)
async def health_check():
    """헬스체크 엔드포인트 (ALB용)"""
    return HealthResponse(
        status="healthy",
        message="API 서버가 정상 작동 중입니다",
        index_built=matcher.index_built if matcher else False,
        gallery_size=len(matcher.gallery_paths) if matcher and matcher.index_built else 0
    )


@app.get("/")
async def root():
    """루트 접근 시 matching.html로 리다이렉트"""
    return FileResponse(WEB_DIR / "matching.html")


@app.get("/matching.html")
async def serve_matching():
    """matching.html 서빙"""
    return FileResponse(WEB_DIR / "matching.html")

# ====================
# 유사도 검색 엔드포인트
# ====================
@app.post("/search/upload")
async def search_upload(
    file: UploadFile = File(...),
    top_k: int = 5
):
    """
    이미지 업로드 및 유사도 검색
    
    Args:
        file: 업로드 이미지 파일
        top_k: 상위 K개 결과 (기본값: 5)
    
    Returns:
        {
            "status": "success",
            "uploaded_file": "경로",
            "top_k_results": [...],
            "total_gallery_size": N
        }
    """
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
        result = matcher.search(str(file_path), top_k=top_k)
        
        # 3. 결과 반환
        return {
            "status": "success",
            "uploaded_file": str(file_path),
            "top_k_results": result.top_k_results,
            "total_gallery_size": result.total_gallery_size
        }
        
    except Exception as e:
        print(f"검색 오류: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ====================
# 이상 검출 엔드포인트
# ====================
@app.post("/detect_anomaly", response_model=AnomalyDetectResponse)
async def detect_anomaly(
    test_image_path: str,
    reference_image_path: Optional[str] = None,
    product_name: Optional[str] = None
):
    """
    PatchCore 이상 검출 수행
    
    Args:
        test_image_path: 테스트 이미지 경로
        reference_image_path: 기준 이미지 경로 (없으면 자동 선정)
        product_name: 제품명 (파일명에서 자동 추출 가능)
    
    Returns:
        AnomalyDetectResponse
    """
    if detector is None:
        raise HTTPException(status_code=500, detail="Anomaly Detector가 초기화되지 않았습니다")
    
    test_path = Path(test_image_path)
    if not test_path.exists():
        raise HTTPException(status_code=404, detail=f"테스트 이미지를 찾을 수 없습니다: {test_path}")
    
    try:
        # 출력 디렉토리 생성
        output_dir = ANOMALY_OUTPUT_DIR / test_path.stem
        output_dir.mkdir(exist_ok=True)
        
        # reference_image_path가 제공되지 않으면 자동 검색
        if not reference_image_path:
            if matcher is None:
                raise HTTPException(status_code=500, detail="유사도 매처가 초기화되지 않았습니다")
            
            result = detector.detect_with_normal_reference(
                test_image_path=str(test_path),
                product_name=product_name,
                similarity_matcher=matcher,
                output_dir=str(output_dir)
            )
        else:
            # 사용자가 제공한 기준 이미지 사용
            ref_path = Path(reference_image_path)
            if not ref_path.exists():
                raise HTTPException(status_code=404, detail=f"기준 이미지를 찾을 수 없습니다: {ref_path}")
            
            result = detector.detect_with_reference(
                test_image_path=str(test_path),
                reference_image_path=str(ref_path),
                product_name=product_name,
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
            reference_normal_url=f"/api/image/{result.get('reference_image_path', '')}",
            mask_url=f"/anomaly/image/{test_path.stem}/mask.png",
            overlay_url=f"/anomaly/image/{test_path.stem}/overlay.png",
            comparison_url=f"/anomaly/image/{test_path.stem}/comparison.png" if "comparison_path" in result else None
        )
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"이상 검출 실패: {str(e)}")

# ====================
# 불량 이미지 등록
# ====================
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
    product_name: str = Form(...),
    defect_name: str = Form(...)
):
    """
    불량 이미지 등록
    
    Args:
        file: 불량 이미지 파일
        product_name: 제품명 (예: prod1)
        defect_name: 불량명 (예: hole, burr)
    
    Returns:
        {
            "status": "success",
            "saved_path": "경로",
            "filename": "파일명",
            "seqno": 번호
        }
    """
    # 저장 경로 설정
    defect_dir = project_root / "data" / "def_split"
    defect_dir.mkdir(parents=True, exist_ok=True)
    
    # 다음 seqno 계산
    next_seqno = get_next_seqno(defect_dir, product_name, defect_name)
    
    # 파일명 생성: {product}_{defect}_{seqno:03d}.{ext}
    ext = Path(file.filename).suffix
    new_filename = f"{product_name}_{defect_name}_{next_seqno:03d}{ext}"
    save_path = defect_dir / new_filename
    
    # 저장
    with save_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # 인덱스 재구축
    index_rebuilt = False
    if matcher and matcher.index_built:
        try:
            matcher.build_index(str(defect_dir))
            matcher.save_index(str(INDEX_DIR))
            index_rebuilt = True
        except Exception as e:
            print(f"인덱스 재구축 실패: {e}")
    
    return JSONResponse(content={
        "status": "success",
        "saved_path": str(save_path),
        "filename": new_filename,
        "product_name": product_name,
        "defect_name": defect_name,
        "seqno": next_seqno,
        "index_rebuilt": index_rebuilt
    })

# ====================
# LLM 대응 매뉴얼 생성 (핵심 로직)
# ====================
async def _manual_core(mode: str, req: ManualGenRequest):
    """
    매뉴얼 생성 공용 코어 함수
    
    Args:
        mode: 'llm' (텍스트 기반) 또는 'vlm' (이미지 포함)
        req: ManualGenRequest
    
    Returns:
        {
            "status": "success",
            "product": "prod1",
            "defect_en": "hole",
            "anomaly_score": 0.XXXX,
            "is_anomaly": true/false,
            "manual": {"원인": [...], "조치": [...]},
            "llm_analysis": "..." (mode='llm')
            "vlm_analysis": "..." (mode='vlm')
        }
    """
    t0 = time.time()

    # ========================================
    # 0) 제품/불량 추출
    # ========================================
    product = req.product_name
    defect = req.defect_name
    
    # 파일명에서 추출 시도 (형식: {product}_{defect}_{seq}.jpg)
    if not product or not defect:
        name = (req.top1_image_path or '').split('/')[-1]
        parts = name.split('_')
        if not product and len(parts) >= 1:
            product = parts[0]
        if not defect and len(parts) >= 2:
            defect = parts[1]

    if not product or not defect:
        raise HTTPException(
            400,
            "product/defect 파악 실패: product_name, defect_name를 제공하거나 TOP-1 파일명 규칙({product}_{defect}_XX.jpg)을 확인하세요."
        )

    # ========================================
    # 1) PatchCore 이상 검출
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
            
            # PatchCore 실행
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
    # 2) DefectMapper + RAG 매뉴얼 검색
    # ========================================
    mapper = vlm_components["mapper"]
    rag = vlm_components["rag"]

    if not mapper:
        raise HTTPException(503, "DefectMapper가 초기화되지 않았습니다")

    defect_info = mapper.get_defect_info(product, defect)
    if not defect_info:
        raise HTTPException(404, f"불량 정보를 찾을 수 없습니다: {product}/{defect}")

    manual_ctx = {"원인": [], "조치": []}
    if rag:
        keywords = mapper.get_search_keywords(product, defect)
        manual_ctx = rag.search_defect_manual(product, defect, keywords)
        
        if req.verbose:
            print(f"✅ RAG 검색 완료: 원인 {len(manual_ctx['원인'])}개, 조치 {len(manual_ctx['조치'])}개")
    else:
        print("⚠️ RAG 미초기화 상태")

    # ========================================
    # 3) LLM/VLM 호출
    # ========================================
    llm_analysis = None
    vlm_analysis = None

    async with httpx.AsyncClient(timeout=60) as client:
        if mode == "llm":
            # 텍스트 기반 LLM 분석
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
                "model_provider": "hyperclovax"  # ✅ 추가: 기본값 또는 req에서 받기
            }
            
            r = await client.post(f"{LLM_SERVER_URL}/analyze", json=payload)
            r.raise_for_status()
            llm_analysis = r.json().get("analysis", "")

        elif mode == "vlm":
            # 이미지 포함 VLM 분석
            prompt = (
                f"[제품] {product}\n"
                f"[불량] {defect_info.ko} ({defect_info.en})\n"
                f"[정식명칭] {defect_info.full_name_ko}\n"
                f"[이상점수] {anomaly_score:.4f}\n"
                f"[판정] {'불량' if is_anomaly else '정상'}\n\n"
                "아래 매뉴얼을 1차 근거로 사용하여 이미지에서 보이는 불량을 분석하세요.\n"
                f"원인(매뉴얼): {manual_ctx.get('원인', [])}\n"
                f"조치(매뉴얼): {manual_ctx.get('조치', [])}\n"
                "매뉴얼 문장을 따옴표로 인용하고, 불확실한 추정은 금지합니다."
            )
            
            r = await client.post(f"{LLM_SERVER_URL}/analyze_vlm", json={
                "image_path": req.image_path,
                "prompt": prompt,
                "max_new_tokens": min(256, req.max_new_tokens),
                "temperature": min(0.3, req.temperature)
            })
            r.raise_for_status()
            vlm_analysis = r.json().get("analysis", "")

        else:
            raise HTTPException(400, f"지원하지 않는 mode: {mode}")

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
        "anomaly_score": float(anomaly_score),
        "is_anomaly": bool(is_anomaly),
        "processing_time": round(time.time() - t0, 2)
    }
    
    if llm_analysis is not None:
        out["llm_analysis"] = llm_analysis
    if vlm_analysis is not None:
        out["vlm_analysis"] = vlm_analysis
    
    return out


@app.post("/manual/generate/llm")
async def manual_generate_llm(req: ManualGenRequest):
    """
    LLM 기반 대응 매뉴얼 생성 (텍스트만)
    
    Request Body:
        {
            "image_path": "업로드된 이미지 경로",
            "top1_image_path": "유사도 TOP-1 이미지 경로 (PatchCore 기준)",
            "product_name": "prod1" (선택, 파일명에서 추출 가능),
            "defect_name": "hole" (선택, 파일명에서 추출 가능),
            "verbose": true (디버그 로그)
        }
    
    Response:
        {
            "status": "success",
            "product": "prod1",
            "anomaly_score": 0.XXXX,
            "is_anomaly": true/false,
            "manual": {"원인": [...], "조치": [...]},
            "llm_analysis": "4개 섹션 분석 결과"
        }
    """
    try:
        return await _manual_core("llm", req)
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"LLM 생성 오류: {str(e)}")


@app.post("/manual/generate/vlm")
async def manual_generate_vlm(req: ManualGenRequest):
    """
    VLM 기반 대응 매뉴얼 생성 (이미지 포함)
    
    Request Body:
        ManualGenRequest와 동일
    
    Response:
        {
            "status": "success",
            "vlm_analysis": "VLM 분석 결과"
        }
    """
    try:
        return await _manual_core("vlm", req)
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"VLM 생성 오류: {str(e)}")

# ====================
# 통합 파이프라인 (검색 + 이상검출 + 매뉴얼)
# ====================
@app.post("/generate_manual_advanced")
async def generate_manual_advanced(request: dict):
    """
    고급 불량 분석 (통합 파이프라인)
    
    워크플로우:
    1. CLIP 유사도 검색 → product/defect 추출
    2. PatchCore 이상 검출 (자동 정상 기준)
    3. RAG 매뉴얼 검색
    4. LLM 대응 방안 생성
    
    Request:
        {
            "image_path": "업로드된 이미지 경로"
        }
    
    Response:
        {
            "status": "success",
            "similarity": {...},
            "anomaly": {...},
            "manual": {...},
            "vlm_analysis": "..."
        }
    """
    start_time = time.time()
    
    try:
        image_path = request.get("image_path")
        if not image_path:
            raise HTTPException(400, "image_path 필수")
        
        # 경로 정규화
        image_path_obj = Path(image_path)
        if not image_path_obj.is_absolute():
            if image_path.startswith("./uploads/") or image_path.startswith("uploads/"):
                filename = image_path.replace("./uploads/", "").replace("uploads/", "")
                image_path_obj = uploads_dir / filename
            else:
                image_path_obj = project_root / image_path
        
        if not image_path_obj.exists():
            raise HTTPException(404, f"이미지를 찾을 수 없습니다: {image_path_obj}")

        result = {
            "status": "success",
            "steps": []
        }
        
        # Step 1: 유사도 검색
        print("\n[Step 1] 유사도 검색...")
        if not matcher or not matcher.index_built:
            raise HTTPException(503, "인덱스가 구축되지 않았습니다")
        
        search_result = matcher.search(str(image_path_obj), top_k=5)
        
        if not search_result.top_k_results:
            raise HTTPException(404, "유사한 이미지를 찾을 수 없습니다")
        
        # TOP-K 중에서 불량 이미지 찾기 (defect 포함)
        product = None
        defect = None
        top_result = None
        
        for result_item in search_result.top_k_results:
            filename = Path(result_item["image_path"]).stem
            parts = filename.split("_")
            
            if len(parts) >= 3:
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
        
        # Step 3: PatchCore 이상 검출
        print(f"\n[Step 3] 이상 영역 검출...")
        if not detector:
            raise HTTPException(503, "AnomalyDetector가 초기화되지 않았습니다")
        
        output_dir = ANOMALY_OUTPUT_DIR / image_path_obj.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        
        anomaly_result = detector.detect_with_normal_reference(
            test_image_path=str(image_path_obj),
            product_name=product,
            similarity_matcher=matcher,
            output_dir=str(output_dir)
        )
        
        print(f"✅ 이상 점수: {anomaly_result['image_score']:.4f}")
        
        result["anomaly"] = {
            "score": float(anomaly_result["image_score"]),
            "is_anomaly": anomaly_result["is_anomaly"],
            "normal_image_url": f"/api/image/{anomaly_result.get('reference_image_path', '')}",
            "overlay_image_url": f"/anomaly/image/{image_path_obj.stem}/overlay.png",
            "mask_image_url": f"/anomaly/image/{image_path_obj.stem}/mask.png"
        }
        
        # Step 4: RAG 매뉴얼 검색
        print(f"\n[Step 4] 매뉴얼 검색...")
        rag = vlm_components.get("rag")
        
        if rag is None:
            print("⚠️  RAG가 비활성화되어 있습니다")
            result["manual"] = {
                "원인": ["RAG 서비스가 비활성화되어 있습니다"],
                "조치": ["PDF 매뉴얼 파일을 추가하세요"]
            }
        else:
            keywords = mapper.get_search_keywords(product, defect)
            manual_context = rag.search_defect_manual(product, defect, keywords)
            result["manual"] = manual_context
            print(f"✅ 매뉴얼 검색 완료")
        
        # Step 5: LLM 분석
        print("[Step 5] LLM 분석...")
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                llm_payload = {
                    "product": product,
                    "defect_en": defect_info.en,
                    "defect_ko": defect_info.ko,
                    "full_name_ko": defect_info.full_name_ko,
                    "anomaly_score": float(result["anomaly"]["score"]),
                    "is_anomaly": bool(result["anomaly"]["is_anomaly"]),
                    "manual_context": result.get("manual", {})
                }
                
                r = await client.post(f"{LLM_SERVER_URL}/analyze", json=llm_payload)
                r.raise_for_status()
                llm_analysis = r.json().get("analysis", "")
                
            print(f"✅ LLM 분석 완료 ({len(llm_analysis)} 문자)")
            
        except Exception as e:
            print(f"⚠️  LLM 분석 실패: {e}")
            llm_analysis = f"LLM 분석 실패: {str(e)}"

        result["vlm_analysis"] = llm_analysis
        result["processing_time"] = round(time.time() - start_time, 2)
        
        print(f"\n✅ 분석 완료: {result['processing_time']}초\n")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"고급 분석 오류: {str(e)}")

# ====================
# 이미지 서빙
# ====================
@app.get("/api/image/{image_path:path}")
async def serve_image(image_path: str):
    """
    이미지 파일 제공
    
    Args:
        image_path: 상대 경로 (예: data/def_split/prod1_hole_001.jpg)
    """
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
            file_path = project_root / image_path
        
        if not file_path.exists():
            raise HTTPException(404, f"이미지를 찾을 수 없습니다: {file_path}")
        
        return FileResponse(str(file_path))
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"이미지 서빙 오류: {e}")
        raise HTTPException(500, str(e))


@app.get("/anomaly/image/{result_id}/{filename}")
async def serve_anomaly_image(result_id: str, filename: str):
    """
    이상 검출 결과 이미지 제공
    
    Args:
        result_id: 결과 ID (이미지 stem)
        filename: 파일명 (mask.png, overlay.png 등)
    """
    file_path = ANOMALY_OUTPUT_DIR / result_id / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="이미지를 찾을 수 없습니다")
    
    return FileResponse(file_path, media_type="image/png")

# ====================
# 서버 실행
# ====================
if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=5000,
        reload=False,  # 프로덕션에서는 False
        log_level="info"
    )
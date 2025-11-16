"""
메인 API 서버 - 라우터 통합
"""

from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import sys
import shutil
from typing import Optional
import uvicorn
from pydantic import BaseModel, Field
import torch
from contextlib import asynccontextmanager
import warnings
import os

import subprocess

# ✅ 불필요한 경고 숨기기
warnings.filterwarnings("ignore", category=RuntimeWarning, module="networkx")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"

# 프로젝트 루트 경로
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 기존 imports
from modules.similarity_matcher import TopKSimilarityMatcher, create_matcher
from modules.similarity_matcher_v2 import create_matcher_v2
from modules.anomaly_detector import AnomalyDetector, create_detector
from modules.vlm import RAGManager, DefectMapper, PromptBuilder


# ====================
# 전역 변수
# ====================
# 전역 변수 추가 (기존 변수들 아래에)


#matcher: Optional[TopKSimilarityMatcher] = None
matcher_defect: Optional[TopKSimilarityMatcher] = None  # 유사도 매칭용 (불량 인덱스)
matcher_normal: Optional[TopKSimilarityMatcher] = None  # 이상 검출용 (정상 인덱스)
matcher = None
detector: Optional[AnomalyDetector] = None
current_index_type: Optional[str] = None
config = None

vlm_components = {
    "rag": None,
    "vlm": None,
    "mapper": None,
    "prompt_builder": PromptBuilder()
}

# ====================
# FastAPI 앱 생성
# ====================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    ✅ 애플리케이션 생명주기 관리 (FastAPI 최신 방식)
    """
    # ========== 시작 시 ==========
    print("="*50)
    print("🚀 애플리케이션 시작")
    print("="*50)
    
    # DB 연결 테스트
    from web.database.connection import test_connection
    test_connection()
    
    """서버 시작 시 초기화"""
    global matcher,matcher_defect,matcher_normal, detector, current_index_type ,config

    # ==================== 설정 파일 로드 ====================
    config_file = project_root / "settings.config"
    config = {}
    
    if config_file.exists():
        print("\n" + "=" * 70)
        print("설정 파일 로드 중...")
        print("=" * 70)
        with open(config_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        config[key.strip()] = value.strip()
        
        print(f"인덱스 버전: {'V2 (DB 메타데이터)' if config.get('USE_INDEX_V2', 'false') == 'true' else 'V1 (파일명 파싱)'}")
        print(f"CLIP 모델: {config.get('CLIP_MODEL_ID', 'ViT-B-32')}/{config.get('CLIP_PRETRAINED', 'openai')}")
        print("=" * 70 + "\n")
    else:
        print(f"⚠️  설정 파일 없음: {config_file}")
        print("기본 설정 사용: V1 인덱스")
    
    USE_INDEX_V2 = config.get('USE_INDEX_V2', 'false') == 'true'
    CLIP_MODEL = config.get('CLIP_MODEL_ID', 'ViT-B-32')
    CLIP_PRETRAINED = config.get('CLIP_PRETRAINED', 'openai')
    CLIP_BATCH_SIZE = int(config.get('CLIP_BATCH_SIZE', '32'))
    CLIP_NUM_WORKERS = int(config.get('CLIP_NUM_WORKERS', '4'))
    CLIP_USE_FP16 = config.get('CLIP_USE_FP16', 'false') == 'true'

    update_tree_on_startup()
    
    print("=" * 60)
    print("유사도 매칭 + Anomaly Detection API 서버 시작")
    print("=" * 60)
    
    # ✅ 업로드 디렉토리 초기화 (임시 파일 삭제)
    print("\n[CLEANUP] 임시 업로드 파일 삭제 중...")
    try:
        deleted_count = 0
        for file_path in UPLOAD_DIR.glob("*"):
            if file_path.is_file():
                file_path.unlink()
                deleted_count += 1
        print(f"✅ {deleted_count}개 임시 파일 삭제 완료")
    except Exception as e:
        print(f"⚠️  임시 파일 삭제 실패: {e}")
    
    # ==================== 인덱스 버전 분기 ====================
    if USE_INDEX_V2:
        # ========== V2: DB 메타데이터 기반 인덱스 ==========
        print("\n" + "=" * 70)
        print("V2 인덱스 구축 (DB 메타데이터 기반)")
        print("=" * 70)
        
        from modules.similarity_matcher_v2 import create_matcher_v2
        from web.database.connection import get_db
        
       
        
        # V2 인덱스 디렉토리
         # V2 인덱스 디렉토리
        INDEX_DIR_V2 = WEB_DIR / "index_cache_v2"
        INDEX_DIR_V2.mkdir(parents=True, exist_ok=True)
        
        defect_index_v2 = INDEX_DIR_V2 / "defect"
        normal_index_v2 = INDEX_DIR_V2 / "normal"
        
        defect_index_v2.mkdir(parents=True, exist_ok=True)
        normal_index_v2.mkdir(parents=True, exist_ok=True)
        
        try:
            # DB 세션 생성
            db = next(get_db())
            
            # 1. 정상 이미지 인덱스 구축
            print(f"\n[1/2] 정상 이미지 인덱스 구축 (DB 기반)...")
             # V2 매처 생성
            matcher_normal = create_matcher_v2(
                model_id=f"{CLIP_MODEL}/{CLIP_PRETRAINED}",
                device="auto",
                use_fp16=CLIP_USE_FP16,
                batch_size=CLIP_BATCH_SIZE,
                num_workers=CLIP_NUM_WORKERS,
                verbose=True
            )
            try:
                info = matcher_normal.build_index_from_db(db, image_type='normal')
                matcher_normal.save_index(str(normal_index_v2))
                print(f"      ✅ 완료: {info['num_images']}개 이미지")
            except Exception as e:
                print(f"      ⚠️  실패: {e}")
                # 저장된 인덱스 로드 시도
                if (normal_index_v2 / "metadata.json").exists():
                    print(f"      → 저장된 인덱스 로드 시도...")
                    matcher_normal.load_index(str(normal_index_v2))
                    print(f"      ✅ 저장된 인덱스 로드 완료")
            
            # 2. 불량 이미지 인덱스 구축
            print(f"\n[2/2] 불량 이미지 인덱스 구축 (DB 기반)...")
            try:
                matcher_defect = create_matcher_v2(
                    model_id=f"{CLIP_MODEL}/{CLIP_PRETRAINED}",
                    device="auto",
                    use_fp16=CLIP_USE_FP16,
                    batch_size=CLIP_BATCH_SIZE,
                    num_workers=CLIP_NUM_WORKERS,
                    verbose=True
                )
                info = matcher_defect.build_index_from_db(db, image_type='defect')
                matcher_defect.save_index(str(defect_index_v2))
                print(f"      ✅ 완료: {info['num_images']}개 이미지")
            except Exception as e:
                print(f"      ⚠️  실패: {e}")
                # 저장된 인덱스 로드 시도
                if (defect_index_v2 / "metadata.json").exists():
                    print(f"      → 저장된 인덱스 로드 시도...")
                    matcher_defect.load_index(str(defect_index_v2))
                    print(f"      ✅ 저장된 인덱스 로드 완료")
            
            # 3. 기본 인덱스를 불량 이미지로 설정
            print("\n🔄 기본 인덱스 설정 (불량 이미지)...")
            if (defect_index_v2 / "metadata.json").exists():
                matcher_defect.load_index(str(defect_index_v2))
                current_index_type = "defect"
                print(f"✅ 불량 이미지 인덱스 로드 완료: {len(matcher_defect.gallery_metadata)}개")
            
        except Exception as e:
            print(f"\n❌ V2 인덱스 구축 실패: {e}")
            import traceback
            traceback.print_exc()
        
        # V2 라우터 초기화
        from routers.search_v2 import init_search_v2_router
        init_search_v2_router(matcher_defect, INDEX_DIR_V2, project_root)
        
        print("\n" + "=" * 70)
        print("V2 인덱스 구축 완료")
        print("=" * 70)


        # ✅ 디버깅: 전역 변수 확인
        print("\n[DEBUG] 전역 변수 확인:")
        print(f"  - matcher_defect is None: {matcher_defect is None}")
        print(f"  - matcher_normal is None: {matcher_normal is None}")
        if matcher_defect:
            print(f"  - matcher_defect.index_type: {matcher_defect.index_type}")
            print(f"  - matcher_defect 갤러리: {len(matcher_defect.gallery_metadata)}개")
            print(f"  - matcher_defect ID: {id(matcher_defect)}")
        if matcher_normal:
            print(f"  - matcher_normal.index_type: {matcher_normal.index_type}")
            print(f"  - matcher_normal 갤러리: {len(matcher_normal.gallery_metadata)}개")
            print(f"  - matcher_normal ID: {id(matcher_normal)}")
        print()
    
    else:
        # ========== V1: 파일명 파싱 기반 인덱스 (기존 코드) ==========
        print("\n" + "=" * 70)
        print("V1 인덱스 구축 (파일명 파싱 기반)")
        print("=" * 70)
        
        # V1 매처 생성
        matcher = create_matcher(
            model_id=f"{CLIP_MODEL}/{CLIP_PRETRAINED}",
            device="auto",
            use_fp16=CLIP_USE_FP16,
            batch_size=CLIP_BATCH_SIZE,
            num_workers=CLIP_NUM_WORKERS,
            verbose=True
        )
        
        # 불량 이미지 인덱스 구축
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
        
        # 정상 이미지 통합 인덱스 구축
        normal_base_dir = project_root / "data" / "patchCore"
        normal_index_path = INDEX_DIR / "normal"
        normal_index_path.mkdir(parents=True, exist_ok=True)
        
        if normal_base_dir.exists():
            try:
                print(f"\n[2/2] 정상 이미지 통합 인덱스 구축 중...")
                print(f"      기본 경로: {normal_base_dir}")
                
                # 모든 제품 폴더 탐색
                product_dirs = [d for d in normal_base_dir.iterdir() if d.is_dir()]
                
                if not product_dirs:
                    print(f"      ⚠️  제품 폴더를 찾을 수 없습니다")
                else:
                    print(f"      발견된 제품: {[d.name for d in product_dirs]}")
                    
                    # 통합 인덱스 구축
                    info = matcher.build_index(str(normal_base_dir))
                    matcher.save_index(str(normal_index_path))
                    
                    print(f"      ✅ 완료: {info['num_images']}개 이미지 (통합)")
                    
                    # 제품별 이미지 개수 표시
                    for prod_dir in product_dirs:
                        ok_dir = prod_dir / "ok"
                        if ok_dir.exists():
                            prod_images = list(ok_dir.glob("*.jpg")) + list(ok_dir.glob("*.png"))
                            print(f"         - {prod_dir.name}: {len(prod_images)}개")
                        
            except Exception as e:
                print(f"      ❌ 실패: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"\n[2/2] ⚠️  정상 이미지 기본 디렉토리 없음: {normal_base_dir}")
        
        # 기본 인덱스를 불량 이미지로 설정
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
        
        # V1 라우터 초기화
        from routers.search import init_search_router
        init_search_router(matcher, INDEX_DIR, project_root)
        
        print("\n" + "=" * 70)
        print("V1 인덱스 구축 완료")
        print("=" * 70)
    
    # ==================== 공통: Anomaly Detector & VLM ====================

    # 4. Anomaly Detector 생성
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
    
    # 5. VLM 컴포넌트 초기화 (기존 함수 있다면)
    init_vlm_components()
    print("✅ VLM Component 초기화 완료")

    # ✅ 6. 라우터 초기화 (매처를 전달)
    from routers.upload import init_upload_router
    from routers.search import init_search_router
    from routers.anomaly import init_anomaly_router
    from routers.manual import init_manual_router


    init_upload_router(UPLOAD_DIR)
    #init_anomaly_router(detector, matcher, ANOMALY_OUTPUT_DIR, project_root, INDEX_DIR)  # ✅ INDEX_DIR 추가
    if USE_INDEX_V2:
        init_anomaly_router(detector, matcher_normal, ANOMALY_OUTPUT_DIR, project_root, INDEX_DIR_V2)
        # ✅ 디버깅: 전달된 매처 확인
        print(f"[DEBUG] anomaly_router에 전달된 matcher:")
        print(f"  - index_type: {matcher_normal.index_type if matcher_normal else 'None'}")
        print(f"  - ID: {id(matcher_normal) if matcher_normal else 'None'}")
    else:
        init_anomaly_router(detector, matcher, ANOMALY_OUTPUT_DIR, project_root, INDEX_DIR)

    init_manual_router(
        vlm_components.get("mapper"),
        vlm_components.get("rag"),
        project_root,
        "http://localhost:5001"  # LLM 서버 URL
    )
    #init_search_router(matcher, INDEX_DIR, project_root)
    #init_search_v2_router(matcher_v2=similarity_matcher_v2, index_dir_v2=INDEX_DIR_V2,proj_root=project_root)

    if USE_INDEX_V2:
        from routers.search_v2 import router as search_v2_router
        app.include_router(search_v2_router)
        print("✅ V2 검색 라우터 등록 완료")
    
    
    print("\n" + "=" * 60)
    print("서버 초기화 완료")
    print("=" * 60 + "\n")
    
    yield  # ✅ 애플리케이션 실행 중

    # ========== 종료 시 ==========
    print("="*50)
    print("🛑 애플리케이션 종료 시작")
    print("="*50)
    
    # ✅ DB 연결 풀 정리
    from web.database.connection import dispose_engine
    dispose_engine()
    
    print("✅ 애플리케이션 종료 완료")


app = FastAPI(
    title="유사도 매칭 + Anomaly Detection API",
    description="CLIP 기반 이미지 유사도 검색 + PatchCore 이상 검출 서비스",
    version="3.0.0",
    lifespan = lifespan
)

# 디렉토리 설정
WEB_DIR = Path(__file__).parent
STATIC_DIR = WEB_DIR / "static"
PAGES_DIR = WEB_DIR / "pages"
UPLOAD_DIR = WEB_DIR / "uploads"
INDEX_DIR = WEB_DIR / "index_cache"

INDEX_DIR_V2 = Path(__file__).parent / "index_cache_v2"

ANOMALY_OUTPUT_DIR = WEB_DIR / "anomaly_outputs"



# 디렉토리 생성
STATIC_DIR.mkdir(exist_ok=True)
(STATIC_DIR / "css").mkdir(exist_ok=True)
(STATIC_DIR / "js").mkdir(exist_ok=True)
PAGES_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)
INDEX_DIR.mkdir(exist_ok=True)
ANOMALY_OUTPUT_DIR.mkdir(exist_ok=True)

# Static 파일 마운트
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)





def init_vlm_components():
    """VLM 컴포넌트 초기화 (서버 시작 시 1회)"""
    global vlm_components
    
    try:
        print("\n" + "="*70)
        print("VLM 컴포넌트 초기화 중...")
        print("="*70)
        
        # 경로 설정
        manual_dir = project_root / "manual_store"
        vector_store_path = project_root / "manual_store"
        mapping_file = project_root / "web" / "defect_mapping.json"
        
        # 매뉴얼 디렉토리 확인
        if not manual_dir.exists():
            print(f"⚠️  매뉴얼 디렉토리가 없습니다: {manual_dir}")
            manual_dir.mkdir(parents=True, exist_ok=True)
        
        # PDF 파일 확인
        pdf_files = list(manual_dir.glob("*.pdf"))
        
        if not pdf_files:
            print(f"⚠️  PDF 매뉴얼 파일이 없습니다: {manual_dir}")
            print("   RAG 기능이 비활성화됩니다.")
            vlm_components["rag"] = None
        else:
            print(f"\n발견된 매뉴얼 파일: {len(pdf_files)}개")
            for pdf in pdf_files:
                product_name = pdf.stem.split("_")[0]
                print(f"  - {pdf.name} (제품: {product_name})")
        
        # 매핑 파일이 없으면 생성
        if not mapping_file.exists():
            print("⚠️  매핑 파일이 없습니다. 기본 파일을 생성합니다...")
            from modules.vlm.defect_mapper import create_default_mapping
            create_default_mapping(mapping_file)
        
        # 1. DefectMapper 초기화
        print("\n1. DefectMapper 초기화...")
        vlm_components["mapper"] = DefectMapper(mapping_file)
        print("   ✅ DefectMapper 초기화 완료")
        
        # 2. UnifiedRAGManager 초기화
        print("\n2. UnifiedRAGManager 초기화...")
        if pdf_files:
            from modules.vlm.rag import create_rag_manager

            # RAG 매니저 초기화
            manual_dir = project_root / "manual_store"
            vector_store_path = manual_dir
            defect_mapping_path = project_root / "web" / "defect_mapping.json"  # 추가

            rag_manager = create_rag_manager(
                manual_dir=manual_dir,
                vector_store_path=vector_store_path,
                defect_mapping_path=defect_mapping_path,  # 추가
                device="cuda" if torch.cuda.is_available() else "cpu",
                force_rebuild=False,
                verbose=True
            )
            '''
            vlm_components["rag"] = create_rag_manager(
                manual_dir=manual_dir,
                vector_store_path=vector_store_path,
                device="cuda",
                force_rebuild=False,  # 기존 인덱스 사용
                verbose=True
            )
            '''
            vlm_components["rag"] = rag_manager

            # 사용 가능한 제품 출력
            #available_products = vlm_components["rag"].get_available_products()
            available_products = rag_manager.get_available_products()
            print(f"\n   사용 가능한 제품: {', '.join(available_products)}")
        else:
            print("   → PDF 없음: RAG 비활성화")
            vlm_components["rag"] = None
        
        print("\n" + "="*70)
        print("✅ VLM 컴포넌트 초기화 완료")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ VLM 초기화 오류: {e}")
        import traceback
        traceback.print_exc()
        
        # 오류 발생 시에도 서버는 계속 실행
        vlm_components["mapper"] = None
        vlm_components["rag"] = None


# ✅ 서버 시작 시 tree 갱신 함수
def update_tree_on_startup():
    """서버 시작 시 디렉토리 트리 갱신"""
    try:
        script_path = project_root / "save_tree.sh"
        
        if not script_path.exists():
            print(f"⚠️  save_tree.sh를 찾을 수 없습니다: {script_path}")
            return
        
        print("\n" + "="*60)
        print("📂 디렉토리 구조 갱신 중...")
        print("="*60)
        
        # 쉘 스크립트 실행
        result = subprocess.run(
            ["bash", str(script_path)],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=30  # 30초 타임아웃
        )
        
        if result.returncode == 0:
            print("✅ 디렉토리 구조 갱신 완료")
            print(result.stdout)
        else:
            print(f"⚠️  갱신 중 오류 발생:")
            print(result.stderr)
    
    except subprocess.TimeoutExpired:
        print("⚠️  tree 갱신 타임아웃 (30초 초과)")
    except Exception as e:
        print(f"⚠️  tree 갱신 실패: {e}")


# ====================
# 라우터 등록
# ====================

from routers.upload import router as upload_router
from routers.search import router as search_router
from routers.anomaly import router as anomaly_router
from routers.manual import router as manual_router
from routers.auth import router as auth_router

from routers.admin.product import router as product_router
from routers.admin.manual import router as admin_manual_router  # ✅ 이름 변경
from routers.admin.defect_type import router as defect_type_router
from routers.admin.image import router as image_router
from routers.admin.dashboard import router as dashboard_router
from routers.admin.deployment import router as deployment_router
from routers.admin.preprocessing import router as preprocessing_router
from routers.admin.model import router as model_router


from routers.search_v2 import router as search_v2_router, init_search_v2_router


app.include_router(auth_router)

# 라우터 등록
app.include_router(upload_router)
app.include_router(search_router)
app.include_router(anomaly_router)
app.include_router(manual_router)


# 기존 라우터 등록 부분 뒤에 추가
app.include_router(product_router)
app.include_router(admin_manual_router)
app.include_router(defect_type_router)
app.include_router(image_router)
app.include_router(dashboard_router)
app.include_router(deployment_router)
app.include_router(preprocessing_router)
app.include_router(model_router)




# ====================
# 기본 라우트 (페이지 서빙)
# ====================

@app.get("/")
async def root():
    """루트 접근 시 업로드 페이지로"""
    #return FileResponse(PAGES_DIR / "upload.html")
    return FileResponse(PAGES_DIR / "login.html")

# 페이지 서빙
@app.get("/login.html")
async def serve_login():
    return FileResponse(PAGES_DIR / "login.html")


@app.get("/upload.html")
async def serve_upload():
    """업로드 페이지"""
    return FileResponse(PAGES_DIR / "upload.html")


@app.get("/search.html")
async def serve_search():
    """검색 페이지"""
    html_path = PAGES_DIR / "search.html"
    if not html_path.exists():
        raise HTTPException(404, "검색 페이지가 아직 구현되지 않았습니다")
    return FileResponse(html_path)


@app.get("/anomaly.html")
async def serve_anomaly():
    """이상 검출 페이지"""
    html_path = PAGES_DIR / "anomaly.html"
    if not html_path.exists():
        raise HTTPException(404, "이상 검출 페이지가 아직 구현되지 않았습니다")
    return FileResponse(html_path)


@app.get("/manual.html")
async def serve_manual():
    """매뉴얼 페이지"""
    html_path = PAGES_DIR / "manual.html"
    if not html_path.exists():
        raise HTTPException(404, "매뉴얼 페이지가 아직 구현되지 않았습니다")
    return FileResponse(html_path)

# v2 테스트용 페이지 
@app.get("/search_v2_test.html")
async def serve_search_v2_test():
    """V2 유사도 검색 테스트 페이지"""
    html_path = PAGES_DIR / "search_v2_test.html"
    if not html_path.exists():
        raise HTTPException(404, "유사도 검색 V2 페이지가 아직 구현되지 않았습니다")
    return FileResponse(html_path)


# 4. 관리자 페이지 서빙 엔드포인트 추가

@app.get("/admin.html")
async def serve_admin_dashboard():
    return FileResponse(PAGES_DIR / "admin.html")
    #return FileResponse(PAGES_DIR / "dashboard.html")

@app.get("/admin/dashboard.html")
async def serve_admin_dashboard():
    return FileResponse(PAGES_DIR / "admin" / "dashboard.html")


@app.get("/admin/product.html")
async def serve_admin_product():
    return FileResponse(PAGES_DIR / "admin" / "admin_product.html")

@app.get("/admin/manual.html")
async def serve_admin_manual():
    return FileResponse(PAGES_DIR / "admin" / "admin_manual.html")

@app.get("/admin/defect-type.html")
async def serve_admin_defect_type():
    return FileResponse(PAGES_DIR / "admin" / "admin_defect_type.html")

@app.get("/admin/image-normal.html")
async def serve_admin_image_normal():
    return FileResponse(PAGES_DIR / "admin" / "admin_image_normal.html")

@app.get("/admin/image-defect.html")
async def serve_admin_image_defect():
    return FileResponse(PAGES_DIR / "admin" / "admin_image_defect.html")

@app.get("/admin/deploy-clip.html")
async def serve_admin_deploy_clip():
    return FileResponse(PAGES_DIR / "admin" / "admin_deploy_clip.html")

@app.get("/admin/deploy-patchcore.html")
async def serve_admin_deploy_patchcore():
    return FileResponse(PAGES_DIR / "admin" / "admin_deploy_patchcore.html")

@app.get("/admin/preprocessing.html")
async def serve_admin_preprocessing():
    return FileResponse(PAGES_DIR / "admin" / "admin_preprocessing.html")

# 페이지 서빙 추가
@app.get("/admin/model.html")
async def serve_admin_model():
    return FileResponse(PAGES_DIR / "admin" / "admin_model.html")

# ====================
# 기존 API 엔드포인트 유지 (하위 호환성)
# ====================

@app.get("/index/status")
async def get_index_status():
    """현재 인덱스 상태 조회"""
    if matcher_defect is None:
        return {
            "status": "error",
            "message": "매처가 초기화되지 않았습니다",
            "current_index_type": None,
            "gallery_count": 0
        }
    
    # V2 체크
    is_v2 = hasattr(matcher_defect, 'gallery_metadata')
    
    if is_v2:
        gallery_count = len(matcher_defect.gallery_metadata) if matcher_defect.gallery_metadata else 0
    else:
        gallery_count = len(matcher.gallery_paths) if matcher.gallery_paths else 0
    
    return {
        "status": "success",
        "current_index_type": current_index_type,
        "gallery_count": gallery_count,
        "index_built": matcher_defect.index_built,
        "model_id": matcher_defect.model_id if hasattr(matcher_defect, 'model_id') else None,
        "version": "v2" if is_v2 else "v1"
    }


@app.post("/build_index")
async def build_index(request: dict):
    """인덱스 재구축"""
    
    class BuildIndexRequest(BaseModel):
        gallery_dir: str = Field(..., description="갤러리 디렉토리")
        save_index: bool = Field(True, description="인덱스 저장 여부")
    
    if matcher is None:
        raise HTTPException(500, "매처가 초기화되지 않았습니다")
    
    req = BuildIndexRequest(**request)
    gallery_dir = Path(req.gallery_dir)
    
    if not gallery_dir.exists():
        raise HTTPException(404, f"디렉토리를 찾을 수 없습니다: {gallery_dir}")
    
    # V2 체크
    is_v2 = hasattr(matcher, 'gallery_metadata')
    
    try:
        if is_v2:
            # V2는 디렉토리 스캔이 아닌 DB 기반이므로 지원하지 않음
            raise HTTPException(
                400, 
                "V2 인덱스는 DB 기반입니다. "
                "관리자 페이지에서 인덱스 재구축을 사용하세요."
            )
        else:
            # V1 인덱스 구축
            info = matcher.build_index(str(gallery_dir))
            
            if req.save_index:
                # 디렉토리명으로 인덱스 타입 판단
                if "def_split" in str(gallery_dir) or "defect" in str(gallery_dir):
                    save_dir = INDEX_DIR / "defect"
                else:
                    save_dir = INDEX_DIR / "normal"
                
                save_dir.mkdir(parents=True, exist_ok=True)
                matcher.save_index(str(save_dir))
                info["index_saved"] = True
                info["index_save_path"] = str(save_dir)
            
            return info
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"인덱스 구축 실패: {str(e)}")


@app.get("/api/image/{image_path:path}")
async def serve_image(image_path: str):
    """이미지 파일 제공 엔드포인트"""
    try:
        # 상대 경로 정규화
        if image_path.startswith("../"):
            image_path = image_path.replace("../", "")
        
        # 경로 처리
        if image_path.startswith("uploads/"):
            file_path = UPLOAD_DIR / image_path.replace("uploads/", "")
        elif image_path.startswith("data/"):
            file_path = project_root / image_path
        else:
            # 기본적으로 project_root 기준
            file_path = project_root / image_path
        
        if not file_path.exists():
            raise HTTPException(404, f"이미지를 찾을 수 없습니다: {file_path}")
        
        return FileResponse(str(file_path))
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[IMAGE] 이미지 서빙 오류: {e}")
        raise HTTPException(500, str(e))


@app.post("/register_defect")
async def register_defect(
    file: UploadFile = File(...),
    product_name: str = Form(...),
    defect_name: str = Form(...)
):
    """불량 이미지 등록"""
    defect_dir = project_root / "data" / "def_split"
    defect_dir.mkdir(parents=True, exist_ok=True)
    
    # 현재 등록된 파일 중 최대 seqno 찾기
    existing_files = list(defect_dir.glob(f"{product_name}_{defect_name}_*"))
    
    max_seqno = 0
    for existing_file in existing_files:
        try:
            stem = existing_file.stem
            parts = stem.split('_')
            if len(parts) >= 3:
                seqno = int(parts[-1])
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
    
    # 인덱스 재구축
    index_rebuilt = False
    if matcher and matcher.index_built:
        try:
            defect_index_path = INDEX_DIR / "defect"
            matcher.build_index(str(defect_dir))
            matcher.save_index(str(defect_index_path))
            index_rebuilt = True
        except Exception as e:
            print(f"[REGISTER] 인덱스 재구축 실패: {e}")
    
    return JSONResponse(content={
        "status": "success",
        "saved_path": str(save_path),
        "filename": new_filename,
        "product_name": product_name,
        "defect_name": defect_name,
        "seqno": new_seqno,
        "index_rebuilt": index_rebuilt
    })



##추후 관리자페이지에서 사용 예정

@app.get("/mapping/status")
async def get_mapping_status():
    """매핑 상태 조회"""
    if vlm_components.get("mapper") is None:
        return {
            "status": "disabled",
            "available_products": []
        }
    
    mapper = vlm_components["mapper"]
    
    products_info = {}
    for product in mapper.get_available_products():
        defects = mapper.get_available_defects(product)
        products_info[product] = {
            "defect_count": len(defects),
            "defects": defects
        }
    
    return {
        "status": "active",
        "products": products_info
    }


@app.post("/mapping/reload")
async def reload_mapping():
    """매핑 파일 재로드"""
    try:
        mapping_file = project_root / "web" / "defect_mapping.json"
        
        if not mapping_file.exists():
            raise HTTPException(404, "매핑 파일이 없습니다")
        
        # 재초기화
        from modules.vlm.defect_mapper import DefectMapper
        vlm_components["mapper"] = DefectMapper(mapping_file)
        
        return {
            "status": "success",
            "message": "매핑 파일 재로드 완료",
            "available_products": vlm_components["mapper"].get_available_products()
        }
    
    except Exception as e:
        raise HTTPException(500, f"재로드 실패: {str(e)}")


@app.get("/health2")
async def health_check():
    """헬스체크 엔드포인트 (ALB 용)"""
    '''
    # V2 체크
    is_v2 = hasattr(matcher_defect, 'gallery_metadata') if matcher else False
    
    if matcher and matcher.index_built:
        if is_v2:
            gallery_size = len(matcher_defect.gallery_metadata) if matcher_defect.gallery_metadata else 0
        else:
            gallery_size = len(matcher.gallery_paths) if matcher.gallery_paths else 0
    else:
        gallery_size = 0
    
    return {
        "status": "healthy",
        "message": "API 서버가 정상 작동 중입니다",
        "index_built": matcher.index_built if matcher else False,
        "gallery_size": gallery_size,
        "matcher_initialized": matcher is not None,
        "detector_initialized": detector is not None,
        "matcher_version": "v2" if is_v2 else "v1"
    }
    '''
    return {
        "status": "healthy",
        "message": "API 서버가 정상 작동 중입니다",

    }

# ====================
# 서버 실행
# ====================

if __name__ == "__main__":

    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=5000,
        reload=True,
        log_level="info"
    )
# llm_server.py
"""
LLM 전용 API 서버
별도 가상환경(venv_llm)에서 실행
tokenizers 0.15.2 사용
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig , AutoProcessor
import uvicorn
import os

app = FastAPI(title="LLM Server", version="1.0.0")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 모델
model = None
tokenizer = None
model_name = None

vlm_model = None
vlm_processor = None
vlm_name = None


class AnalysisRequest(BaseModel):
    """불량 분석 요청"""
    product: str
    defect_en: str
    defect_ko: str
    full_name_ko: str
    anomaly_score: float
    is_anomaly: bool
    manual_context: Dict[str, List[str]]
    max_new_tokens: int = 512
    temperature: float = 0.7


class AnalysisResponse(BaseModel):
    """불량 분석 응답"""
    status: str
    analysis: str
    model: str


@app.on_event("startup")
async def load_model():
    """서버 시작 시 모델 로드"""
    global model, tokenizer, model_name
    
    # 환경변수 또는 기본값
    model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"
    
    print("=" * 60)
    print(f"🤖 LLM 서버 시작 중...")
    print(f"📦 모델: {model_name}")
    print("=" * 60)
    
    try:
        # 양자화 설정
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        print("🔄 토크나이저 로드 중...")
        try :
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_fast=True,
                local_files_only=False,
                force_download=True,   # 캐시가 이상하면 새로 받기
            )
            print("✅ 토크나이저 로드 완료")
        except Exception as e :
            print(f"[WARN] Fast tokenizer failed: {e}\n--> Falling back to slow tokenizer.")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_fast=False,
                local_files_only=False,
                force_download=True,
            )
            print("✅ 토크나이저 로드 완료 (slow)")
        
        print("🔄 모델 로드 중 (4-bit 양자화)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            trust_remote_code=True,
            device_map="auto"
        )
        print("✅ 모델 로드 완료")
        
        print("=" * 60)
        print("✅ LLM 서버 준비 완료")
        print(f"🌐 포트: 5001")
        print("=" * 60)
        
        try:
            vlm_name = os.getenv("VLM_MODEL", "llava-hf/llava-1.5-7b-hf")  # 예시
            print(f"🔄 VLM 로드 시도: {vlm_name}")
            global vlm_model, vlm_processor
            vlm_processor = AutoProcessor.from_pretrained(vlm_name, trust_remote_code=True)
            vlm_model = AutoModelForCausalLM.from_pretrained(
                vlm_name, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
            )
            print("✅ VLM 로드 완료")
        except Exception as e:
            print(f"⚠️ VLM 로드 건너뜀: {e}")
            vlm_model = None
            vlm_processor = None
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        raise


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(request: AnalysisRequest):
    """불량 분석 수행"""
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다")
    
    try:
        # 프롬프트 생성
        prompt = _build_prompt(request)
        
        # HyperCLOVA-X 형식으로 생성
        chat = [
            {"role": "tool_list", "content": ""},
            {"role": "system", "content": "당신은 제조업 품질 관리 전문가입니다."},
            {"role": "user", "content": prompt}
        ]
        
        inputs = tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)
        
        # 생성
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_length=request.max_new_tokens + inputs["input_ids"].shape[1],
                stop_strings=["<|endofturn|>", "<|stop|>"],
                tokenizer=tokenizer,
                temperature=request.temperature,
                do_sample=True if request.temperature > 0 else False
            )
        
        # 디코딩
        generated_text = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        return AnalysisResponse(
            status="success",
            analysis=generated_text.strip(),
            model=model_name
        )
        
    except Exception as e:
        print(f"❌ 분석 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class VLMAnalysisRequest(BaseModel):
    image_path: str
    prompt: str
    max_new_tokens: int = 256
    temperature: float = 0.2

@app.post("/analyze_vlm")
async def analyze_vlm(req: VLMAnalysisRequest):
    if vlm_model is None or vlm_processor is None:
        raise HTTPException(503, detail="VLM not loaded")
    try:
        from PIL import Image
        img = Image.open(req.image_path).convert("RGB")
        inputs = vlm_processor(images=img, text=req.prompt, return_tensors="pt").to(vlm_model.device)
        with torch.no_grad():
            out_ids = vlm_model.generate(
                **inputs, max_new_tokens=req.max_new_tokens,
                temperature=req.temperature, do_sample=req.temperature > 0
            )
        text = vlm_processor.batch_decode(out_ids, skip_special_tokens=True)[0]
        return {"status": "success", "analysis": text, "model": vlm_name}
    except Exception as e:
        raise HTTPException(500, detail=str(e))

def _build_prompt(request: AnalysisRequest) -> str:
    causes = "\n".join([f"- {c}" for c in request.manual_context.get("원인", [])])
    actions = "\n".join([f"- {a}" for a in request.manual_context.get("조치", [])])

    manual_present = bool(causes.strip() or actions.strip())
    manual_block = f"""
    ### 발생 원인(매뉴얼 발췌)
    {causes if causes else "매뉴얼 정보 없음"}

    ### 조치 가이드(매뉴얼 발췌)
    {actions if actions else "매뉴얼 정보 없음"}
    """

    policy = (
      "반드시 위의 '매뉴얼 발췌'를 1차 근거로 사용하고, 다른 추정은 금지하세요."
      if manual_present else
      "매뉴얼 정보가 없으므로 합리적 가정을 명시적으로 표기해 제시하세요."
    )

    prompt = f"""당신은 제조업 품질 전문가입니다. 아래 불량 정보를 분석하세요.

    ## 불량 정보
    - 제품: {request.product}
    - 불량 유형: {request.defect_ko} ({request.defect_en})
    - 정식 명칭: {request.full_name_ko}
    - 이상 검출 점수: {request.anomaly_score:.4f}
    - 불량 판정: {"불량" if request.is_anomaly else "정상"}

    {manual_block}

    ## 작성 규칙
    1) {policy}
    2) 매뉴얼 근거 문장을 "따옴표"로 인용하고, 항목별로 매핑해 주세요.
    3) [불량 현황 요약] → [원인 분석] → [대응 방안] → [예방 조치] 순으로 작성.

    ## 출력:
    - 불릿/번호 목록 위주로 간결하게.
    """
    return prompt


@app.get("/health")
async def health():
    """헬스체크"""
    return {
        "status": "healthy",
        "model": model_name,
        "model_loaded": model is not None
    }


@app.get("/")
async def root():
    """루트"""
    return {
        "service": "LLM Server",
        "model": model_name,
        "endpoints": ["/analyze", "/health"]
    }


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5001,
        log_level="info"
    )
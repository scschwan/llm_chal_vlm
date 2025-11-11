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
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import uvicorn

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


def _build_prompt(request: AnalysisRequest) -> str:
    """분석 프롬프트 생성"""
    
    causes = "\n".join([f"- {c}" for c in request.manual_context.get("원인", [])])
    actions = "\n".join([f"- {a}" for a in request.manual_context.get("조치", [])])
    
    prompt = f"""당신은 제조업 품질 관리 전문가입니다. 다음 불량 정보를 분석하세요.

## 불량 정보
- 제품: {request.product}
- 불량 유형: {request.defect_ko} ({request.defect_en})
- 정식 명칭: {request.full_name_ko}
- 이상 검출 점수: {request.anomaly_score:.4f}
- 불량 판정: {"불량" if request.is_anomaly else "정상"}

## 매뉴얼 참조

### 발생 원인
{causes if causes else "매뉴얼 정보 없음"}

### 조치 가이드
{actions if actions else "매뉴얼 정보 없음"}

## 작성 요청
다음 내용을 포함한 분석 보고서를 작성하세요:

1. **불량 현황 요약**: 검출된 불량의 특징과 심각도
2. **원인 분석**: 매뉴얼을 참고한 발생 원인
3. **대응 방안**: 구체적이고 실행 가능한 조치 방법
4. **예방 조치**: 재발 방지 권장사항

현장에서 즉시 활용 가능하도록 구체적으로 작성하세요.
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
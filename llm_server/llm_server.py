# llm_server.py  (LLM + VLM 동시 지원 버전)

import os
import time
from typing import Dict, List, Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    LlavaForConditionalGeneration,
)

# =========================
# FastAPI
# =========================
app = FastAPI(title="LLM/VLM Server", version="1.0")

# =========================
# 전역 모델 핸들 (LLM)
# =========================
llm_name: Optional[str] = None
llm_model: Optional[AutoModelForCausalLM] = None
llm_tokenizer: Optional[AutoTokenizer] = None

# =========================
# 전역 모델 핸들 (VLM - LLaVA)
# =========================
vlm_name: Optional[str] = None
vlm_model: Optional[LlavaForConditionalGeneration] = None
vlm_processor: Optional[AutoProcessor] = None

# =========================
# 요청/응답 스키마
# =========================
class AnalysisRequest(BaseModel):
    product: str
    defect_en: str
    defect_ko: str
    full_name_ko: str
    anomaly_score: float = 0.0
    is_anomaly: bool = False
    manual_context: Dict[str, List[str]] = {}
    max_new_tokens: int = 512
    temperature: float = 0.7

class VLMAnalysisRequest(BaseModel):
    image_path: str
    prompt: str
    max_new_tokens: int = 256
    temperature: float = 0.2

# =========================
# 유틸: 프롬프트 빌더(LLM)
# =========================
def _build_prompt(req: AnalysisRequest) -> str:
    causes = "\n".join([f"- {c}" for c in req.manual_context.get("원인", [])])
    actions = "\n".join([f"- {a}" for a in req.manual_context.get("조치", [])])

     # 디버깅 로그
    print(f"[DEBUG] 매뉴얼 컨텍스트 수신:")
    print(f"  원인 개수: {len(req.manual_context.get('원인', []))}")
    print(f"  조치 개수: {len(req.manual_context.get('조치', []))}")
    
    if not causes.strip() and not actions.strip():
        print("⚠️  매뉴얼 정보가 비어있습니다!")
        
    manual_present = bool(causes.strip() or actions.strip())
    manual_block = f"""
### 발생 원인(매뉴얼 발췌)
{causes if causes else "매뉴얼 정보 없음"}

### 조치 가이드(매뉴얼 발췌)
{actions if actions else "매뉴얼 정보 없음"}
""".strip()

    policy = (
        "반드시 위의 '매뉴얼 발췌'를 1차 근거로 사용하고, 다른 추정은 금지하세요."
        if manual_present else
        "매뉴얼 정보가 없으므로 합리적 가정을 명시적으로 표기해 제시하세요."
    )

    prompt = f"""당신은 제조업 품질 전문가입니다. 아래 불량 정보를 분석하세요.

## 불량 정보
- 제품: {req.product}
- 불량 유형: {req.defect_ko} ({req.defect_en})
- 정식 명칭: {req.full_name_ko}
- 이상 검출 점수: {req.anomaly_score:.4f}
- 불량 판정: {"불량" if req.is_anomaly else "정상"}

{manual_block}

## 작성 규칙
1) {policy}
2) 매뉴얼 근거 문장을 "따옴표"로 인용하고, 항목별로 매핑해 주세요.
3) [불량 현황 요약] → [원인 분석] → [대응 방안] → [예방 조치] 순으로 작성.

## 출력:
- 불릿/번호 목록 위주로 간결하게.
""".strip()
    return prompt

# =========================
# 모델 로더
# =========================
@app.on_event("startup")
async def load_models_on_startup():
    global llm_name, llm_model, llm_tokenizer
    global vlm_name, vlm_model, vlm_processor

    # ---- LLM ----
    try:
        llm_name = os.getenv("LLM_MODEL", "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B")
        print(f"🔄 LLM 로드 시도: {llm_name}")

        # 토크나이저: fast 우선, 실패 시 slow
        try:
            llm_tokenizer = AutoTokenizer.from_pretrained(
                llm_name, use_fast=True, trust_remote_code=True
            )
            print("✅ LLM 토크나이저 로드 완료 (fast)")
        except Exception as e:
            print(f"[WARN] LLM fast tokenizer 실패: {e} → slow 재시도")
            llm_tokenizer = AutoTokenizer.from_pretrained(
                llm_name, use_fast=False, trust_remote_code=True
            )
            print("✅ LLM 토크나이저 로드 완료 (slow)")

        llm_model = AutoModelForCausalLM.from_pretrained(
            llm_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        print("✅ LLM 로드 완료")
    except Exception as e:
        print(f"⚠️ LLM 로드 실패: {e}")
        llm_name = None
        llm_model = None
        llm_tokenizer = None

    # ---- VLM (LLaVA) ----
    try:
        vlm_name = os.getenv("VLM_MODEL", "llava-hf/llava-1.5-7b-hf")
        print(f"🔄 VLM 로드 시도: {vlm_name}")

        vlm_model = LlavaForConditionalGeneration.from_pretrained(
            vlm_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        vlm_processor = AutoProcessor.from_pretrained(vlm_name)
        print("✅ VLM 로드 완료")
    except Exception as e:
        print(f"⚠️ VLM 로드 건너뜀: {e}")
        vlm_name = None
        vlm_model = None
        vlm_processor = None

# =========================
# 루트/헬스
# =========================
@app.get("/")
def root():
    return {
        "service": "LLM/VLM Server",
        "models": {
            "llm": llm_name,
            "vlm": vlm_name,
        },
        "endpoints": ["/analyze", "/analyze_vlm", "/health"],
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "llm": {"name": llm_name, "loaded": llm_model is not None},
        "vlm": {"name": vlm_name, "loaded": vlm_model is not None},
    }

# =========================
# LLM 분석
# =========================
@app.post("/analyze")
def analyze(req: AnalysisRequest):
    if llm_model is None or llm_tokenizer is None:
        raise HTTPException(503, detail="LLM not loaded")

    prompt = _build_prompt(req)

    # device 추출 (device_map="auto"일 때도 첫 파라미터의 device 사용)
    try:
        device = next(llm_model.parameters()).device
    except StopIteration:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inputs = llm_tokenizer(prompt, return_tensors="pt").to(device)

    do_sample = (req.temperature or 0) > 0
    gen_kwargs = dict(
        max_new_tokens=min(max(req.max_new_tokens, 16), 2048),
        temperature=float(max(min(req.temperature, 1.5), 0.0)),
        do_sample=do_sample,
    )
    if do_sample:
        gen_kwargs.update(dict(top_p=0.9))

    with torch.no_grad():
        output_ids = llm_model.generate(**inputs, **gen_kwargs)

    text = llm_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return {
        "status": "success",
        "analysis": text,
        "model": llm_name,
        "used_temperature": gen_kwargs["temperature"],
        "max_new_tokens": gen_kwargs["max_new_tokens"],
    }

# =========================
# VLM 분석 (LLaVA)
# =========================
@app.post("/analyze_vlm")
def analyze_vlm(req: VLMAnalysisRequest):
    if vlm_model is None or vlm_processor is None:
        raise HTTPException(503, detail="VLM not loaded")
    if not os.path.exists(req.image_path):
        raise HTTPException(400, detail=f"image_path not found: {req.image_path}")

    try:
        img = Image.open(req.image_path).convert("RGB")

        # 신형 Processor: 채팅 템플릿(멀티모달) 지원
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": req.prompt.strip()},
                    ],
                }
            ]
            prompt_text = vlm_processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        except Exception:
            # 구형 호환: 텍스트 그대로
            prompt_text = req.prompt.strip()

        inputs = vlm_processor(images=img, text=prompt_text, return_tensors="pt").to(vlm_model.device)

        do_sample = (req.temperature or 0) > 0
        gen_kwargs = dict(
            max_new_tokens=min(max(req.max_new_tokens, 16), 1024),
            temperature=float(max(min(req.temperature, 1.5), 0.0)),
            do_sample=do_sample,
        )
        if do_sample:
            gen_kwargs.update(dict(top_p=0.9))

        with torch.no_grad():
            out = vlm_model.generate(**inputs, **gen_kwargs)

        text = vlm_processor.batch_decode(out, skip_special_tokens=True)[0]
        return {
            "status": "success",
            "analysis": text,
            "model": vlm_name,
            "used_temperature": gen_kwargs["temperature"],
            "max_new_tokens": gen_kwargs["max_new_tokens"],
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(500, detail=f"VLM inference error: {e}")

# =========================
# 서버 실행 (개발용)
# =========================
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "5001"))
    uvicorn.run("llm_server:app", host="0.0.0.0", port=port, reload=False)

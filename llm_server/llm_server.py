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
    """개선된 프롬프트 빌더"""
    
    # 매뉴얼 정리 (중복 제거 및 간결화)
    causes_list = req.manual_context.get("원인", [])
    actions_list = req.manual_context.get("조치", [])
    
    # 해당 불량만 필터링 (defect_en 기준)
    causes = []
    actions = []
    
    for cause_text in causes_list:
        # 현재 불량(defect_en)과 관련된 내용만 추출
        if req.defect_en.lower() in cause_text.lower() or req.defect_ko in cause_text:
            # 깔끔하게 정리
            lines = [line.strip() for line in cause_text.split('\n') 
                    if line.strip() and not line.strip().startswith(('burr', 'hole', 'scratch', 'Hole', 'burr', 'Scratch'))]
            causes.extend(lines[:3])  # 최대 3줄
    
    for action_text in actions_list:
        if req.defect_en.lower() in action_text.lower() or req.defect_ko in action_text:
            lines = [line.strip() for line in action_text.split('\n') 
                    if line.strip() and not line.strip().startswith(('burr', 'hole', 'scratch', 'Hole', 'burr', 'Scratch'))]
            actions.extend(lines[:3])  # 최대 3줄
    
    # 매뉴얼 정보 유무 확인
    has_manual = bool(causes or actions)
    
    # 이상 검출 판정 설명
    anomaly_status = "불량" if req.is_anomaly else "정상"
    score_interpretation = ""
    if req.anomaly_score > 0.5:
        score_interpretation = "(높은 이상 점수 - 명확한 불량)"
    elif req.anomaly_score > 0.1:
        score_interpretation = "(중간 이상 점수 - 경미한 불량)"
    elif req.is_anomaly:
        score_interpretation = "(낮은 이상 점수 - 경계선상)"
    else:
        score_interpretation = "(정상 범위 - 불량 미검출)"
    
    prompt = f"""당신은 제조업 품질 전문가입니다. 아래 불량 정보를 **간결하게** 분석하세요.

## 검사 정보
- 제품: {req.product}
- 불량 유형: {req.defect_ko} ({req.defect_en})
- 정식 명칭: {req.full_name_ko}
- 이상 검출 점수: {req.anomaly_score:.4f} {score_interpretation}
- 최종 판정: {anomaly_status}

## 매뉴얼 정보
"""
    
    if has_manual:
        if causes:
            prompt += "\n**발생 원인:**\n"
            for i, cause in enumerate(causes[:3], 1):
                prompt += f"{i}. {cause}\n"
        
        if actions:
            prompt += "\n**조치 가이드:**\n"
            for i, action in enumerate(actions[:3], 1):
                prompt += f"{i}. {action}\n"
    else:
        prompt += "- 매뉴얼 정보 없음 (일반적인 제조 지식 기반 분석 필요)\n"
    
    prompt += f"""
## 작성 지침
1. 위 매뉴얼 정보를 **직접 인용**하며 분석
2. 매뉴얼 문장은 "따옴표"로 표시
3. **4개 섹션만** 작성: [불량 현황] → [원인 분석] → [대응 방안] → [예방 조치]
4. 각 섹션은 **2-3줄로 간결하게**
5. 예시나 템플릿 문구 반복 금지

## 출력 형식
### 불량 현황 요약
- (2-3줄 요약)

### 원인 분석
- (매뉴얼 인용 + 분석)

### 대응 방안
- (즉시 조치사항 2-3개)

### 예방 조치
- (재발 방지 방안 2-3개)
"""
    
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
        repetition_penalty=1.2,  # ✅ 추가: 반복 방지
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

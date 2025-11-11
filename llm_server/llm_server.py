# llm_server.py - EXAONE 3.5 및 HyperCLOVAX 지원

import os
import time
from typing import Dict, List, Optional, Tuple
from enum import Enum
import uvicorn
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
app = FastAPI(title="LLM/VLM Server", version="2.0")

# =========================
# 모델 타입 정의
# =========================
class LLMProvider(str, Enum):
    HYPERCLOVAX = "hyperclovax"
    EXAONE = "exaone"



# =========================
# 전역 모델 핸들
# =========================

hyperclovax_model: Optional[AutoModelForCausalLM] = None
hyperclovax_tokenizer: Optional[AutoTokenizer] = None

exaone_model: Optional[AutoModelForCausalLM] = None
exaone_tokenizer: Optional[AutoTokenizer] = None

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
    model_provider: Optional[str] = None  # 'hyperclovax' 또는 'exaone'

class VLMAnalysisRequest(BaseModel):
    image_path: str
    prompt: str
    max_new_tokens: int = 256
    temperature: float = 0.2


# =========================
# 프롬프트 빌더
# =========================
def _build_prompt_text(req: AnalysisRequest) -> str:
    """프롬프트 텍스트 생성 (공통)"""
    causes = req.manual_context.get("원인", [])
    actions = req.manual_context.get("조치", [])
    has_manual = bool(causes or actions)
    
    if req.is_anomaly:
        status = f"불량 검출 (이상점수: {req.anomaly_score:.4f})"
    else:
        status = f"정상 범위 (이상점수: {req.anomaly_score:.4f})"
    
    prompt = f"""당신은 제조 품질 전문가입니다. 아래 정보를 바탕으로 간결한 보고서를 작성하세요.

【검사 결과】
제품: {req.product}
불량: {req.defect_ko} ({req.defect_en})
판정: {status}

【매뉴얼】
"""
    
    if has_manual:
        if causes:
            prompt += "발생 원인:\n"
            for i, cause in enumerate(causes, 1):
                prompt += f"{i}. {cause}\n"
        
        if actions:
            prompt += "\n조치 방법:\n"
            for i, action in enumerate(actions, 1):
                prompt += f"{i}. {action}\n"
    else:
        prompt += "※ 매뉴얼 정보 없음\n"
    
    prompt += """
【지침】
- 위 매뉴얼 내용을 직접 인용 (따옴표 사용)
- 4개 섹션만 작성 (각 2-3문장)
- 추측이나 예시 반복 금지

【출력 형식】
### 불량 현황
(판정 결과 요약)

### 원인 분석  
(매뉴얼 원인 인용)

### 대응 방안
(즉시 조치 2-3개)

### 예방 조치
(재발 방지 2-3개)

위 4개 섹션만 작성하고 종료하세요. 추가 설명이나 예시 불필요.
"""
    return prompt

def _prepare_inputs_hyperclovax(prompt_text: str, tokenizer):
    """HyperCLOVAX용 입력 준비"""
    # 단순 텍스트 토크나이징
    return tokenizer(prompt_text, return_tensors="pt")

def _prepare_inputs_exaone(prompt_text: str, tokenizer):
    """EXAONE 3.5용 입력 준비 (chat template 사용)"""
    messages = [
        {
            "role": "system", 
            "content": "You are EXAONE model from LG AI Research, a helpful assistant specialized in manufacturing quality control."
        },
        {
            "role": "user",
            "content": prompt_text
        }
    ]
    
    # Chat template 적용
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    
    return {"input_ids": input_ids}

# =========================
# 모델 초기화 (수정)
# =========================
@app.on_event("startup")
async def load_models_on_startup():
    global hyperclovax_model, hyperclovax_tokenizer
    global exaone_model, exaone_tokenizer
    global vlm_name, vlm_model, vlm_processor
    
    print("=" * 60)
    print("LLM/VLM 서버 시작")
    print("=" * 60)
    
    # 1. HyperCLOVAX 로드
    print("\n[1/3] HyperCLOVAX 로드 중...")
    try:
        model_id = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"
        print(f"🔄 로드 시도: {model_id}")
        
        try:
            hyperclovax_tokenizer = AutoTokenizer.from_pretrained(
                model_id, use_fast=True, trust_remote_code=True
            )
        except:
            hyperclovax_tokenizer = AutoTokenizer.from_pretrained(
                model_id, use_fast=False, trust_remote_code=True
            )
        
        hyperclovax_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        print("✅ HyperCLOVAX 로드 완료")
    except Exception as e:
        print(f"❌ HyperCLOVAX 로드 실패: {e}")
    
    # 2. EXAONE 3.5 로드
    print("\n[2/3] EXAONE 3.5 로드 중...")
    try:
        model_id = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
        print(f"🔄 로드 시도: {model_id}")
        
        try:
            exaone_tokenizer = AutoTokenizer.from_pretrained(
                model_id, use_fast=True, trust_remote_code=True
            )
        except:
            exaone_tokenizer = AutoTokenizer.from_pretrained(
                model_id, use_fast=False, trust_remote_code=True
            )
        
        exaone_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        print("✅ EXAONE 3.5 로드 완료")
    except Exception as e:
        print(f"❌ EXAONE 로드 실패: {e}")
    
    # 3. VLM 로드
    print("\n[3/3] VLM 로드 중...")
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
        print(f"⚠️ VLM 로드 실패: {e}")
    
    print("\n" + "=" * 60)
    print("✅ 서버 초기화 완료")
    print(f"  - HyperCLOVAX: {'로드됨' if hyperclovax_model else '실패'}")
    print(f"  - EXAONE 3.5: {'로드됨' if exaone_model else '실패'}")
    print(f"  - VLM: {'로드됨' if vlm_model else '실패'}")
    print("=" * 60 + "\n")



# =========================
# LLM 분석 (수정)
# =========================
@app.post("/analyze")
def analyze(req: AnalysisRequest):
    """
    LLM 기반 분석
    
    Args:
        req.model_provider: 'hyperclovax' 또는 'exaone' (기본값: hyperclovax)
    """
    # 모델 선택
    provider = req.model_provider or LLMProvider.HYPERCLOVAX
    
    if provider == LLMProvider.EXAONE:
        if exaone_model is None or exaone_tokenizer is None:
            raise HTTPException(503, "EXAONE 모델이 로드되지 않았습니다")
        llm_model = exaone_model
        llm_tokenizer = exaone_tokenizer
    else:  # HYPERCLOVAX
        if hyperclovax_model is None or hyperclovax_tokenizer is None:
            raise HTTPException(503, "HyperCLOVAX 모델이 로드되지 않았습니다")
        llm_model = hyperclovax_model
        llm_tokenizer = hyperclovax_tokenizer
    
    # 프롬프트 텍스트 생성
    prompt_text = _build_prompt_text(req)
    
    # 모델별 입력 준비
    if provider == LLMProvider.EXAONE:
        inputs = _prepare_inputs_exaone(prompt_text, llm_tokenizer)
    else:  # HYPERCLOVAX
        inputs = _prepare_inputs_hyperclovax(prompt_text, llm_tokenizer)
    
    # GPU로 이동
    device = next(llm_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 생성 파라미터
    do_sample = (req.temperature or 0) > 0
    gen_kwargs = {
        "max_new_tokens": min(max(req.max_new_tokens, 16), 800),
        "temperature": float(max(min(req.temperature, 1.5), 0.0)),
        "do_sample": do_sample,
        "repetition_penalty": 1.3,
    }
    
    if do_sample:
        gen_kwargs["top_p"] = 0.9
    
    # EXAONE은 eos_token_id 명시
    if provider == LLMProvider.EXAONE:
        gen_kwargs["eos_token_id"] = llm_tokenizer.eos_token_id

    # 추론
    with torch.no_grad():
        output_ids = llm_model.generate(**inputs, **gen_kwargs)

    # 디코딩 (프롬프트 제외)
    if provider == LLMProvider.EXAONE:
        # EXAONE: 전체 출력 디코딩 후 파싱
        full_text = llm_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # ASSISTANT 응답 부분만 추출
        if "ASSISTANT:" in full_text:
            text = full_text.split("ASSISTANT:")[-1].strip()
        else:
            text = full_text
    else:
        # HyperCLOVAX: 프롬프트 제외
        generated_ids = output_ids[0][inputs['input_ids'].shape[1]:]
        text = llm_tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # 후처리
    text = text.split("assistant")[0].strip()
    text = text.split("[회사")[0].strip()
    
    # 예방 조치 이후 자르기
    lines = text.split('\n')
    prevention_idx = -1
    for i, line in enumerate(lines):
        if "예방 조치" in line or "예방조치" in line:
            prevention_idx = i
            break
    
    if prevention_idx > 0:
        text = '\n'.join(lines[:prevention_idx + 7])
    
    return {
        "status": "success",
        "analysis": text,
        "model": "EXAONE-3.5" if provider == LLMProvider.EXAONE else "HyperCLOVAX",
        "model_provider": provider,
        "used_temperature": gen_kwargs["temperature"],
        "max_new_tokens": gen_kwargs["max_new_tokens"],
    }

# =========================
# VLM 분석 (변경 없음)
# =========================
@app.post("/analyze_vlm")
def analyze_vlm(req: VLMAnalysisRequest):
    if vlm_model is None or vlm_processor is None:
        raise HTTPException(503, detail="VLM not loaded")
    if not os.path.exists(req.image_path):
        raise HTTPException(400, detail=f"image_path not found: {req.image_path}")

    try:
        img = Image.open(req.image_path).convert("RGB")

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
        import traceback
        traceback.print_exc()
        raise HTTPException(500, detail=f"VLM inference error: {e}")

# =========================
# 서버 실행
# =========================
if __name__ == "__main__":

    port = int(os.getenv("PORT", "5001"))
    uvicorn.run("llm_server:app", host="0.0.0.0", port=port, reload=False)
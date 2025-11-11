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
    """깔끔한 프롬프트 생성"""
    
    # 매뉴얼 정보 (이미 정리된 리스트)
    causes = req.manual_context.get("원인", [])
    actions = req.manual_context.get("조치", [])
    
    has_manual = bool(causes or actions)
    
    # 판정 상태
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
        #llm_name = os.getenv("LLM_MODEL", "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct")
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
# llm_server.py의 analyze 함수 수정
@app.post("/analyze")
def analyze(req: AnalysisRequest):
    print("\n" + "="*60)
    print("[LLM ANALYZE] 요청 시작")
    print("="*60)
    
    if llm_model is None or llm_tokenizer is None:
        print("[ERROR] LLM 모델이 로드되지 않음")
        raise HTTPException(503, detail="LLM not loaded")

    # 1. 프롬프트 생성
    prompt = _build_prompt(req)
    print(f"\n[PROMPT] 길이: {len(prompt)} 문자")
    print(f"[PROMPT] 첫 200자:\n{prompt[:200]}...")
    
    # 2. 토크나이징
    device = next(llm_model.parameters()).device
    print(f"\n[DEVICE] {device}")
    
    inputs = llm_tokenizer(prompt, return_tensors="pt").to(device)
    input_length = inputs['input_ids'].shape[1]
    print(f"[INPUT] 토큰 수: {input_length}")

    # 3. 생성 파라미터
    do_sample = (req.temperature or 0) > 0
    gen_kwargs = dict(
        max_new_tokens=min(max(req.max_new_tokens, 16), 800),
        temperature=float(max(min(req.temperature, 1.5), 0.0)),
        do_sample=do_sample,
        repetition_penalty=1.3,
    )
    if do_sample:
        gen_kwargs.update(dict(top_p=0.9))
    
    print(f"\n[GEN_KWARGS]")
    print(f"  - max_new_tokens: {gen_kwargs['max_new_tokens']}")
    print(f"  - temperature: {gen_kwargs['temperature']}")
    print(f"  - do_sample: {do_sample}")
    print(f"  - repetition_penalty: {gen_kwargs['repetition_penalty']}")

    # 4. 생성
    print(f"\n[GENERATE] 시작...")
    start_time = time.time()
    
    with torch.no_grad():
        output_ids = llm_model.generate(**inputs, **gen_kwargs)
    
    gen_time = time.time() - start_time
    output_length = output_ids.shape[1]
    print(f"[GENERATE] 완료 ({gen_time:.2f}초)")
    print(f"[OUTPUT] 전체 토큰 수: {output_length}")
    print(f"[OUTPUT] 생성된 토큰 수: {output_length - input_length}")

    # 5. 디코딩
    print(f"\n[DECODE] 시작...")
    generated_ids = output_ids[0][input_length:]
    text = llm_tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    print(f"[DECODE] 원본 길이: {len(text)} 문자")
    #print(f"[DECODE] 원본 첫 300자:\n{text[:300]}...")
    print(f"[DECODE] 원본 첫 300자:\n{text}...")
    
    # 6. 후처리
    print(f"\n[POST-PROCESS] 시작...")
    
    # assistant 분리
    if "assistant" in text:
        before_len = len(text)
        text = text.split("assistant")[0].strip()
        print(f"  - 'assistant' 분리: {before_len} -> {len(text)} 문자")
    
    # [회사 분리
    if "[회사" in text:
        before_len = len(text)
        text = text.split("[회사")[0].strip()
        print(f"  - '[회사' 분리: {before_len} -> {len(text)} 문자")
    
    # 예방 조치 이후 자르기
    lines = text.split('\n')
    prevention_idx = -1
    for i, line in enumerate(lines):
        if "예방 조치" in line or "예방조치" in line:
            prevention_idx = i
            print(f"  - '예방 조치' 발견: 라인 {i}")
            break
    
    if prevention_idx > 0:
        before_lines = len(lines)
        text = '\n'.join(lines[:prevention_idx + 7])
        print(f"  - 예방 조치 이후 자르기: {before_lines} -> {prevention_idx + 7} 라인")
    
    print(f"\n[FINAL] 최종 길이: {len(text)} 문자")
    print(f"[FINAL] 최종 라인 수: {len(text.split(chr(10)))}")
    print(f"[FINAL] 첫 500자:\n{text[:500]}...")
    
    print("\n" + "="*60)
    print("[LLM ANALYZE] 요청 완료")
    print("="*60 + "\n")
    
    return {
        "status": "success",
        "analysis": text,
        "model": llm_name,
        "used_temperature": gen_kwargs["temperature"],
        "max_new_tokens": gen_kwargs["max_new_tokens"],
        "debug_info": {
            "input_tokens": input_length,
            "output_tokens": output_length - input_length,
            "generation_time": f"{gen_time:.2f}s",
            "final_length": len(text)
        }
    }

# =========================
# VLM 분석 (LLaVA)
# =========================
@app.post("/analyze_vlm")
def analyze_vlm(req: VLMAnalysisRequest):
    print("\n" + "="*60)
    print("[VLM ANALYZE] 요청 시작")
    print("="*60)
    
    if vlm_model is None or vlm_processor is None:
        print("[ERROR] VLM 모델이 로드되지 않음")
        raise HTTPException(503, detail="VLM not loaded")
    
    if not os.path.exists(req.image_path):
        print(f"[ERROR] 이미지 파일 없음: {req.image_path}")
        raise HTTPException(400, detail=f"image_path not found: {req.image_path}")

    try:
        # 1. 이미지 로드
        print(f"\n[IMAGE] 로드: {req.image_path}")
        img = Image.open(req.image_path).convert("RGB")
        print(f"[IMAGE] 크기: {img.size}")

        # 2. 프롬프트 준비
        print(f"\n[PROMPT] 길이: {len(req.prompt)} 문자")
        print(f"[PROMPT] 첫 200자:\n{req.prompt[:200]}...")
        
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
            print("[TEMPLATE] Chat template 적용 성공")
        except Exception as e:
            print(f"[TEMPLATE] Chat template 실패, 기본 텍스트 사용: {e}")
            prompt_text = req.prompt.strip()

        # 3. 입력 준비
        print(f"\n[PROCESSOR] 입력 준비 중...")
        inputs = vlm_processor(images=img, text=prompt_text, return_tensors="pt").to(vlm_model.device)
        print(f"[PROCESSOR] 완료")

        # 4. 생성 파라미터
        do_sample = (req.temperature or 0) > 0
        gen_kwargs = dict(
            max_new_tokens=min(max(req.max_new_tokens, 16), 1024),
            temperature=float(max(min(req.temperature, 1.5), 0.0)),
            do_sample=do_sample,
        )
        if do_sample:
            gen_kwargs.update(dict(top_p=0.9))
        
        print(f"\n[GEN_KWARGS]")
        print(f"  - max_new_tokens: {gen_kwargs['max_new_tokens']}")
        print(f"  - temperature: {gen_kwargs['temperature']}")
        print(f"  - do_sample: {do_sample}")

        # 5. 생성
        print(f"\n[GENERATE] 시작...")
        start_time = time.time()
        
        with torch.no_grad():
            out = vlm_model.generate(**inputs, **gen_kwargs)
        
        gen_time = time.time() - start_time
        print(f"[GENERATE] 완료 ({gen_time:.2f}초)")

        # 6. 디코딩
        print(f"\n[DECODE] 시작...")
        text = vlm_processor.batch_decode(out, skip_special_tokens=True)[0]
        
        print(f"[DECODE] 원본 길이: {len(text)} 문자")
        print(f"[DECODE] 원본 전체:\n{text}")
        
        print("\n" + "="*60)
        print("[VLM ANALYZE] 요청 완료")
        print("="*60 + "\n")
        
        return {
            "status": "success",
            "analysis": text,
            "model": vlm_name,
            "used_temperature": gen_kwargs["temperature"],
            "max_new_tokens": gen_kwargs["max_new_tokens"],
            "debug_info": {
                "generation_time": f"{gen_time:.2f}s",
                "output_length": len(text)
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"\n[ERROR] VLM 추론 오류:")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, detail=f"VLM inference error: {e}")

# =========================
# 서버 실행 (개발용)
# =========================
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "5001"))
    uvicorn.run("llm_server:app", host="0.0.0.0", port=port, reload=False)

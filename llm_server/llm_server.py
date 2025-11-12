# llm_server.py 수정

import os
import time
import re
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
    max_new_tokens: int = 1024  # ✅ 기본값 증가
    #temperature: float = 0.7
    temperature: float = 0.2

class VLMAnalysisRequest(BaseModel):
    image_path: str
    prompt: str
    max_new_tokens: int = 1024  # ✅ 기본값 증가
    temperature: float = 0.2

# =========================
# 유틸: 프롬프트 빌더(LLM)
# =========================
def _build_prompt(req: AnalysisRequest) -> str:
    """LLM용 프롬프트 생성"""
    
    # 매뉴얼 정보
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
- 정확히 4개 섹션만 작성 (각 2-3문장)
- 추측이나 예시 반복 금지
- 각 섹션은 반드시 "**섹션명**" 형식으로 시작

【출력 형식】
**불량 현황**
(판정 결과 요약 2-3문장)

**원인 분석**
(매뉴얼 원인 인용 2-3문장)

**대응 방안**
(즉시 조치 2-3개 항목)

**예방 조치**
(재발 방지 2-3개 항목)

위 4개 섹션만 작성하고 즉시 종료하세요. 추가 설명 불필요.
"""
    
    return prompt

# =========================
# 유틸: 프롬프트 빌더(VLM) ✅ 추가
# =========================
def _build_prompt_vlm(
    image_path: str,
    product: str,
    defect_en: str,
    defect_ko: str,
    full_name_ko: str,
    anomaly_score: float,
    is_anomaly: bool,
    manual_context: Dict[str, List[str]]
) -> str:
    """VLM용 프롬프트 생성 (이미지 포함)"""
    
    causes = manual_context.get("원인", [])
    actions = manual_context.get("조치", [])
    has_manual = bool(causes or actions)
    
    # 판정 상태
    if is_anomaly:
        status = f"불량 검출 (이상점수: {anomaly_score:.4f})"
    else:
        status = f"정상 범위 (이상점수: {anomaly_score:.4f})"
    
    prompt = f"""당신은 제조 품질 전문가입니다. 이미지를 보고 아래 정보를 바탕으로 보고서를 작성하세요.

【검사 결과】
제품: {product}
불량: {defect_ko} ({defect_en})
정식명칭: {full_name_ko}
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
- 이미지에서 보이는 불량을 매뉴얼과 연관지어 분석
- 매뉴얼 문장을 따옴표로 인용
- 정확히 4개 섹션만 작성
- 불확실한 추정 금지

【출력 형식】
**불량 현황**
(이미지 기반 판정 2-3문장)

**원인 분석**
(매뉴얼 원인 + 이미지 분석 2-3문장)

**대응 방안**
(즉시 조치 2-3개 항목)

**예방 조치**
(재발 방지 2-3개 항목)

위 4개 섹션만 작성하고 종료하세요.
"""
    
    return prompt

# =========================
# 유틸: LLM 응답 슬라이싱 ✅ 개선
# =========================
def _extract_four_sections(text: str) -> str:
    """
    4개 섹션(불량 현황, 원인 분석, 대응 방안, 예방 조치)만 추출
    
    Args:
        text: LLM 원본 응답
    
    Returns:
        4개 섹션만 포함된 정제된 텍스트
    """
    
    # 1. 불필요한 앞부분 제거
    text = text.split("assistant")[0].strip()
    text = text.split("[회사")[0].strip()
    
    # 불필요한 서론 제거
    unwanted_prefixes = [
        "제출된 문서에서",
        "보고서 제목:",
        "보고서를 작성합니다",
        "다음과 같이 작성하였습니다"
    ]
    for prefix in unwanted_prefixes:
        if text.startswith(prefix):
            # 첫 번째 **로 시작하는 라인까지 제거
            lines = text.split('\n')
            for i, line in enumerate(lines):
                if line.strip().startswith('**'):
                    text = '\n'.join(lines[i:])
                    break
    
    # 2. 4개 섹션 헤더 찾기
    section_patterns = [
        r'\*\*불량\s*현황\*\*',
        r'\*\*원인\s*분석\*\*',
        r'\*\*대응\s*방안\*\*',
        r'\*\*예방\s*조치\*\*'
    ]
    
    section_positions = []
    for pattern in section_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            section_positions.append((match.start(), match.group()))
    
    # 섹션을 찾지 못한 경우 원본 반환
    if len(section_positions) < 4:
        print(f"[WARN] 4개 섹션을 찾지 못함 (발견: {len(section_positions)}개)")
        return text
    
    # 3. 섹션별로 분리
    section_positions.sort(key=lambda x: x[0])
    
    # 예방 조치 섹션 끝 찾기
    last_section_start = section_positions[3][0]
    
    # 예방 조치 이후 3-5개 라인만 포함
    lines = text[last_section_start:].split('\n')
    
    # 빈 라인 제외하고 실제 내용만 카운트
    content_lines = 0
    end_line_idx = 0
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith('**'):
            content_lines += 1
        
        # 예방 조치 내용 3-5줄 확보
        if content_lines >= 4:
            end_line_idx = i + 1
            break
    
    # 섹션 4개 추출
    if end_line_idx > 0:
        extracted = text[:last_section_start] + '\n'.join(lines[:end_line_idx])
    else:
        # 기본값: 예방 조치 + 7줄
        end_pos = len(text)
        for i, line in enumerate(lines[:10], 1):
            if i == 7:
                end_pos = text.index(line, last_section_start) + len(line)
                break
        extracted = text[:end_pos]
    
    # 4. 마지막 정제
    extracted = extracted.strip()
    
    # 코드 블록이나 이상한 마크다운 제거
    extracted = re.sub(r'```.*?```', '', extracted, flags=re.DOTALL)
    extracted = re.sub(r'```.*', '', extracted)
    
    return extracted

# =========================
# 모델 로더
# =========================
@app.on_event("startup")
async def load_models_on_startup():
    global llm_name, llm_model, llm_tokenizer
    global vlm_name, vlm_model, vlm_processor

    print("\n" + "="*60)
    print("LLM/VLM 서버 시작")
    print("="*60)

    # ---- LLM ----
    try:
        llm_name = os.getenv("LLM_MODEL", "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B")
        print(f"\n[1/2] LLM 로드 시도: {llm_name}")

        try:
            llm_tokenizer = AutoTokenizer.from_pretrained(
                llm_name, use_fast=True, trust_remote_code=True
            )
            print("✅ LLM 토크나이저 로드 완료 (fast)")
        except Exception as e:
            print(f"[WARN] LLM fast tokenizer 실패 → slow 재시도")
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
        print(f"\n[2/2] VLM 로드 시도: {vlm_name}")

        vlm_model = LlavaForConditionalGeneration.from_pretrained(
            vlm_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        vlm_processor = AutoProcessor.from_pretrained(vlm_name)
        print("✅ VLM 로드 완료")
        
    except Exception as e:
        print(f"⚠️ VLM 로드 실패: {e}")
        vlm_name = None
        vlm_model = None
        vlm_processor = None
    
    print("\n" + "="*60)
    print("서버 초기화 완료")
    print("="*60 + "\n")

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
# LLM 분석 ✅ 개선
# =========================
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
    
    # 2. 토크나이징
    device = next(llm_model.parameters()).device
    print(f"[DEVICE] {device}")
    
    inputs = llm_tokenizer(prompt, return_tensors="pt").to(device)
    input_length = inputs['input_ids'].shape[1]
    print(f"[INPUT] 토큰 수: {input_length}")

    # 3. 생성 파라미터
    do_sample = (req.temperature or 0) > 0
    gen_kwargs = dict(
        max_new_tokens=min(max(req.max_new_tokens, 16), 1024),  # ✅ 최대 1024
        temperature=float(max(min(req.temperature, 1.5), 0.0)),
        do_sample=do_sample,
        repetition_penalty=1.3,
    )
    if do_sample:
        gen_kwargs.update(dict(top_p=0.9))
    
    print(f"\n[GEN_KWARGS] max_new_tokens={gen_kwargs['max_new_tokens']}, temp={gen_kwargs['temperature']}")

    # 4. 생성
    print(f"[GENERATE] 시작...")
    start_time = time.time()
    
    with torch.no_grad():
        output_ids = llm_model.generate(**inputs, **gen_kwargs)
    
    gen_time = time.time() - start_time
    output_length = output_ids.shape[1]
    print(f"[GENERATE] 완료 ({gen_time:.2f}초, 생성 토큰: {output_length - input_length})")

    # 5. 디코딩
    generated_ids = output_ids[0][input_length:]
    text = llm_tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    print(f"[DECODE] 원본 길이: {len(text)} 문자")
    
    # 6. 4개 섹션 추출 ✅ 개선된 슬라이싱
    text = _extract_four_sections(text)
    
    print(f"[FINAL] 최종 길이: {len(text)} 문자")
    print(f"[FINAL] 최종 라인 수: {len(text.split(chr(10)))}")
    
    print("="*60 + "\n")
    
    return {
        "status": "success",
        "analysis": text,
        "model": llm_name,
        "used_temperature": gen_kwargs["temperature"],
        "max_new_tokens": gen_kwargs["max_new_tokens"],
    }

# =========================
# VLM 분석 ✅ 개선 (프롬프트 정제 추가)
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

        # 2. 프롬프트 사용 (기존 prompt 또는 정제된 prompt)
        prompt_text = req.prompt.strip()
        print(f"[PROMPT] 길이: {len(prompt_text)} 문자")
        
        # Chat template 적용 시도
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            prompt_text = vlm_processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            print("[TEMPLATE] Chat template 적용 성공")
        except Exception:
            print("[TEMPLATE] Chat template 실패, 기본 텍스트 사용")

        # 3. 입력 준비
        inputs = vlm_processor(images=img, text=prompt_text, return_tensors="pt").to(vlm_model.device)

        # 4. 생성 파라미터
        do_sample = (req.temperature or 0) > 0
        gen_kwargs = dict(
            max_new_tokens=min(max(req.max_new_tokens, 16), 1024),  # ✅ 최대 1024
            temperature=float(max(min(req.temperature, 1.5), 0.0)),
            do_sample=do_sample,
        )
        if do_sample:
            gen_kwargs.update(dict(top_p=0.9))
        
        print(f"[GEN_KWARGS] max_new_tokens={gen_kwargs['max_new_tokens']}, temp={gen_kwargs['temperature']}")

        # 5. 생성
        print(f"[GENERATE] 시작...")
        start_time = time.time()
        
        with torch.no_grad():
            out = vlm_model.generate(**inputs, **gen_kwargs)
        
        gen_time = time.time() - start_time
        print(f"[GENERATE] 완료 ({gen_time:.2f}초)")

        # 6. 디코딩
        text = vlm_processor.batch_decode(out, skip_special_tokens=True)[0]
        print(f"[DECODE] 원본 길이: {len(text)} 문자")
        
        # 7. VLM 응답 정제 ✅
        # ASSISTANT: 이후 텍스트만 추출
        if "ASSISTANT:" in text:
            text = text.split("ASSISTANT:")[-1].strip()
            print("[CLEAN] ASSISTANT: 이후 추출")
        
        # USER: 이전까지만
        if "USER:" in text:
            text = text.split("USER:")[0].strip()
            print("[CLEAN] USER: 이전까지 추출")
        
        # 4개 섹션 추출
        text = _extract_four_sections(text)
        
        print(f"[FINAL] 최종 길이: {len(text)} 문자")
        print("="*60 + "\n")
        
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
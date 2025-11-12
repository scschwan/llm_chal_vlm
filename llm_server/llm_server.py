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
app = FastAPI(title="LLM/VLM Server", version="2.0")

# =========================
# 전역 모델 핸들
# =========================
# HyperCLOVAX
hyperclovax_name: Optional[str] = None
hyperclovax_model: Optional[AutoModelForCausalLM] = None
hyperclovax_tokenizer: Optional[AutoTokenizer] = None

# EXAONE 3.5
exaone_name: Optional[str] = None
exaone_model: Optional[AutoModelForCausalLM] = None
exaone_tokenizer: Optional[AutoTokenizer] = None

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
    """HyperCLOVAX 분석 요청"""
    product: str
    defect_en: str
    defect_ko: str
    full_name_ko: str
    anomaly_score: float = 0.0
    is_anomaly: bool = False
    manual_context: Dict[str, List[str]] = {}
    max_new_tokens: int = 1024  # ✅ 기본값 증가
    temperature: float = 0.7
    #temperature: float = 0.2

class ExaoneAnalysisRequest(BaseModel):
    """EXAONE 3.5 분석 요청"""
    product: str
    defect_en: str
    defect_ko: str
    full_name_ko: str
    anomaly_score: float = 0.0
    is_anomaly: bool = False
    manual_context: Dict[str, List[str]] = {}
    max_new_tokens: int = 1024
    temperature: float = 0.7
    # EXAONE 전용 파라미터
    top_p: float = 0.9
    repetition_penalty: float = 1.1

class VLMAnalysisRequest(BaseModel):
    image_path: str
    # ✅ VLM 프롬프트 빌더를 위한 추가 필드
    product: Optional[str] = None
    defect_en: Optional[str] = None
    defect_ko: Optional[str] = None
    full_name_ko: Optional[str] = None
    anomaly_score: Optional[float] = None
    is_anomaly: Optional[bool] = None
    manual_context: Optional[Dict[str, List[str]]] = None
    # 기존 필드
    prompt: Optional[str] = None  # ✅ Optional로 변경
    max_new_tokens: int = 1024
    temperature: float = 0.7

# =========================
# 유틸: 프롬프트 빌더(LLM)
# =========================
def _build_prompt_hyperclovax(req: AnalysisRequest) -> str:
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


def _build_prompt_exaone(req: ExaoneAnalysisRequest) -> str:
    """EXAONE 3.5용 프롬프트 (Chat Template 형식)"""
    causes = req.manual_context.get("원인", [])
    actions = req.manual_context.get("조치", [])
    has_manual = bool(causes or actions)
    
    if req.is_anomaly:
        status = f"불량 검출 (이상점수: {req.anomaly_score:.4f})"
    else:
        status = f"정상 범위 (이상점수: {req.anomaly_score:.4f})"
    
    system_prompt = "당신은 제조 품질 관리 전문가입니다. 주어진 정보를 바탕으로 정확하고 간결한 불량 분석 보고서를 작성합니다."
    
    user_content = f"""다음 검사 결과를 분석하여 보고서를 작성하세요.

【검사 결과】
제품: {req.product}
불량 유형: {req.defect_ko} ({req.defect_en})
정식 명칭: {req.full_name_ko}
판정: {status}

【참조 매뉴얼】
"""
    
    if has_manual:
        if causes:
            user_content += "발생 원인:\n"
            for i, cause in enumerate(causes, 1):
                user_content += f"{i}. {cause}\n"
        if actions:
            user_content += "\n조치 방법:\n"
            for i, action in enumerate(actions, 1):
                user_content += f"{i}. {action}\n"
    else:
        user_content += "※ 매뉴얼 정보 없음\n"
    
    user_content += """
【작성 지침】
1. 매뉴얼 내용을 직접 인용 (따옴표 사용)
2. 정확히 4개 섹션만 작성
3. 각 섹션은 2-3문장으로 간결하게
4. 추측이나 불필요한 설명 금지

【출력 형식】
**불량 현황**
(판정 결과 요약)

**원인 분석**
(매뉴얼 원인 인용)

**대응 방안**
(즉시 조치사항)

**예방 조치**
(재발 방지 방안)

위 4개 섹션만 작성하세요.
"""
    
    return system_prompt, user_content


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
- 위 매뉴얼 내용을 직접 인용 (따옴표 사용)
- 정확히 4개 섹션만 작성
- 불확실한 추정 금지
- 각 섹션은 반드시 "**섹션명**" 형식으로 시작

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
    global hyperclovax_name, hyperclovax_model, hyperclovax_tokenizer
    global exaone_name, exaone_model, exaone_tokenizer
    global vlm_name, vlm_model, vlm_processor

    print("\n" + "="*60)
    print("LLM/VLM 서버 시작")
    print("="*60)

    # ---- LLM ----
    try:
        #llm_name = os.getenv("LLM_MODEL", "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B")
        hyperclovax_name ="naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"
        print(f"\n[1/2] LLM 로드 시도: {hyperclovax_name}")

        try:
            hyperclovax_tokenizer = AutoTokenizer.from_pretrained(
                hyperclovax_name, use_fast=True, trust_remote_code=True
            )
            print("✅ LLM 토크나이저 로드 완료 (fast)")
        except Exception as e:
            print(f"[WARN] LLM fast tokenizer 실패 → slow 재시도")
            hyperclovax_tokenizer = AutoTokenizer.from_pretrained(
                hyperclovax_name, use_fast=False, trust_remote_code=True
            )
            print("✅ LLM 토크나이저 로드 완료 (slow)")

        hyperclovax_model = AutoModelForCausalLM.from_pretrained(
            hyperclovax_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        print("✅ LLM 로드 완료")
        
    except Exception as e:
        print(f"⚠️ LLM 로드 실패: {e}")
        hyperclovax_name = None
        hyperclovax_model = None
        hyperclovax_tokenizer = None

    # ---- EXAONE 3.5 ----
    try:
        exaone_name = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
        print(f"\n[2/3] EXAONE 3.5 로드 시도: {exaone_name}")

        try:
            exaone_tokenizer = AutoTokenizer.from_pretrained(
                exaone_name, use_fast=True, trust_remote_code=True
            )
            print("✅ EXAONE 토크나이저 로드 완료 (fast)")
        except:
            exaone_tokenizer = AutoTokenizer.from_pretrained(
                exaone_name, use_fast=False, trust_remote_code=True
            )
            print("✅ EXAONE 토크나이저 로드 완료 (slow)")

        exaone_model = AutoModelForCausalLM.from_pretrained(
            exaone_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        print("✅ EXAONE 3.5 로드 완료")
        
    except Exception as e:
        print(f"⚠️ EXAONE 로드 실패: {e}")
        exaone_name = None
        exaone_model = None
        exaone_tokenizer = None

    # ---- VLM (LLaVA) ----
    try:
        vlm_name = os.getenv("VLM_MODEL", "llava-hf/llava-1.5-7b-hf")
        print(f"\n[3/3] VLM 로드 시도: {vlm_name}")

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
    print("[HyperCLOVAX ANALYZE] 요청 시작")
    print("="*60)

    if hyperclovax_model is None or hyperclovax_tokenizer is None:
        print("[ERROR] HyperCLOVAX 모델이 로드되지 않음")
        raise HTTPException(503, detail="HyperCLOVAX not loaded")

    # 1. 프롬프트 생성
    prompt = _build_prompt_hyperclovax(req)
    print(f"\n[PROMPT] 길이: {len(prompt)} 문자")
    
    # 2. 토크나이징
    device = next(hyperclovax_model.parameters()).device
    print(f"[DEVICE] {device}")


    inputs = hyperclovax_tokenizer(prompt, return_tensors="pt").to(device)
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
        output_ids = hyperclovax_model.generate(**inputs, **gen_kwargs)
    
    gen_time = time.time() - start_time
    output_length = output_ids.shape[1]
    print(f"[GENERATE] 완료 ({gen_time:.2f}초, 생성 토큰: {output_length - input_length})")

    # 5. 디코딩
    generated_ids = output_ids[0][input_length:]
    text = hyperclovax_tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    print(f"[DECODE] 원본 길이: {len(text)} 문자")
    if len(text) < 1000 : 
        print(f"[DECODE] 원본 데아터: {text}")
    
    # 6. 4개 섹션 추출 ✅ 개선된 슬라이싱
    text = _extract_four_sections(text)
    
    print(f"[FINAL] 최종 길이: {len(text)} 문자")
    print(f"[FINAL] 최종 라인 수: {len(text.split(chr(10)))}")
    
    print("="*60 + "\n")
    
    return {
        "status": "success",
        "analysis": text,
        "model": "HyperCLOVAX",
        "used_temperature": gen_kwargs["temperature"],
        "max_new_tokens": gen_kwargs["max_new_tokens"],
    }

# =========================
# EXAONE 3.5 분석
# =========================
@app.post("/analyze_exaone")
def analyze_exaone(req: ExaoneAnalysisRequest):
    print("\n" + "="*60)
    print("[EXAONE 3.5 ANALYZE] 요청 시작")
    print("="*60)
    
    if exaone_model is None or exaone_tokenizer is None:
        print("[ERROR] EXAONE 모델이 로드되지 않음")
        raise HTTPException(503, detail="EXAONE not loaded")

    # Chat template 형식으로 프롬프트 생성
    system_prompt, user_content = _build_prompt_exaone(req)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    
    # apply_chat_template 사용
    input_ids = exaone_tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    
    device = next(exaone_model.parameters()).device
    input_ids = input_ids.to(device)
    input_length = input_ids.shape[1]
    
    print(f"[INPUT] 토큰 수: {input_length}")

    do_sample = (req.temperature or 0) > 0
    gen_kwargs = dict(
        max_new_tokens=min(max(req.max_new_tokens, 16), 1024),
        temperature=float(max(min(req.temperature, 1.5), 0.0)),
        do_sample=do_sample,
        repetition_penalty=req.repetition_penalty,
        eos_token_id=exaone_tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs.update(dict(top_p=req.top_p))

    print(f"[GEN_KWARGS] max_tokens={gen_kwargs['max_new_tokens']}, temp={gen_kwargs['temperature']}, rep_penalty={gen_kwargs['repetition_penalty']}")

    start_time = time.time()
    with torch.no_grad():
        output_ids = exaone_model.generate(input_ids=input_ids, **gen_kwargs)
    
    gen_time = time.time() - start_time
    
    # 전체 출력 디코딩
    full_text = exaone_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print(f"[DECODE] 원본 길이: {len(full_text)} 문자")
    if len(full_text) < 1500 : 
        print(f"[DECODE] 원본 데아터: {full_text}")
  
    '''
    # ASSISTANT: 이후 텍스트만 추출
    if "ASSISTANT:" in full_text:
        text = full_text.split("ASSISTANT:")[-1].strip()
        print("[CLEAN] ASSISTANT: 이후 추출")
    else:
        text = full_text
    '''
    text = _extract_four_sections(full_text)
    
    print(f"[EXAONE] 완료 ({gen_time:.2f}초, {len(text)} 문자)")
    print(f"[FINAL] 최종 라인 수: {len(text.split(chr(10)))}")
    print("="*60 + "\n")
    
    return {
        "status": "success",
        "analysis": text,
        "model": "EXAONE-3.5-2.4B",
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

        # 2. 프롬프트 생성 ✅ _build_prompt_vlm 사용
        if req.product and req.defect_ko and req.manual_context is not None:
            # 구조화된 프롬프트 생성
            prompt_text = _build_prompt_vlm(
                image_path=req.image_path,
                product=req.product,
                defect_en=req.defect_en or "",
                defect_ko=req.defect_ko,
                full_name_ko=req.full_name_ko or req.defect_ko,
                anomaly_score=req.anomaly_score or 0.0,
                is_anomaly=req.is_anomaly if req.is_anomaly is not None else False,
                manual_context=req.manual_context
            )
            print("[PROMPT] _build_prompt_vlm 사용")
        elif req.prompt:
            # 기존 방식 (직접 프롬프트 제공)
            prompt_text = req.prompt.strip()
            print("[PROMPT] 직접 제공된 프롬프트 사용")
        else:
            raise HTTPException(400, "product/defect_ko/manual_context 또는 prompt 필드가 필요합니다")
        
        print(f"[PROMPT] 길이: {len(prompt_text)} 문자")
        
        # 3. Chat template 적용 시도
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

        # 4. 입력 준비
        inputs = vlm_processor(images=img, text=prompt_text, return_tensors="pt").to(vlm_model.device)

        # 5. 생성 파라미터
        do_sample = (req.temperature or 0) > 0
        gen_kwargs = dict(
            max_new_tokens=min(max(req.max_new_tokens, 16), 1024),
            temperature=float(max(min(req.temperature, 1.5), 0.0)),
            do_sample=do_sample,
        )
        if do_sample:
            gen_kwargs.update(dict(top_p=0.9))
        
        print(f"[GEN_KWARGS] max_new_tokens={gen_kwargs['max_new_tokens']}, temp={gen_kwargs['temperature']}")

        # 6. 생성
        print(f"[GENERATE] 시작...")
        start_time = time.time()
        
        with torch.no_grad():
            out = vlm_model.generate(**inputs, **gen_kwargs)
        
        gen_time = time.time() - start_time
        print(f"[GENERATE] 완료 ({gen_time:.2f}초)")

        # 7. 디코딩
        text = vlm_processor.batch_decode(out, skip_special_tokens=True)[0]
        print(f"[DECODE] 원본 길이: {len(text)} 문자")
        
        # 8. VLM 응답 정제
        if "ASSISTANT:" in text:
            text = text.split("ASSISTANT:")[-1].strip()
            print("[CLEAN] ASSISTANT: 이후 추출")
        
        if "USER:" in text:
            text = text.split("USER:")[0].strip()
            print("[CLEAN] USER: 이전까지 추출")
        
        # 9. 4개 섹션 추출
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
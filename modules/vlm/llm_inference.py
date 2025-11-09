"""
LLM (Text-only Language Model) 추론 엔진
RAG 기반 불량 분석 및 매뉴얼 생성
"""

from __future__ import annotations
from typing import Dict, List
import torch

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        BitsAndBytesConfig
    )
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False


class LLMInference:
    """텍스트 기반 LLM 추론"""
    
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        device: str = "cuda",
        use_4bit: bool = True,
        verbose: bool = True
    ):
        if not LLM_AVAILABLE:
            raise ImportError("Transformers 라이브러리가 필요합니다")
        
        self.model_name = model_name
        self.device = device
        self.verbose = verbose
        
        if self.verbose:
            print(f"🤖 LLM 모델 로드 중: {model_name}")
        
        # 양자화 설정
        quantization_config = None
        if use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            if self.verbose:
                print("⚙️  4-bit 양자화 활성화")
        
        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 모델 로드
        model_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": device
        }
        
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        if self.verbose:
            print("✅ LLM 모델 로드 완료")
    
    def analyze_defect_with_context(
        self,
        product: str,
        defect_en: str,
        defect_ko: str,
        full_name_ko: str,
        anomaly_score: float,
        is_anomaly: bool,
        manual_context: Dict[str, List[str]],
        max_new_tokens: int = 512,
        temperature: float = 0.7
    ) -> str:
        """
        RAG 컨텍스트 기반 불량 분석
        
        Args:
            product: 제품명
            defect_en: 영문 불량명
            defect_ko: 한글 불량명  
            full_name_ko: 전체 한글 불량명
            anomaly_score: 이상 점수
            is_anomaly: 이상 여부
            manual_context: RAG 검색 결과 {"원인": [...], "조치": [...]}
            max_new_tokens: 최대 생성 토큰
            temperature: 샘플링 온도
        
        Returns:
            분석 결과 텍스트
        """
        if self.verbose:
            print("📝 프롬프트 생성 중...")
        
        # 프롬프트 구성
        prompt = self._build_analysis_prompt(
            product=product,
            defect_en=defect_en,
            defect_ko=defect_ko,
            full_name_ko=full_name_ko,
            anomaly_score=anomaly_score,
            is_anomaly=is_anomaly,
            manual_context=manual_context
        )
        
        if self.verbose:
            print(f"🔮 LLM 추론 중... (프롬프트 길이: {len(prompt)} 문자)")
        
        # 토큰화
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to(self.device)
        
        # 생성
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # 디코딩 (입력 제외)
        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        if self.verbose:
            print(f"✅ LLM 분석 완료 ({len(generated_text)} 문자)")
        
        return generated_text.strip()
    
    def _build_analysis_prompt(
        self,
        product: str,
        defect_en: str,
        defect_ko: str,
        full_name_ko: str,
        anomaly_score: float,
        is_anomaly: bool,
        manual_context: Dict[str, List[str]]
    ) -> str:
        """분석 프롬프트 생성"""
        
        # 매뉴얼 컨텍스트 정리
        causes = "\n".join([f"- {c}" for c in manual_context.get("원인", [])])
        actions = "\n".join([f"- {a}" for a in manual_context.get("조치", [])])
        
        prompt = f"""[INST] 당신은 제조업 품질 관리 전문가입니다. 아래 정보를 바탕으로 불량 분석 및 대응 방안을 제시하세요.

## 불량 정보
- 제품: {product}
- 불량 유형: {defect_ko} ({defect_en})
- 정식 명칭: {full_name_ko}
- 이상 검출 점수: {anomaly_score:.4f}
- 불량 판정: {"불량" if is_anomaly else "정상"}

## 매뉴얼 참조 정보

### 발생 원인
{causes if causes else "매뉴얼 정보 없음"}

### 조치 가이드
{actions if actions else "매뉴얼 정보 없음"}

## 요청사항
위 정보를 종합하여 다음 내용을 포함한 상세한 분석 보고서를 작성하세요:

1. **불량 현황 요약**: 검출된 불량의 특징과 심각도
2. **원인 분석**: 매뉴얼을 참고한 발생 원인 설명
3. **대응 방안**: 구체적이고 실행 가능한 조치 방법
4. **예방 조치**: 재발 방지를 위한 권장사항

답변은 한국어로 작성하고, 현장에서 즉시 활용 가능하도록 구체적으로 작성하세요. [/INST]

"""
        return prompt
    
    def unload_model(self):
        """모델 언로드"""
        if self.verbose:
            print("🗑️  LLM 모델 언로드 중...")
        
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()
        
        if self.verbose:
            print("✅ LLM 모델 언로드 완료")
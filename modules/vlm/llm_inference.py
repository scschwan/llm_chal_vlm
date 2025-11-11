"""
LLM (Text-only Language Model) 추론 엔진
RAG 기반 불량 분석 및 매뉴얼 생성
HyperCLOVA-X 또는 EXAONE 지원 추가
"""

from __future__ import annotations
from typing import Dict, List, Optional
import torch
import os

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
    
    # 지원 모델 목록
    SUPPORTED_MODELS = {
        "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
        "hyperclovax": "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B",
        #"exaone": "LGAI-EXAONE/EXAONE-4.0.1-32B"
        "exaone": "LGAI-EXAONE/EXAONE-4.0-1.2B"
    }
    
    def __init__(
        self,
        model_name: str = "hyperclovax",  # 기본값 변경
        device: str = "cuda",
        use_4bit: bool = True,
        verbose: bool = True,
        cache_dir: Optional[str] = None  # 캐시 디렉토리 지정
    ):
        if not LLM_AVAILABLE:
            raise ImportError("Transformers 라이브러리가 필요합니다")
        
        # 모델명 해석
        if model_name in self.SUPPORTED_MODELS:
            model_path = self.SUPPORTED_MODELS[model_name]
            self.model_type = model_name
        else:
            model_path = model_name
            self.model_type = "custom"
        
        self.model_name = model_path
        self.device = device
        self.verbose = verbose
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/huggingface")
        
        if self.verbose:
            print(f"🤖 LLM 모델 로드 중: {model_path}")
            print(f"📂 캐시 디렉토리: {self.cache_dir}")
        
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
        
        try:
            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                cache_dir=self.cache_dir,
                trust_remote_code=True  # HyperCLOVA-X/EXAONE용
            )
            
            # 모델 로드
            model_kwargs = {
                "cache_dir": self.cache_dir,
                "trust_remote_code": True  # HyperCLOVA-X/EXAONE용
            }
            
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
            else:
                # EXAONE은 bfloat16 권장
                if self.model_type == "exaone":
                    model_kwargs["torch_dtype"] = "bfloat16"
                else:
                    model_kwargs["torch_dtype"] = torch.float16
                model_kwargs["device_map"] = device
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **model_kwargs
            )
            
            if self.verbose:
                print("✅ LLM 모델 로드 완료")
                
        except Exception as e:
            if self.verbose:
                print(f"⚠️  LLM 로드 실패: {e}")
            raise
    
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
        
        # 모델별 입력 형식 처리
        if self.model_type == "hyperclovax":
            return self._generate_hyperclovax(prompt, max_new_tokens, temperature)
        elif self.model_type == "exaone":
            return self._generate_exaone(prompt, max_new_tokens, temperature)
        else:  # mistral 또는 기타
            return self._generate_mistral(prompt, max_new_tokens, temperature)
    
    def _generate_hyperclovax(self, prompt: str, max_new_tokens: int, temperature: float) -> str:
        """HyperCLOVA-X 전용 생성"""
        chat = [
            {"role": "tool_list", "content": ""},
            {"role": "system", "content": "당신은 제조업 품질 관리 전문가입니다."},
            {"role": "user", "content": prompt}
        ]
        
        inputs = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.device)
        
        output_ids = self.model.generate(
            **inputs,
            max_length=max_new_tokens + inputs["input_ids"].shape[1],
            stop_strings=["<|endofturn|>", "<|stop|>"],
            tokenizer=self.tokenizer,
            temperature=temperature,
            do_sample=True if temperature > 0 else False
        )
        
        # 입력 제외하고 디코딩
        generated_text = self.tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_text.strip()
    
    def _generate_exaone(self, prompt: str, max_new_tokens: int, temperature: float) -> str:
        """EXAONE 전용 생성"""
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        
        output = self.model.generate(
            input_ids.to(self.device),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True if temperature > 0 else False
        )
        
        generated_text = self.tokenizer.decode(
            output[0][input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_text.strip()
    
    def _generate_mistral(self, prompt: str, max_new_tokens: int, temperature: float) -> str:
        """Mistral 전용 생성 (기존 방식)"""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to(self.device)
        
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
        
        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
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
        """분석 프롬프트 생성 (모델별 최적화)"""
        
        causes = "\n".join([f"- {c}" for c in manual_context.get("원인", [])])
        actions = "\n".join([f"- {a}" for a in manual_context.get("조치", [])])
        
        # HyperCLOVA-X / EXAONE은 한국어 특화이므로 프롬프트 간소화
        if self.model_type in ["hyperclovax", "exaone"]:
            prompt = f"""당신은 제조업 품질 관리 전문가입니다. 다음 불량 정보를 분석하세요.

                ## 불량 정보
                - 제품: {product}
                - 불량 유형: {defect_ko} ({defect_en})
                - 정식 명칭: {full_name_ko}
                - 이상 검출 점수: {anomaly_score:.4f}
                - 불량 판정: {"불량" if is_anomaly else "정상"}

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
        else:  # Mistral
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
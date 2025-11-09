"""
VLM (Vision Language Model) 추론 엔진
멀티모달 이미지 분석
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Optional, Union
import torch
from PIL import Image

try:
    from transformers import (
        LlavaNextProcessor,
        LlavaNextForConditionalGeneration,
        BitsAndBytesConfig
    )
    LLAVA_AVAILABLE = True
except ImportError:
    # LlavaNext를 사용할 수 없는 경우 대체
    try:
        from transformers import (
            AutoProcessor,
            LlavaForConditionalGeneration as LlavaNextForConditionalGeneration,
            BitsAndBytesConfig
        )
        LlavaNextProcessor = AutoProcessor
        LLAVA_AVAILABLE = True
    except ImportError:
        LLAVA_AVAILABLE = False
        print("⚠️ Transformers LLaVA 모델을 사용할 수 없습니다.")



class VLMInference:
    """VLM 추론 엔진"""
    
    def __init__(
        self,
        model_name: str = "llava-hf/llava-v1.6-mistral-7b-hf",
        device: str = "cuda",
        use_4bit: bool = False,
        use_8bit: bool = False,
        verbose: bool = True
    ):
        if not LLAVA_AVAILABLE:
            raise ImportError(
                "Transformers의 LLaVA 모델을 사용할 수 없습니다. "
                "transformers 버전을 4.37.0 이상으로 업그레이드하세요: "
                "pip install transformers>=4.37.0 --upgrade"
            )
        
        self.model_name = model_name
        self.device = device
        self.verbose = verbose
        
        if self.verbose:
            print(f"🤖 VLM 모델 로드 중: {model_name}")
        
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
        elif use_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
            if self.verbose:
                print("⚙️  8-bit 양자화 활성화")
        
        # ✅ Processor 로드 - try-except 추가
        try:
            self.processor = LlavaNextProcessor.from_pretrained(model_name)
        except Exception as e:
            if self.verbose:
                print(f"⚠️  LlavaNextProcessor 로드 실패: {e}")
                print("   AutoProcessor로 재시도...")
            
            # 폴백: AutoProcessor 사용
            from transformers import AutoProcessor
            try:
                self.processor = AutoProcessor.from_pretrained(model_name)
            except Exception as e2:
                if self.verbose:
                    print(f"⚠️  AutoProcessor도 실패: {e2}")
                
                # 마지막 시도: LlavaProcessor
                try:
                    from transformers import LlavaProcessor
                    self.processor = LlavaProcessor.from_pretrained(model_name)
                    if self.verbose:
                        print("✅ LlavaProcessor로 대체 성공")
                except Exception as e3:
                    raise ImportError(
                        f"Processor 로드 실패. 모든 시도 실패:\n"
                        f"1. LlavaNextProcessor: {e}\n"
                        f"2. AutoProcessor: {e2}\n"
                        f"3. LlavaProcessor: {e3}"
                    )
        
        # 모델 로드
        model_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": device
        }
        
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        
        # ✅ 모델 로드도 try-except 추가
        try:
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                model_name,
                **model_kwargs
            )
        except Exception as e:
            if self.verbose:
                print(f"⚠️  LlavaNextForConditionalGeneration 로드 실패: {e}")
                print("   LlavaForConditionalGeneration으로 재시도...")
            
            from transformers import LlavaForConditionalGeneration
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_name,
                **model_kwargs
            )
        
        if self.verbose:
            print("✅ VLM 모델 로드 완료")
    
    def analyze_defect_with_segmentation(
        self,
        normal_image_path: Union[str, Path],
        defect_image_path: Union[str, Path],
        overlay_image_path: Union[str, Path],
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True
    ) -> str:
        """
        3개 이미지 기반 불량 분석
        """
        if self.verbose:
            print("🖼️  이미지 로드 중...")
        
        # 이미지 로드
        images = [
            Image.open(normal_image_path).convert("RGB"),
            Image.open(defect_image_path).convert("RGB"),
            Image.open(overlay_image_path).convert("RGB")
        ]
        
        if self.verbose:
            print(f"📝 프롬프트 길이: {len(prompt)} 문자")
            print("🔮 VLM 추론 중...")
        
        # 대화 형식 구성
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # ✅ 수정: try-except로 안전하게 처리
        try:
            # 프롬프트 템플릿 적용
            text_prompt = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True
            )
        except TypeError as e:
            # apply_chat_template이 실패하면 직접 프롬프트 구성
            if self.verbose:
                print(f"⚠️  Chat template 적용 실패, 직접 프롬프트 구성: {e}")
            
            # LLaVA 기본 프롬프트 형식
            text_prompt = f"USER: <image><image><image>\n{prompt}\nASSISTANT:"
        
        # 입력 준비
        try:
            inputs = self.processor(
                text=text_prompt,
                images=images,
                return_tensors="pt",
                padding=True
            ).to(self.device)
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Processor 오류: {e}")
                print("   기본 처리 방식으로 재시도...")
            
            # 폴백: 이미지와 텍스트를 따로 처리
            inputs = self.processor(
                images=images,
                text=text_prompt,
                return_tensors="pt",
                padding=True
            ).to(self.device)
        
        # 생성
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.processor.tokenizer.pad_token_id if hasattr(self.processor, 'tokenizer') else None
            )
        
        # 디코딩
        try:
            generated_text = self.processor.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
        except:
            # 전체 디코딩
            generated_text = self.processor.decode(
                outputs[0],
                skip_special_tokens=True
            )
        
        if self.verbose:
            print(f"✅ VLM 분석 완료 ({len(generated_text)} 문자)")
        
        return generated_text.strip()
    
    def analyze_simple(
        self,
        image_paths: List[Union[str, Path]],
        prompt: str,
        max_new_tokens: int = 512
    ) -> str:
        """
        간단한 멀티 이미지 분석 (유연한 이미지 개수)
        
        Args:
            image_paths: 이미지 경로 리스트
            prompt: 분석 프롬프트
            max_new_tokens: 최대 생성 토큰 수
        
        Returns:
            VLM 분석 결과
        """
        # 이미지 로드
        images = [Image.open(p).convert("RGB") for p in image_paths]
        
        # 대화 구성
        content = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": prompt})
        
        conversation = [{"role": "user", "content": content}]
        
        # 프롬프트 적용
        text_prompt = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True
        )
        
        # 추론
        inputs = self.processor(
            text=text_prompt,
            images=images,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True
            )
        
        generated_text = self.processor.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_text.strip()
    
    def unload_model(self):
        """모델 언로드 (메모리 확보)"""
        if self.verbose:
            print("🗑️  VLM 모델 언로드 중...")
        
        del self.model
        del self.processor
        torch.cuda.empty_cache()
        
        if self.verbose:
            print("✅ VLM 모델 언로드 완료")


if __name__ == "__main__":
    # 간단한 테스트
    print("VLM 추론 엔진 테스트")
    print("실제 모델을 로드하려면 충분한 GPU 메모리가 필요합니다.")
    print("\n사용 예시:")
    print("""
        vlm = VLMInference(
            model_name="llava-hf/llava-v1.6-mistral-7b-hf",
            use_4bit=True,  # 메모리 절약
            verbose=True
        )

        result = vlm.analyze_defect_with_segmentation(
            normal_image_path="normal.jpg",
            defect_image_path="defect.jpg",
            overlay_image_path="overlay.jpg",
            prompt="불량을 분석하세요..."
        )

        print(result)
    """)
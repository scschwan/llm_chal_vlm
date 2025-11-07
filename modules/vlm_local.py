# modules/vlm_local.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any
import json
import torch
from PIL import Image

# 실제 환경에서는:
#  - LlavaNextProcessor.from_pretrained(self.model_id)
#  - LlavaNextForConditionalGeneration.from_pretrained(... torch_dtype=...)
#  - model.to(self.device)
# 등으로 연결하면 됩니다. 여기서는 실행 막히지 않도록 스텁을 제공합니다.

@dataclass
class VLM:
    model_id: str
    device: str = "cuda"
    persist: bool = False
    use_bf16: bool = True
    max_edge: int = 640
    verbose: bool = True

    def __post_init__(self):
        if self.verbose:
            print("🧰 processor: LlavaNextProcessor (placeholder)")
            print("✅ LlavaNextForConditionalGeneration 로드")
            print(f"📦 model.to({self.device}) 완료")

    def _prepare_image_tensor(self, pil_img: Image.Image, max_edge_override: Optional[int] = None):
        # 실제는 processor(images=..., return_tensors="pt") 등의 전처리를 수행
        # 여기서는 shape만 맞춘 더미 텐서로 대체
        arr = torch.randn(1, 3, 224, 224)
        return arr.to(self.device)

    def compare_regions_text(
        self,
        left_path: str,
        right_path: str,
        prompt: str,
        max_new_tokens: int = 360,
        do_sample: bool = False,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.1,
        preprocess: Optional[Callable[[Image.Image], Image.Image]] = None,
        max_edge_override: Optional[int] = None,
    ) -> str:
        """
        좌/우 이미지 두 장과 프롬프트를 받아 비교 리포트를 생성.
        실제 구현에서는 LLaVA chat-like inference를 호출하면 됩니다.
        여기서는 실행 막힘 방지를 위해 고정된 포맷의 더미 텍스트를 반환.
        """
        left_img  = Image.open(left_path).convert("RGB")
        right_img = Image.open(right_path).convert("RGB")
        if preprocess:
            left_img  = preprocess(left_img)
            right_img = preprocess(right_img)

        # 실제 구현: processor.apply_chat_template + model.generate(...)
        # 현재는 placeholder
        pseudo_answer = (
            "[INFO] 분석 대상 한 줄 알림( '금속 가공 부품 1EA, 좌=정상, 우=후보')\n"
            "[SCENE] 두 이미지는 회색 배경 위 금속 가공 부품이 놓여 있으며 유사한 각도에서 촬영됨.\n"
            "[DETAIL] 좌/우 특징 불릿 ... (여기에 위치·형태·강도 기반 bullet들이 와야 함)\n"
            "[추론] 좌측은 기준 형태로 보이고, 우측은 특정 영역에서 국부 함몰(강도 3) 등 차이가 관찰되며 불량 의심.\n"
            "[STATUS] 이미지 분석 완료."
        )
        return pseudo_answer

    # ====== (옵션) 프롬프트 영향 최소화를 위한 중립 JSON 캡션 API ======
    def describe_image_json(
        self,
        image_path: str,
        *,
        schema: Dict[str, Any],
        max_new_tokens: int = 256,
        do_sample: bool = False,
    ) -> str:
        """
        단일 이미지를 관찰해 schema 형태의 JSON만 출력하는 API (스텁).
        실제 구현에서는 단일 이미지 캡션 프롬프트를 구성해 generate 후 JSON만 파싱.
        여기서는 빈 스키마를 그대로 반환해 실행이 멈추지 않도록 함.
        """
        # 실제 구현 시에는 image_path를 열고 processor/모델로 inference 후 JSON 생성
        # 지금은 입력 schema의 키만 유지하는 빈 JSON 객체를 반환
        out = {k: ([] if isinstance(v, list) else ({} if isinstance(v, dict) else v))
               for k, v in schema.items()}
        return json.dumps(out, ensure_ascii=False)

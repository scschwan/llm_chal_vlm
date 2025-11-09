"""
VLM 프롬프트 생성기
불량 분석을 위한 구조화된 프롬프트 생성
"""

from __future__ import annotations
from typing import List, Dict


class PromptBuilder:
    """VLM 프롬프트 빌더"""
    
    @staticmethod
    def build_defect_analysis_prompt(
        product: str,
        defect_en: str,
        defect_ko: str,
        full_name_ko: str,
        anomaly_regions: List[Dict],
        manual_context: Dict[str, List[str]],
        image_description: str = "3개의 이미지가 제공됩니다"
    ) -> str:
        """
        불량 분석 프롬프트 생성
        
        Args:
            product: 제품명
            defect_en: 영어 불량명
            defect_ko: 한글 불량명
            full_name_ko: 한글 전체명
            anomaly_regions: PatchCore 검출 영역 리스트
            manual_context: RAG 검색 결과 {"원인": [...], "조치": [...]}
            image_description: 이미지 설명
        
        Returns:
            프롬프트 문자열
        """
        # 이상 영역 텍스트 생성
        if anomaly_regions:
            regions_text = "\n".join([
                f"- 영역 {i+1}: 위치(x={r.get('x', 0)}, y={r.get('y', 0)}), "
                f"크기({r.get('w', 0)}×{r.get('h', 0)}px), "
                f"이상 점수={r.get('score', 0):.2f}"
                for i, r in enumerate(anomaly_regions)
            ])
        else:
            regions_text = "- 이상 영역 검출되지 않음"
        
        # 매뉴얼 컨텍스트 텍스트 생성
        cause_text = "\n".join([f"• {c}" for c in manual_context.get("원인", [])])
        action_text = "\n".join([f"• {a}" for a in manual_context.get("조치", [])])
        
        if not cause_text:
            cause_text = "(매뉴얼 정보 없음)"
        if not action_text:
            action_text = "(매뉴얼 정보 없음)"
        
        prompt = f"""당신은 제조 불량 검사 전문가입니다.

[제공된 이미지]
{image_description}:
1. 첫 번째 이미지: 정상 기준 이미지 (PatchCore 선정)
2. 두 번째 이미지: 검사 대상 불량 의심 이미지
3. 세 번째 이미지: AI가 검출한 이상 영역 표시 (빨간색 오버레이)

[검출 정보]
- 제품: {product}
- 불량 유형: {defect_ko} ({defect_en})
- 불량 상세: {full_name_ko}

[AI 검출 이상 영역]
{regions_text}

[참조 매뉴얼]

<발생 원인>
{cause_text}

<조치 가이드>
{action_text}

[분석 요청]
위 3개 이미지와 검출 정보를 종합하여 다음을 답변하세요:

## 1. 불량 위치 확인
- AI가 빨간색으로 표시한 이상 영역을 확인하세요
- 해당 영역이 실제 {defect_ko} 불량과 일치하는지 판단하세요
- 정확한 위치를 자연어로 설명하세요 (예: "중앙 상단부", "좌측 하단 모서리", "전체적으로 분포")

## 2. 불량 특징 분석
- 첫 번째 정상 이미지와 두 번째 불량 이미지를 비교하세요
- 불량의 크기, 형태, 범위를 설명하세요
- 심각도를 평가하세요 (경미/보통/심각)

## 3. 조치 방안
- 위 매뉴얼의 조치 가이드를 참고하여 구체적인 대응 방법을 제시하세요
- 우선순위가 높은 조치부터 나열하세요
- 현장에서 즉시 적용 가능한 실용적인 방법을 제안하세요

[출력 규칙]
- 각 섹션을 명확히 구분하세요
- 구체적이고 실용적인 내용으로 작성하세요
- 불확실한 내용은 "추정", "가능성" 등으로 표현하세요
- 전문 용어는 쉬운 설명을 함께 제공하세요
"""
        return prompt.strip()
    
    @staticmethod
    def build_simple_analysis_prompt(
        defect_ko: str,
        defect_en: str
    ) -> str:
        """
        간단한 분석 프롬프트 (RAG 없이)
        
        Args:
            defect_ko: 한글 불량명
            defect_en: 영어 불량명
        
        Returns:
            프롬프트 문자열
        """
        return f"""당신은 제조 불량 검사 전문가입니다.

[검출된 불량]
- 불량 유형: {defect_ko} ({defect_en})

[분석 요청]
제공된 이미지들을 분석하여:
1. 불량이 발생한 위치
2. 불량의 특징 및 심각도
3. 일반적인 조치 방안

을 간략히 설명하세요.
"""


if __name__ == "__main__":
    # 테스트
    builder = PromptBuilder()
    
    test_regions = [
        {"x": 120, "y": 80, "w": 45, "h": 60, "score": 0.85},
        {"x": 200, "y": 150, "w": 30, "h": 40, "score": 0.72}
    ]
    
    test_manual = {
        "원인": [
            "금형 접합면의 틈새로 용탕이 새어나와 얇은 판 모양 돌출 형성",
            "몰드 클램프 압력 부족 또는 금형 마모"
        ],
        "조치": [
            "금형 조립면 마모·오염 점검 후 재연마",
            "클램프 압력 및 위치 보정"
        ]
    }
    
    prompt = builder.build_defect_analysis_prompt(
        product="prod1",
        defect_en="burr",
        defect_ko="버",
        full_name_ko="날개 버, 얇은 돌출",
        anomaly_regions=test_regions,
        manual_context=test_manual
    )
    
    print("=== 생성된 프롬프트 ===")
    print(prompt)
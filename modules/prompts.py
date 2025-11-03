from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class PromptConfig:
    # 출력 섹션 헤더
    headings: tuple[str, ...] = ("INFO", "SCENE", "DETAIL", "추론", "STATUS")

    # 최소 불릿 개수
    min_left_points: int = 3
    min_right_points: int = 3
    max_points_per_side: int = 6

    # 언어 설정 (우린 ko 고정)
    language: str = "ko"

    # 텍스트/워터마크 읽지 말기
    ignore_text_reading: bool = True

    # 마지막 STATUS 문구
    status_text: str = "이미지 분석 완료."

    # 위치/강도 표현 가이드
    require_grid_location: bool = True
    require_height_percent: bool = True
    require_severity: bool = True
    severity_scale_hint: str = "강도 1(매우 약함)~5(매우 강함)"

    # 허용된 형태 어휘(허구 표현 줄이기 위해 힌트로만 제공)
    allowed_shape_terms: tuple[str, ...] = (
        "세로 굴곡","가로 굴곡","국부 함몰","가장자리 말림",
        "수직 스크래치","수평 스크래치","점상 오염","얼룩",
        "링(홈) 주기 변화","윤곽 흐림","반사 하이라이트","내용물 수위 변화",
        "뚜껑 유무","나사산 노출"
    )

    allowed_scene_terms: tuple[str, ...] = (
        "배경 종류","조명 밝기","그림자 유무","반사/광택","프레이밍/크롭"
    )


def _suppress_text_note(cfg: PromptConfig) -> str:
    if not cfg.ignore_text_reading:
        return ""
    return (
        "- 이미지 내 텍스트/워터마크/로고/숫자는 읽지 말고 의미화하지 마십시오. "
        "브랜드 추정 금지."
    )


def _fmt_list(name: str, items: Iterable[str]) -> str:
    xs = [x for x in (items or []) if x]
    return f"- {name}: " + (", ".join(xs) if xs else "없음")


def build_ok_def_pair_prompt(
    evidence_summary: str = "",
    roi_hint: str = "",
    grid_hints: Iterable[str] | None = None,
    hotspots: Iterable[str] | None = None,
    defect_level: str = "불명확",
    cfg: PromptConfig | None = None,
) -> str:
    """
    좌측 이미지를 정상(OK 기준), 우측 이미지를 비교/후보(DEF)로 본다.
    반드시 [INFO]/[SCENE]/[DETAIL]/[추론]/[STATUS] 순서로 출력하도록 유도.
    DETAIL은 좌/우 특징, 공통점을 bullet로 묘사하되
    위치/범위/형태/강도 등 정형화된 표현을 강요.
    """
    cfg = cfg or PromptConfig()

    aux_lines: List[str] = []
    if roi_hint:
        aux_lines.append(f"- ROI 힌트: {roi_hint}")
    if evidence_summary:
        aux_lines.append(f"- Evidence: {evidence_summary}")
    aux_lines.append(_fmt_list("SSIM 격자 힌트", grid_hints or []))
    aux_lines.append(_fmt_list("형상 변형 후보", hotspots or []))
    aux_lines.append(f"- 모델 1차 판별(우측): {defect_level}")

    location = "격자(상단/중단/하단 × 좌/중앙/우)"
    percent  = "높이 범위 %(예: 35~55%)"
    severity = f"강도 등급({cfg.severity_scale_hint})"

    detail_rule = (
        f"- [좌측=정상 기준] 최소 {cfg.min_left_points}개 불릿:\n"
        f"  [{location}, {percent}, 형태, 길이/폭(상대), {severity}] 형식으로 작성.\n"
        f"- [우측=비교 대상] 최소 {cfg.min_right_points}개 불릿:\n"
        f"  동일한 형식으로 작성.\n"
        f"- [공통] 좌우에서 동일하게 보이는 요소만 1~2개 불릿으로 요약.\n"
        f"- 형태/표면 상태 기술 시 다음 어휘 위주로 사용: "
        f"{', '.join(cfg.allowed_shape_terms)}\n"
        f"- 위치 없이 '찌그러짐 있음' 식의 포괄 표현 금지. 반드시 위치/범위/강도 포함."
    )

    guardrails = "\n".join([
        "- 출력에는 반드시 실제 관측 사실만 포함.",
        "- 원인 추정(예:'압력 때문에'), 성능/안전 판정, 수명 추정 금지.",
        "- 수치 날조 금지. 확실하지 않으면 '불명확(이유: …)'라고만 기록.",
        _suppress_text_note(cfg),
    ])

    return f"""
당신은 시각 검사 전문가입니다.
좌측 이미지는 '정상 기준(OK)', 우측 이미지는 '후보/비교 대상(DEF)'입니다.
두 이미지를 동시에 보고, 아래 섹션을 **그대로** 출력하십시오.
섹션 헤더: [{cfg.headings[0]}] → [{cfg.headings[1]}] → [{cfg.headings[2]}] → [{cfg.headings[3]}] → [{cfg.headings[4]}]

[{cfg.headings[0]}]
- 분석 대상 한 줄 알림 (예: '금속 가공 부품 1EA, 좌=정상, 우=후보')
- 반드시 실제 대상(부품/파트/물체)을 명시

[{cfg.headings[1]}]
- 장면/배경/촬영 조건을 1문장으로 요약
- 조명, 배경 종류, 배치 상태 등 객관적 사실만 언급

[{cfg.headings[2]}]
- 좌측(정상) 특징 불릿 나열
- 우측(후보) 특징 불릿 나열
- [공통] 불릿 1~2개 (좌우가 동일하게 보이는 요소만)
- 각 불릿은 다음 형식을 따른다:
  [{location}, {percent}, 형태/표면 상태, 길이/폭(상대), {severity}]
  예: "[중단 좌측, 40~60%, 수직 스크래치 여러 줄, 길이 짧음, 강도 2]"

[{cfg.headings[3]}]
- 핵심 차이 2~3가지 한 문장 요약
- 가능하면 우측(후보)의 이상 여부를 매우 짧게 정리
- 불량/정상 단정 대신 '의심', '뚜렷한 차이' 정도 표현 허용

[{cfg.headings[4]}]
- "{cfg.status_text}"

[DETAIL 작성 규칙]
{detail_rule}

[보조 정보(모델 내부 참고용)]
{chr(10).join(aux_lines)}

[규칙]
{guardrails}
""".strip()
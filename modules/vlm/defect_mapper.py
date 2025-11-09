"""
불량명 매핑 유틸리티
영어 불량명 ↔ 한글 불량명 변환 및 검색 키워드 제공
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class DefectInfo:
    """불량 정보"""
    en: str              # 영어명
    ko: str              # 한글명
    full_name_ko: str    # 한글 전체명
    pdf_keywords: List[str]  # PDF 검색 키워드


class DefectMapper:
    """불량명 매핑 관리자"""
    
    def __init__(self, mapping_file: str | Path):
        """
        Args:
            mapping_file: defect_mapping.json 경로
        """
        self.mapping_file = Path(mapping_file)
        self.mapping_data = self._load_mapping()
    
    def _load_mapping(self) -> Dict:
        """매핑 파일 로드"""
        if not self.mapping_file.exists():
            raise FileNotFoundError(
                f"매핑 파일을 찾을 수 없습니다: {self.mapping_file}"
            )
        
        with open(self.mapping_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_defect_info(self, product: str, defect_en: str) -> Optional[DefectInfo]:
        """
        불량 정보 조회
        
        Args:
            product: 제품명 (예: "prod1")
            defect_en: 영어 불량명 (예: "burr")
        
        Returns:
            DefectInfo 또는 None
        """
        if product not in self.mapping_data:
            return None
        
        if defect_en not in self.mapping_data[product]:
            return None
        
        data = self.mapping_data[product][defect_en]
        return DefectInfo(
            en=data["en"],
            ko=data["ko"],
            full_name_ko=data["full_name_ko"],
            pdf_keywords=data["pdf_keywords"]
        )
    
    def get_search_keywords(self, product: str, defect_en: str) -> List[str]:
        """
        PDF 검색용 키워드 반환
        
        Args:
            product: 제품명
            defect_en: 영어 불량명
        
        Returns:
            검색 키워드 리스트
        """
        info = self.get_defect_info(product, defect_en)
        if info is None:
            return [defect_en]  # 기본값
        
        return info.pdf_keywords
    
    def get_all_products(self) -> List[str]:
        """등록된 모든 제품명 반환"""
        return list(self.mapping_data.keys())
    
    def get_defects_for_product(self, product: str) -> List[str]:
        """특정 제품의 모든 불량명 반환"""
        if product not in self.mapping_data:
            return []
        
        return list(self.mapping_data[product].keys())


# 매핑 파일 생성 유틸리티
def create_default_mapping(output_path: str | Path):
    """
    기본 매핑 파일 생성
    
    Args:
        output_path: 저장 경로 (web/defect_mapping.json)
    """
    default_mapping = {
        "prod1": {
            "burr": {
                "en": "burr",
                "ko": "버",
                "full_name_ko": "날개 버, 얇은 돌출",
                "pdf_keywords": ["burr", "버", "날개", "돌출"]
            },
            "hole": {
                "en": "hole",
                "ko": "기공",
                "full_name_ko": "기공",
                "pdf_keywords": ["hole", "기공", "공극", "공기"]
            },
            "scratch": {
                "en": "scratch",
                "ko": "긁힘",
                "full_name_ko": "긁힘",
                "pdf_keywords": ["scratch", "긁힘", "스크래치", "흠집"]
            }
        },
        "prod2": {
            "crack": {
                "en": "crack",
                "ko": "균열",
                "full_name_ko": "균열",
                "pdf_keywords": ["crack", "균열", "크랙"]
            },
            "dent": {
                "en": "dent",
                "ko": "찌그러짐",
                "full_name_ko": "찌그러짐",
                "pdf_keywords": ["dent", "찌그러짐", "덴트"]
            }
        },
        "prod3": {
            "stain": {
                "en": "stain",
                "ko": "얼룩",
                "full_name_ko": "얼룩",
                "pdf_keywords": ["stain", "얼룩", "오염"]
            },
            "discolor": {
                "en": "discolor",
                "ko": "변색",
                "full_name_ko": "변색",
                "pdf_keywords": ["discolor", "변색", "색변화"]
            }
        }
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(default_mapping, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 매핑 파일 생성 완료: {output_path}")


if __name__ == "__main__":
    # 테스트 및 기본 파일 생성
    import sys
    from pathlib import Path
    
    # 프로젝트 루트에서 실행 가정
    project_root = Path(__file__).parent.parent.parent
    mapping_file = project_root / "web" / "defect_mapping.json"
    
    if not mapping_file.exists():
        print("기본 매핑 파일 생성 중...")
        create_default_mapping(mapping_file)
    
    # 테스트
    mapper = DefectMapper(mapping_file)
    
    print("\n=== 매핑 테스트 ===")
    info = mapper.get_defect_info("prod1", "burr")
    if info:
        print(f"불량명: {info.ko} ({info.en})")
        print(f"전체명: {info.full_name_ko}")
        print(f"키워드: {info.pdf_keywords}")
    
    print(f"\n등록된 제품: {mapper.get_all_products()}")
    print(f"prod1 불량: {mapper.get_defects_for_product('prod1')}")
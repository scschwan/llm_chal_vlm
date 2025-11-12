"""
불량 정보 매핑 관리자
"""
from pathlib import Path
from typing import Optional, List, Dict
from dataclasses import dataclass
import json


@dataclass
class DefectInfo:
    """불량 정보"""
    en: str
    ko: str
    full_name_ko: str
    keywords: List[str]


class DefectMapper:
    """불량 정보 매핑 관리자"""
    
    def __init__(self, mapping_file: Path):
        self.mapping_file = Path(mapping_file)
        
        if not self.mapping_file.exists():
            print(f"⚠️  매핑 파일이 없습니다: {self.mapping_file}")
            print("   기본 매핑 파일을 생성합니다...")
            create_default_mapping(self.mapping_file)
        
        # 매핑 데이터 로드
        with open(self.mapping_file, 'r', encoding='utf-8') as f:
            self.mapping_data = json.load(f)
        
        print(f"[DefectMapper] 매핑 로드 완료: {len(self.mapping_data.get('products', {}))}개 제품")
        
        # 제품 목록 출력
        for product_id in self.mapping_data.get('products', {}).keys():
            defect_count = len(self.mapping_data['products'][product_id].get('defects', {}))
            print(f"  - {product_id}: {defect_count}개 불량 유형")
    
    def get_defect_info(self, product: str, defect: str) -> Optional[DefectInfo]:
        """
        불량 정보 조회
        
        Args:
            product: 제품명 (예: "leather")
            defect: 불량명 (예: "fold")
        
        Returns:
            DefectInfo 또는 None
        """
        products = self.mapping_data.get('products', {})
        
        if product not in products:
            print(f"⚠️  제품을 찾을 수 없습니다: {product}")
            print(f"   사용 가능한 제품: {list(products.keys())}")
            return None
        
        defects = products[product].get('defects', {})
        
        if defect not in defects:
            print(f"⚠️  불량을 찾을 수 없습니다: {product}/{defect}")
            print(f"   사용 가능한 불량: {list(defects.keys())}")
            return None
        
        defect_data = defects[defect]
        
        return DefectInfo(
            en=defect_data.get('en', defect),
            ko=defect_data.get('ko', defect),
            full_name_ko=defect_data.get('full_name_ko', f"{defect} 불량"),
            keywords=defect_data.get('keywords', [defect])
        )
    
    def get_search_keywords(self, product: str, defect: str) -> List[str]:
        """검색 키워드 반환"""
        defect_info = self.get_defect_info(product, defect)
        
        if defect_info:
            return defect_info.keywords
        
        return [defect]
    
    def get_available_products(self) -> List[str]:
        """사용 가능한 제품 목록"""
        return list(self.mapping_data.get('products', {}).keys())
    
    def get_available_defects(self, product: str) -> List[str]:
        """특정 제품의 사용 가능한 불량 목록"""
        products = self.mapping_data.get('products', {})
        
        if product not in products:
            return []
        
        return list(products[product].get('defects', {}).keys())
    
    def add_product(self, product_id: str, product_name_ko: str):
        """새 제품 추가"""
        if 'products' not in self.mapping_data:
            self.mapping_data['products'] = {}
        
        if product_id in self.mapping_data['products']:
            print(f"⚠️  제품이 이미 존재합니다: {product_id}")
            return
        
        self.mapping_data['products'][product_id] = {
            "name_ko": product_name_ko,
            "defects": {}
        }
        
        self._save()
        print(f"✅ 제품 추가 완료: {product_id} ({product_name_ko})")
    
    def add_defect(
        self,
        product: str,
        defect_id: str,
        defect_ko: str,
        full_name_ko: str,
        keywords: List[str] = None
    ):
        """특정 제품에 불량 추가"""
        products = self.mapping_data.get('products', {})
        
        if product not in products:
            print(f"⚠️  제품을 찾을 수 없습니다: {product}")
            return
        
        if 'defects' not in products[product]:
            products[product]['defects'] = {}
        
        if defect_id in products[product]['defects']:
            print(f"⚠️  불량이 이미 존재합니다: {product}/{defect_id}")
            return
        
        if keywords is None:
            keywords = [defect_id, defect_ko]
        
        products[product]['defects'][defect_id] = {
            "en": defect_id,
            "ko": defect_ko,
            "full_name_ko": full_name_ko,
            "keywords": keywords
        }
        
        self._save()
        print(f"✅ 불량 추가 완료: {product}/{defect_id} ({defect_ko})")
    
    def _save(self):
        """매핑 데이터 저장"""
        with open(self.mapping_file, 'w', encoding='utf-8') as f:
            json.dump(self.mapping_data, f, ensure_ascii=False, indent=2)


def create_default_mapping(mapping_file: Path):
    """기본 매핑 파일 생성"""
    mapping_file.parent.mkdir(parents=True, exist_ok=True)
    
    default_mapping = {
        "products": {
            "prod1": {
                "name_ko": "제품1",
                "defects": {
                    "hole": {
                        "en": "hole",
                        "ko": "구멍",
                        "full_name_ko": "구멍 불량",
                        "keywords": ["구멍", "홀", "천공", "penetration"]
                    },
                    "burr": {
                        "en": "burr",
                        "ko": "버",
                        "full_name_ko": "버 불량",
                        "keywords": ["버", "burr", "돌기", "protrusion"]
                    },
                    "scratch": {
                        "en": "scratch",
                        "ko": "스크래치",
                        "full_name_ko": "스크래치 불량",
                        "keywords": ["스크래치", "긁힘", "scratch", "abrasion"]
                    }
                }
            },
            "grid": {
                "name_ko": "그리드",
                "defects": {
                    "hole": {
                        "en": "hole",
                        "ko": "구멍",
                        "full_name_ko": "구멍 불량",
                        "keywords": ["구멍", "홀", "천공"]
                    },
                    "burr": {
                        "en": "burr",
                        "ko": "버",
                        "full_name_ko": "버 불량",
                        "keywords": ["버", "burr", "돌기"]
                    },
                    "scratch": {
                        "en": "scratch",
                        "ko": "스크래치",
                        "full_name_ko": "스크래치 불량",
                        "keywords": ["스크래치", "긁힘"]
                    }
                }
            },
            "carpet": {
                "name_ko": "카펫",
                "defects": {
                    "hole": {
                        "en": "hole",
                        "ko": "구멍",
                        "full_name_ko": "구멍 불량",
                        "keywords": ["구멍", "홀", "천공"]
                    },
                    "burr": {
                        "en": "burr",
                        "ko": "버",
                        "full_name_ko": "버 불량",
                        "keywords": ["버", "burr", "돌기"]
                    },
                    "scratch": {
                        "en": "scratch",
                        "ko": "스크래치",
                        "full_name_ko": "스크래치 불량",
                        "keywords": ["스크래치", "긁힘"]
                    },
                    "stain": {
                        "en": "stain",
                        "ko": "얼룩",
                        "full_name_ko": "얼룩 불량",
                        "keywords": ["얼룩", "오염", "stain", "contamination"]
                    }
                }
            },
            "leather": {
                "name_ko": "가죽",
                "defects": {
                    "hole": {
                        "en": "hole",
                        "ko": "구멍",
                        "full_name_ko": "구멍 불량",
                        "keywords": ["구멍", "홀", "천공"]
                    },
                    "burr": {
                        "en": "burr",
                        "ko": "버",
                        "full_name_ko": "버 불량",
                        "keywords": ["버", "burr", "돌기"]
                    },
                    "scratch": {
                        "en": "scratch",
                        "ko": "스크래치",
                        "full_name_ko": "스크래치 불량",
                        "keywords": ["스크래치", "긁힘"]
                    },
                    "fold": {
                        "en": "fold",
                        "ko": "주름",
                        "full_name_ko": "주름 불량",
                        "keywords": ["주름", "접힘", "fold", "wrinkle", "crease"]
                    },
                    "stain": {
                        "en": "stain",
                        "ko": "얼룩",
                        "full_name_ko": "얼룩 불량",
                        "keywords": ["얼룩", "오염", "stain"]
                    },
                    "color": {
                        "en": "color",
                        "ko": "색상",
                        "full_name_ko": "색상 불량",
                        "keywords": ["색상", "변색", "color", "discoloration"]
                    }
                }
            }
        }
    }
    
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(default_mapping, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 기본 매핑 파일 생성 완료: {mapping_file}")
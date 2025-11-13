"""
defect_mapping.json 관리 모듈

제안2 기반: en, ko만 저장하고 keywords는 코드에서 자동 확장
"""
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
import re
from docx import Document


class DefectMappingManager:
    """
    defect_mapping.json CRUD 관리자
    
    특징:
    - 제품 단위 CRUD (제품명 기준)
    - 매뉴얼 파일(DOCX)에서 자동 추출
    - 최소 정보(en, ko)만 저장
    """
    
    def __init__(self, mapping_file_path: Path, verbose: bool = True):
        """
        Args:
            mapping_file_path: defect_mapping.json 파일 경로
            verbose: 로그 출력 여부
        """
        self.mapping_file = Path(mapping_file_path)
        self.verbose = verbose
        
        # 파일이 없으면 빈 구조 생성
        if not self.mapping_file.exists():
            self._init_empty_mapping()
        
        self.mapping = self._load_mapping()
    
    def _init_empty_mapping(self):
        """빈 mapping 구조 초기화"""
        empty_mapping = {"products": {}}
        self.mapping_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.mapping_file, 'w', encoding='utf-8') as f:
            json.dump(empty_mapping, f, ensure_ascii=False, indent=2)
        
        if self.verbose:
            print(f"[MAPPING] 빈 mapping 파일 생성: {self.mapping_file}")
    
    def _load_mapping(self) -> Dict:
        """mapping 파일 로드"""
        try:
            with open(self.mapping_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            if self.verbose:
                print(f"[MAPPING] 로드 실패: {e}")
            return {"products": {}}
    
    def _save_mapping(self):
        """mapping 파일 저장"""
        try:
            with open(self.mapping_file, 'w', encoding='utf-8') as f:
                json.dump(self.mapping, f, ensure_ascii=False, indent=2)
            
            if self.verbose:
                print(f"[MAPPING] 저장 완료: {self.mapping_file}")
        except Exception as e:
            if self.verbose:
                print(f"[MAPPING] 저장 실패: {e}")
            raise
    
    def extract_defects_from_docx(
        self, 
        docx_path: Path
    ) -> List[Tuple[str, str]]:
        """
        DOCX 파일에서 불량 유형 추출
        
        Args:
            docx_path: DOCX 파일 경로
        
        Returns:
            [(en, ko), ...] 리스트
        """
        try:
            doc = Document(docx_path)
            full_text = '\n'.join([
                para.text.strip() 
                for para in doc.paragraphs 
                if para.text.strip()
            ])
            
            defects = []
            
            # 패턴 1: 1️⃣ hole (기공)
            # 패턴 2: 1️⃣ Bent Defect (휨·압흔 불량)
            patterns = [
                r'[0-9]️⃣\s*([a-zA-Z]+)\s*[\(（]\s*([가-힣·,/\s]+)\s*[\)）]',
                r'[0-9]️⃣\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+Defect\s*[\(（]\s*([가-힣·,/\s]+)\s*[\)）]',
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, full_text, re.IGNORECASE)
                for en_part, ko_part in matches:
                    # 영문 정리
                    en_name = en_part.strip().lower().replace('defect', '').strip()
                    
                    # 한글 정리 (불량 제거, 쉼표 앞부분만)
                    ko_name = ko_part.replace('불량', '').strip()
                    if ',' in ko_name:
                        ko_name = ko_name.split(',')[0].strip()
                    
                    if en_name and ko_name:
                        defects.append((en_name, ko_name))
            
            # 중복 제거
            defects = list(dict.fromkeys(defects))
            
            if self.verbose:
                print(f"[EXTRACT] {docx_path.name}에서 {len(defects)}개 불량 추출")
                for en, ko in defects:
                    print(f"  - {en}: {ko}")
            
            return defects
            
        except Exception as e:
            if self.verbose:
                print(f"[EXTRACT] 실패: {e}")
            return []
    
    def create_product(
        self, 
        product_id: str, 
        product_name_ko: str,
        manual_docx_path: Optional[Path] = None
    ) -> bool:
        """
        신규 제품 생성
        
        Args:
            product_id: 제품 ID (예: prod1, grid)
            product_name_ko: 제품 한글명
            manual_docx_path: 매뉴얼 DOCX 파일 경로 (선택)
        
        Returns:
            성공 여부
        """
        if product_id in self.mapping["products"]:
            if self.verbose:
                print(f"[CREATE] 제품이 이미 존재합니다: {product_id}")
            return False
        
        # 제품 기본 구조 생성
        self.mapping["products"][product_id] = {
            "name_ko": product_name_ko,
            "defects": {}
        }
        
        # 매뉴얼 파일이 있으면 불량 자동 추출
        if manual_docx_path and manual_docx_path.exists():
            defects = self.extract_defects_from_docx(manual_docx_path)
            
            for en, ko in defects:
                self.mapping["products"][product_id]["defects"][en] = {
                    "en": en,
                    "ko": ko
                }
        
        self._save_mapping()
        
        if self.verbose:
            defect_count = len(self.mapping["products"][product_id]["defects"])
            print(f"[CREATE] 제품 생성 완료: {product_id} ({defect_count}개 불량)")
        
        return True
    
    def update_product(
        self, 
        product_id: str, 
        product_name_ko: Optional[str] = None,
        manual_docx_path: Optional[Path] = None,
        merge_defects: bool = True
    ) -> bool:
        """
        기존 제품 업데이트
        
        Args:
            product_id: 제품 ID
            product_name_ko: 새로운 제품명 (선택)
            manual_docx_path: 매뉴얼 DOCX 파일 (선택)
            merge_defects: True=기존 불량 유지하고 새 불량 추가, 
                          False=기존 불량 삭제하고 완전 교체
        
        Returns:
            성공 여부
        """
        if product_id not in self.mapping["products"]:
            if self.verbose:
                print(f"[UPDATE] 제품을 찾을 수 없습니다: {product_id}")
            return False
        
        # 제품명 업데이트
        if product_name_ko:
            self.mapping["products"][product_id]["name_ko"] = product_name_ko
        
        # 매뉴얼 파일이 있으면 불량 업데이트
        if manual_docx_path and manual_docx_path.exists():
            new_defects = self.extract_defects_from_docx(manual_docx_path)
            
            if not merge_defects:
                # 완전 교체
                self.mapping["products"][product_id]["defects"] = {}
            
            # 새 불량 추가/업데이트
            for en, ko in new_defects:
                self.mapping["products"][product_id]["defects"][en] = {
                    "en": en,
                    "ko": ko
                }
        
        self._save_mapping()
        
        if self.verbose:
            defect_count = len(self.mapping["products"][product_id]["defects"])
            print(f"[UPDATE] 제품 업데이트 완료: {product_id} ({defect_count}개 불량)")
        
        return True
    
    def delete_product(self, product_id: str) -> bool:
        """
        제품 삭제
        
        Args:
            product_id: 제품 ID
        
        Returns:
            성공 여부
        """
        if product_id not in self.mapping["products"]:
            if self.verbose:
                print(f"[DELETE] 제품을 찾을 수 없습니다: {product_id}")
            return False
        
        del self.mapping["products"][product_id]
        self._save_mapping()
        
        if self.verbose:
            print(f"[DELETE] 제품 삭제 완료: {product_id}")
        
        return True
    
    def get_product(self, product_id: str) -> Optional[Dict]:
        """
        제품 정보 조회
        
        Args:
            product_id: 제품 ID
        
        Returns:
            제품 정보 또는 None
        """
        return self.mapping["products"].get(product_id)
    
    def list_products(self) -> List[str]:
        """
        전체 제품 목록
        
        Returns:
            제품 ID 리스트
        """
        return list(self.mapping["products"].keys())
    
    def add_defect(
        self, 
        product_id: str, 
        defect_en: str, 
        defect_ko: str
    ) -> bool:
        """
        제품에 불량 추가
        
        Args:
            product_id: 제품 ID
            defect_en: 불량 영문명
            defect_ko: 불량 한글명
        
        Returns:
            성공 여부
        """
        if product_id not in self.mapping["products"]:
            if self.verbose:
                print(f"[ADD_DEFECT] 제품을 찾을 수 없습니다: {product_id}")
            return False
        
        self.mapping["products"][product_id]["defects"][defect_en] = {
            "en": defect_en,
            "ko": defect_ko
        }
        
        self._save_mapping()
        
        if self.verbose:
            print(f"[ADD_DEFECT] 불량 추가 완료: {product_id}/{defect_en}")
        
        return True
    
    def update_defect(
        self, 
        product_id: str, 
        defect_en: str, 
        defect_ko: str
    ) -> bool:
        """
        불량 정보 수정
        
        Args:
            product_id: 제품 ID
            defect_en: 불량 영문명
            defect_ko: 새로운 한글명
        
        Returns:
            성공 여부
        """
        if product_id not in self.mapping["products"]:
            if self.verbose:
                print(f"[UPDATE_DEFECT] 제품을 찾을 수 없습니다: {product_id}")
            return False
        
        if defect_en not in self.mapping["products"][product_id]["defects"]:
            if self.verbose:
                print(f"[UPDATE_DEFECT] 불량을 찾을 수 없습니다: {defect_en}")
            return False
        
        self.mapping["products"][product_id]["defects"][defect_en]["ko"] = defect_ko
        
        self._save_mapping()
        
        if self.verbose:
            print(f"[UPDATE_DEFECT] 불량 수정 완료: {product_id}/{defect_en}")
        
        return True
    
    def delete_defect(
        self, 
        product_id: str, 
        defect_en: str
    ) -> bool:
        """
        불량 삭제
        
        Args:
            product_id: 제품 ID
            defect_en: 불량 영문명
        
        Returns:
            성공 여부
        """
        if product_id not in self.mapping["products"]:
            if self.verbose:
                print(f"[DELETE_DEFECT] 제품을 찾을 수 없습니다: {product_id}")
            return False
        
        if defect_en not in self.mapping["products"][product_id]["defects"]:
            if self.verbose:
                print(f"[DELETE_DEFECT] 불량을 찾을 수 없습니다: {defect_en}")
            return False
        
        del self.mapping["products"][product_id]["defects"][defect_en]
        
        self._save_mapping()
        
        if self.verbose:
            print(f"[DELETE_DEFECT] 불량 삭제 완료: {product_id}/{defect_en}")
        
        return True
    
    def batch_create_from_directory(
        self, 
        manual_dir: Path,
        product_name_mapping: Optional[Dict[str, str]] = None
    ) -> int:
        """
        디렉토리의 모든 DOCX 파일에서 제품 일괄 생성
        
        Args:
            manual_dir: 매뉴얼 파일이 있는 디렉토리
            product_name_mapping: {파일명: 한글명} 매핑 (선택)
                예: {"prod1_menual.docx": "주조제품"}
        
        Returns:
            생성된 제품 수
        """
        if not manual_dir.exists():
            if self.verbose:
                print(f"[BATCH] 디렉토리를 찾을 수 없습니다: {manual_dir}")
            return 0
        
        docx_files = list(manual_dir.glob("*.docx"))
        
        if not docx_files:
            if self.verbose:
                print(f"[BATCH] DOCX 파일을 찾을 수 없습니다: {manual_dir}")
            return 0
        
        created_count = 0
        
        for docx_file in docx_files:
            # 파일명에서 제품 ID 추출 (예: prod1_menual.docx -> prod1)
            product_id = docx_file.stem.split('_')[0]
            
            # 한글명 결정
            if product_name_mapping and docx_file.name in product_name_mapping:
                product_name_ko = product_name_mapping[docx_file.name]
            else:
                product_name_ko = product_id
            
            # 제품 생성 (이미 존재하면 스킵)
            if product_id not in self.mapping["products"]:
                success = self.create_product(
                    product_id=product_id,
                    product_name_ko=product_name_ko,
                    manual_docx_path=docx_file
                )
                
                if success:
                    created_count += 1
        
        if self.verbose:
            print(f"[BATCH] {created_count}개 제품 생성 완료")
        
        return created_count
    
    def print_summary(self):
        """전체 매핑 정보 출력"""
        print("\n" + "="*70)
        print("defect_mapping.json 현황")
        print("="*70)
        
        if not self.mapping["products"]:
            print("등록된 제품이 없습니다.")
            return
        
        for product_id, product_data in self.mapping["products"].items():
            defect_count = len(product_data["defects"])
            print(f"\n📦 {product_id} ({product_data['name_ko']}): {defect_count}개 불량")
            
            for defect_en, defect_data in product_data["defects"].items():
                print(f"  - {defect_en}: {defect_data['ko']}")
        
        print("\n" + "="*70)


if __name__ == "__main__":
    """사용 예시"""
    
    # 매니저 초기화
    manager = DefectMappingManager(
        mapping_file_path=Path("/tmp/defect_mapping_test.json"),
        verbose=True
    )
    
    # 예시 1: 단일 제품 생성
    print("\n[예시 1] 단일 제품 생성")
    manager.create_product(
        product_id="test_product",
        product_name_ko="테스트제품"
    )
    
    # 예시 2: 매뉴얼에서 자동 추출
    print("\n[예시 2] 매뉴얼에서 자동 추출")
    manual_path = Path("/mnt/user-data/uploads/prod1_menual.docx")
    if manual_path.exists():
        manager.create_product(
            product_id="prod1",
            product_name_ko="주조제품",
            manual_docx_path=manual_path
        )
    
    # 예시 3: 불량 추가
    print("\n[예시 3] 불량 추가")
    manager.add_defect(
        product_id="test_product",
        defect_en="crack",
        defect_ko="균열"
    )
    
    # 예시 4: 제품 조회
    print("\n[예시 4] 제품 조회")
    product_info = manager.get_product("test_product")
    print(json.dumps(product_info, ensure_ascii=False, indent=2))
    
    # 예시 5: 전체 목록
    print("\n[예시 5] 전체 제품 목록")
    products = manager.list_products()
    print(f"등록된 제품: {products}")
    
    # 예시 6: 요약 출력
    print("\n[예시 6] 요약 출력")
    manager.print_summary()
    
    # 예시 7: 제품 삭제
    print("\n[예시 7] 제품 삭제")
    manager.delete_product("test_product")
    
    manager.print_summary()
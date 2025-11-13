"""
RAG 매니저 - 통합 매뉴얼 검색 (메타데이터 필터링)
"""
from pathlib import Path
from typing import List, Dict, Optional
import re
import json
import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


class UnifiedRAGManager:
    """
    통합 RAG 매니저 (메타데이터 기반 필터링)
    
    특징:
    - 모든 매뉴얼을 한번에 임베딩
    - 메타데이터로 제품별 필터링
    - defect_mapping.json 기반 동적 불량 처리
    - 캐싱으로 빠른 로딩
    """
    
    def __init__(
        self,
        manual_dir: Path,
        vector_store_path: Path,
        defect_mapping_path: Optional[Path] = None,
        device: str = "cuda",
        force_rebuild: bool = False,
        verbose: bool = True
    ):
        self.manual_dir = Path(manual_dir)
        self.vector_store_path = Path(vector_store_path)
        self.device = device
        self.verbose = verbose
        
        # defect_mapping.json 로드
        self.defect_mapping = {}
        if defect_mapping_path and defect_mapping_path.exists():
            with open(defect_mapping_path, 'r', encoding='utf-8') as f:
                mapping_data = json.load(f)
                self.defect_mapping = mapping_data.get("products", {})
            
            if self.verbose:
                print(f"[RAG] defect_mapping.json 로드 완료: {len(self.defect_mapping)}개 제품")
        
        # 임베딩 모델 초기화
        if self.verbose:
            print("[RAG] 임베딩 모델 로딩 중...")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="jhgan/ko-sbert-nli",
            model_kwargs={'device': self.device}
        )
        
        # 벡터 스토어 초기화
        self.vectorstore: Optional[FAISS] = None
        self.file_metadata: Dict[str, Dict] = {}
        
        # 벡터 스토어 로드 또는 구축
        index_path = self.vector_store_path / "unified_index"
        metadata_path = self.vector_store_path / "file_metadata.json"
        
        if index_path.exists() and metadata_path.exists() and not force_rebuild:
            # 기존 인덱스 로드
            self._load_index(index_path, metadata_path)
        else:
            # 새로 구축
            self._build_index()
    
    def _build_index(self):
        """모든 매뉴얼을 통합 임베딩"""
        if self.verbose:
            print("\n" + "="*70)
            print("통합 RAG 인덱스 구축 중...")
            print("="*70)
        
        # PDF 파일 수집
        pdf_files = list(self.manual_dir.glob("*.pdf"))
        
        if not pdf_files:
            raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {self.manual_dir}")
        
        if self.verbose:
            print(f"\n발견된 매뉴얼: {len(pdf_files)}개")
            for pdf in pdf_files:
                print(f"  - {pdf.name}")
        
        all_documents = []
        self.file_metadata = {}
        
        # 각 PDF 처리
        for pdf_file in pdf_files:
            try:
                # 제품명 추출 (파일명에서)
                product_name = pdf_file.stem.split("_")[0]
                
                if self.verbose:
                    print(f"\n처리 중: {pdf_file.name} (제품: {product_name})")
                
                # PDF 로드
                loader = PyPDFLoader(str(pdf_file))
                docs = loader.load()
                
                if self.verbose:
                    print(f"  페이지 수: {len(docs)}")
                
                # 메타데이터 추가
                for i, doc in enumerate(docs):
                    doc.metadata.update({
                        "product": product_name,
                        "source_file": pdf_file.name,
                        "page": i + 1
                    })
                
                all_documents.extend(docs)
                
                # 파일 메타데이터 저장
                self.file_metadata[pdf_file.name] = {
                    "product": product_name,
                    "page_count": len(docs),
                    "file_path": str(pdf_file)
                }
                
            except Exception as e:
                print(f"⚠️  {pdf_file.name} 처리 실패: {e}")
                continue
        
        if not all_documents:
            raise ValueError("처리된 문서가 없습니다")
        
        if self.verbose:
            print(f"\n총 문서 수: {len(all_documents)}")
            print("\n텍스트 분할 중...")
        
        # 텍스트 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", ".", " ", ""]
        )
        
        chunks = text_splitter.split_documents(all_documents)
        
        if self.verbose:
            print(f"총 청크 수: {len(chunks)}")
            print("\n벡터 임베딩 생성 중... (시간이 다소 걸릴 수 있습니다)")
        
        # 벡터 스토어 구축
        self.vectorstore = FAISS.from_documents(
            chunks,
            self.embeddings
        )
        
        # 저장
        self._save_index()
        
        if self.verbose:
            print("\n" + "="*70)
            print("✅ 통합 RAG 인덱스 구축 완료")
            print("="*70)
            print(f"제품 수: {len(self.file_metadata)}")
            print(f"총 청크 수: {len(chunks)}")
            print()
    
    def _save_index(self):
        """인덱스 저장"""
        self.vector_store_path.mkdir(parents=True, exist_ok=True)
        
        # FAISS 인덱스 저장
        index_path = self.vector_store_path / "unified_index"
        self.vectorstore.save_local(str(index_path))
        
        # 메타데이터 저장
        metadata_path = self.vector_store_path / "file_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.file_metadata, f, ensure_ascii=False, indent=2)
        
        if self.verbose:
            print(f"인덱스 저장 완료: {index_path}")
    
    def _load_index(self, index_path: Path, metadata_path: Path):
        """기존 인덱스 로드"""
        if self.verbose:
            print(f"[RAG] 기존 인덱스 로드 중: {index_path}")
        
        try:
            # FAISS 인덱스 로드
            self.vectorstore = FAISS.load_local(
                str(index_path),
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            # 메타데이터 로드
            with open(metadata_path, "r", encoding="utf-8") as f:
                self.file_metadata = json.load(f)
            
            if self.verbose:
                print(f"[RAG] 로드 완료: {len(self.file_metadata)}개 제품 매뉴얼")
                for filename, meta in self.file_metadata.items():
                    print(f"  - {filename} (제품: {meta['product']})")
        
        except Exception as e:
            print(f"⚠️  인덱스 로드 실패: {e}")
            print("새로 구축합니다...")
            self._build_index()
    
    def _get_all_defect_types(self, product: str) -> List[str]:
        """
        제품의 모든 불량 유형 가져오기 (defect_mapping.json 기반)
        
        Args:
            product: 제품명
        
        Returns:
            불량 유형 리스트 (영문)
        """
        if product not in self.defect_mapping:
            return []
        
        return list(self.defect_mapping[product].get("defects", {}).keys())
    
    def _build_defect_pattern(self, product: str, defect: str) -> Optional[str]:
        """
        동적으로 불량 섹션 패턴 생성 (defect_mapping.json 기반)
        
        Args:
            product: 제품명
            defect: 현재 불량명
        
        Returns:
            정규식 패턴 문자열
        """
        all_defects = self._get_all_defect_types(product)
        
        if not all_defects:
            return None
        
        # 현재 불량 제외한 다른 불량들
        other_defects = [d for d in all_defects if d != defect]
        
        if not other_defects:
            # 다른 불량이 없으면 문서 끝까지
            pattern = rf'{defect}\s*[\(（][^)）]+[\)）](.*?)$'
        else:
            # 다른 불량명들을 OR 조건으로
            other_pattern = '|'.join(other_defects)
            pattern = rf'{defect}\s*[\(（][^)）]+[\)）](.*?)(?=(?:{other_pattern})[\(（]|$)'
        
        return pattern
    
    def _extract_structured_sections(
        self,
        text: str,
        product: str,
        defect: str
    ) -> Dict[str, List[str]]:
        """
        텍스트에서 원인/조치 섹션을 구조화하여 추출
        
        Args:
            text: 검색된 텍스트
            product: 제품명
            defect: 불량명 (예: hole, burr, scratch)
        
        Returns:
            {"원인": [...], "조치": [...]}
        """
        results = {"원인": [], "조치": []}
        
        # defect_mapping.json 기반 동적 패턴 생성
        defect_pattern = self._build_defect_pattern(product, defect)
        
        if defect_pattern:
            match = re.search(defect_pattern, text, re.IGNORECASE | re.DOTALL)
            
            if match:
                section = match.group(1)
            else:
                # 패턴 매칭 실패 시 불량명이 포함된 단락 추출
                if defect.lower() not in text.lower():
                    return results
                
                paragraphs = text.split('\n\n')
                section = '\n\n'.join([p for p in paragraphs if defect.lower() in p.lower()])
        else:
            # defect_mapping에 없는 경우 전체 텍스트 사용
            if defect.lower() not in text.lower():
                return results
            
            paragraphs = text.split('\n\n')
            section = '\n\n'.join([p for p in paragraphs if defect.lower() in p.lower()])
        
        if not section.strip():
            return results
        
        # 원인 추출
        cause_patterns = [
            r'발생\s*원인(.*?)(?:조치\s*가이드|조치\s*방법|대응\s*방법|$)',
            r'원인(.*?)(?:조치|대응|해결|$)',
            r'발생\s*이유(.*?)(?:조치|대응|해결|$)'
        ]
        
        for pattern in cause_patterns:
            cause_match = re.search(pattern, section, re.DOTALL | re.IGNORECASE)
            if cause_match:
                cause_text = cause_match.group(1)
                # 불릿 포인트 추출
                causes = [
                    line.strip().lstrip('•-*').strip()
                    for line in cause_text.split('\n')
                    if line.strip() and (
                        line.strip().startswith('•') or 
                        line.strip().startswith('-') or 
                        line.strip().startswith('*') or
                        line.strip().startswith('・')
                    )
                ]
                if causes:
                    results["원인"].extend(causes[:3])
                    break
        
        # 원인을 찾지 못한 경우, "원인" 키워드가 포함된 문장 추출
        if not results["원인"]:
            lines = section.split('\n')
            for line in lines:
                if any(keyword in line for keyword in ["원인", "발생", "이유", "때문"]):
                    cleaned = line.strip().lstrip('•-*・').strip()
                    if cleaned and len(cleaned) > 10:
                        results["원인"].append(cleaned)
                        if len(results["원인"]) >= 3:
                            break
        
        # 조치 추출
        action_patterns = [
            r'조치\s*가이드(.*?)(?:요약|참고|$)',
            r'조치\s*방법(.*?)(?:요약|참고|$)',
            r'대응\s*방법(.*?)(?:요약|참고|$)',
            r'해결\s*방법(.*?)(?:요약|참고|$)'
        ]
        
        for pattern in action_patterns:
            action_match = re.search(pattern, section, re.DOTALL | re.IGNORECASE)
            if action_match:
                action_text = action_match.group(1)
                # 불릿 포인트 추출
                actions = [
                    line.strip().lstrip('•-*').strip()
                    for line in action_text.split('\n')
                    if line.strip() and (
                        line.strip().startswith('•') or 
                        line.strip().startswith('-') or 
                        line.strip().startswith('*') or
                        line.strip().startswith('・')
                    )
                ]
                if actions:
                    results["조치"].extend(actions[:3])
                    break
        
        # 조치를 찾지 못한 경우, "조치" 키워드가 포함된 문장 추출
        if not results["조치"]:
            lines = section.split('\n')
            for line in lines:
                if any(keyword in line for keyword in ["조치", "대응", "해결", "방법", "수정"]):
                    cleaned = line.strip().lstrip('•-*・').strip()
                    if cleaned and len(cleaned) > 10:
                        results["조치"].append(cleaned)
                        if len(results["조치"]) >= 3:
                            break
        
        return results
    
    def search_by_product(
        self,
        product: str,
        query: str,
        k: int = 3,
        include_other_products: bool = True
    ) -> Dict[str, List[str]]:
        """
        제품별 매뉴얼 검색
        
        Args:
            product: 제품명
            query: 검색 쿼리
            k: 반환할 결과 수
            include_other_products: 다른 제품 매뉴얼도 포함할지 여부
        
        Returns:
            {"원인": [...], "조치": [...]}
        """
        if self.vectorstore is None:
            raise RuntimeError("벡터 스토어가 초기화되지 않았습니다")
        
        results = {"원인": [], "조치": []}
        
        # 1단계: 동일 제품 매뉴얼 우선 검색
        try:
            primary_docs = self.vectorstore.similarity_search(
                query,
                k=k * 3,  # 더 많이 검색하여 필터링
                filter={"product": product}
            )
            
            if self.verbose:
                print(f"[RAG] {product} 제품 매뉴얼 검색: {len(primary_docs)}개 결과")
            
        except Exception as e:
            # 필터링 실패 시 전체 검색
            if self.verbose:
                print(f"[RAG] 필터링 검색 실패, 전체 검색으로 전환: {e}")
            primary_docs = self.vectorstore.similarity_search(query, k=k * 3)
        
        # 디버깅: 검색된 텍스트 샘플 출력
        if self.verbose and primary_docs:
            print(f"[DEBUG] 검색된 텍스트 샘플 (상위 3개):")
            for i, doc in enumerate(primary_docs[:3]):
                content_preview = doc.page_content[:150].replace('\n', ' ')
                print(f"  [{i}] {content_preview}...")
        
        # 2단계: 결과가 부족하면 다른 제품 매뉴얼 보조 검색
        if include_other_products and len(primary_docs) < k * 2:
            additional_needed = k * 2 - len(primary_docs)
            
            try:
                secondary_docs = self.vectorstore.similarity_search(
                    query,
                    k=additional_needed * 2
                )
                
                # 중복 제거 및 다른 제품만 추가
                for doc in secondary_docs:
                    if doc not in primary_docs and doc.metadata.get("product") != product:
                        primary_docs.append(doc)
                        if len(primary_docs) >= k * 3:
                            break
                
                if self.verbose and len(secondary_docs) > 0:
                    print(f"[RAG] 보조 검색: {len(secondary_docs)}개 추가")
            
            except Exception as e:
                if self.verbose:
                    print(f"[RAG] 보조 검색 실패: {e}")
        
        # 3단계: 전체 텍스트 결합 및 구조화 추출
        if primary_docs:
            full_text = "\n\n".join([doc.page_content for doc in primary_docs])
            
            # 검색 쿼리에서 불량명 추출
            defect = query.split()[0] if query else ""
            
            # 구조화된 섹션 추출 (defect_mapping 기반)
            structured = self._extract_structured_sections(full_text, product, defect)
            results["원인"] = structured["원인"][:k]
            results["조치"] = structured["조치"][:k]
        
        # 4단계: 여전히 결과가 없으면 키워드 기반 단순 추출
        if not results["원인"] and not results["조치"]:
            if self.verbose:
                print("[RAG] 구조화 추출 실패, 키워드 기반 추출 시도")
            
            for doc in primary_docs[:k * 2]:
                content = doc.page_content
                
                # 원인 키워드
                if len(results["원인"]) < k and any(kw in content for kw in ["원인", "발생", "이유", "때문"]):
                    results["원인"].append(content[:200])
                
                # 조치 키워드
                if len(results["조치"]) < k and any(kw in content for kw in ["조치", "대응", "해결", "방법", "수정"]):
                    results["조치"].append(content[:200])
        
        if self.verbose:
            print(f"[RAG] 최종 결과: 원인 {len(results['원인'])}개, 조치 {len(results['조치'])}개")
        
        return results
    
    def search_defect_manual(
        self,
        product: str,
        defect: str,
        keywords: List[str] = None,
        k: int = 3
    ) -> Dict[str, List[str]]:
        """
        불량 매뉴얼 검색 (제품 + 불량 기반)
        
        Args:
            product: 제품명
            defect: 불량명
            keywords: 추가 검색 키워드
            k: 반환할 결과 수
        
        Returns:
            {"원인": [...], "조치": [...]}
        """
        # 검색 쿼리 생성
        query_parts = [defect]
        
        if keywords:
            query_parts.extend(keywords)
        
        query = " ".join(query_parts)
        
        if self.verbose:
            print(f"[RAG] 검색 쿼리: '{query}' (제품: {product})")
        
        return self.search_by_product(
            product=product,
            query=query,
            k=k,
            include_other_products=True
        )
    
    def get_available_products(self) -> List[str]:
        """사용 가능한 제품 목록 반환"""
        return list(set(meta["product"] for meta in self.file_metadata.values()))
    
    def rebuild_index(self):
        """인덱스 강제 재구축"""
        if self.verbose:
            print("[RAG] 인덱스 강제 재구축 시작...")
        
        self._build_index()


def create_rag_manager(
    manual_dir: Path,
    vector_store_path: Path,
    defect_mapping_path: Optional[Path] = None,
    device: str = "cuda",
    force_rebuild: bool = False,
    verbose: bool = True
) -> UnifiedRAGManager:
    """
    RAG 매니저 생성 헬퍼 함수
    
    Args:
        manual_dir: 매뉴얼 디렉토리
        vector_store_path: 벡터 스토어 저장 경로
        defect_mapping_path: defect_mapping.json 경로
        device: 디바이스
        force_rebuild: 강제 재구축 여부
        verbose: 로그 출력 여부
    
    Returns:
        UnifiedRAGManager 인스턴스
    """
    return UnifiedRAGManager(
        manual_dir=manual_dir,
        vector_store_path=vector_store_path,
        defect_mapping_path=defect_mapping_path,
        device=device,
        force_rebuild=force_rebuild,
        verbose=verbose
    )
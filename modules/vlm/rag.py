"""
RAG 매니저 - 통합 매뉴얼 검색 (메타데이터 필터링)
"""
from pathlib import Path
from typing import List, Dict, Optional
import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import json


class UnifiedRAGManager:
    """
    통합 RAG 매니저 (메타데이터 기반 필터링)
    
    특징:
    - 모든 매뉴얼을 한번에 임베딩
    - 메타데이터로 제품별 필터링
    - 캐싱으로 빠른 로딩
    """
    
    def __init__(
        self,
        manual_dir: Path,
        vector_store_path: Path,
        device: str = "cuda",
        force_rebuild: bool = False,
        verbose: bool = True
    ):
        self.manual_dir = Path(manual_dir)
        self.vector_store_path = Path(vector_store_path)
        self.device = device
        self.verbose = verbose
        
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
                k=k * 2,  # 여유있게 검색
                filter={"product": product}
            )
            
            if self.verbose:
                print(f"[RAG] {product} 제품 매뉴얼 검색: {len(primary_docs)}개 결과")
            
        except Exception as e:
            # 필터링 실패 시 전체 검색
            if self.verbose:
                print(f"[RAG] 필터링 검색 실패, 전체 검색으로 전환: {e}")
            primary_docs = self.vectorstore.similarity_search(query, k=k)
        
        # 2단계: 결과가 부족하면 다른 제품 매뉴얼 보조 검색
        if include_other_products and len(primary_docs) < k:
            additional_needed = k - len(primary_docs)
            
            try:
                secondary_docs = self.vectorstore.similarity_search(
                    query,
                    k=additional_needed * 2
                )
                
                # 중복 제거 및 다른 제품만 추가
                for doc in secondary_docs:
                    if doc not in primary_docs and doc.metadata.get("product") != product:
                        primary_docs.append(doc)
                        if len(primary_docs) >= k * 2:
                            break
                
                if self.verbose and len(secondary_docs) > 0:
                    print(f"[RAG] 보조 검색: {len(secondary_docs)}개 추가")
            
            except Exception as e:
                if self.verbose:
                    print(f"[RAG] 보조 검색 실패: {e}")
        
        # 3단계: 원인/조치 분류
        for doc in primary_docs[:k * 2]:
            content = doc.page_content
            
            # 원인 키워드
            if any(keyword in content for keyword in ["원인", "발생", "이유", "때문"]):
                if len(results["원인"]) < k:
                    results["원인"].append(content)
            
            # 조치 키워드
            if any(keyword in content for keyword in ["조치", "대응", "해결", "방법", "수정"]):
                if len(results["조치"]) < k:
                    results["조치"].append(content)
        
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
        query_parts = [defect, "불량"]
        
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
    device: str = "cuda",
    force_rebuild: bool = False,
    verbose: bool = True
) -> UnifiedRAGManager:
    """
    RAG 매니저 생성 헬퍼 함수
    
    Args:
        manual_dir: 매뉴얼 디렉토리
        vector_store_path: 벡터 스토어 저장 경로
        device: 디바이스
        force_rebuild: 강제 재구축 여부
        verbose: 로그 출력 여부
    
    Returns:
        UnifiedRAGManager 인스턴스
    """
    return UnifiedRAGManager(
        manual_dir=manual_dir,
        vector_store_path=vector_store_path,
        device=device,
        force_rebuild=force_rebuild,
        verbose=verbose
    )
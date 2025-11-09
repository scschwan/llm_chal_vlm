"""
RAG (Retrieval-Augmented Generation) 매니저
LangChain 기반 PDF 매뉴얼 검색
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Optional
import re

# LangChain imports
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document


class RAGManager:
    """PDF 매뉴얼 RAG 관리자"""
    
    def __init__(
        self,
        pdf_path: str | Path,
        embedding_model: str = "jhgan/ko-sbert-nli",
        vector_store_path: Optional[str | Path] = None,
        device: str = "cuda",
        verbose: bool = True
    ):
        """
        Args:
            pdf_path: PDF 매뉴얼 경로
            embedding_model: 임베딩 모델명
            vector_store_path: 벡터 DB 캐시 경로 (None이면 매번 새로 구축)
            device: 디바이스 (cuda/cpu)
            verbose: 로그 출력 여부
        """
        self.pdf_path = Path(pdf_path)
        self.vector_store_path = Path(vector_store_path) if vector_store_path else None
        self.device = device
        self.verbose = verbose
        
        # 임베딩 모델 초기화
        if self.verbose:
            print(f"🔤 임베딩 모델 로드 중: {embedding_model}")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': self.device}
        )
        
        # 벡터 DB 로드 또는 구축
        self.vectorstore = self._load_or_build_vectorstore()
        
        if self.verbose:
            print("✅ RAG 매니저 초기화 완료")
    
    def _load_or_build_vectorstore(self) -> FAISS:
        """벡터 DB 로드 또는 신규 구축"""
        # 캐시 경로가 있고 존재하면 로드
        if self.vector_store_path and self.vector_store_path.exists():
            if self.verbose:
                print(f"📂 벡터 DB 로드 중: {self.vector_store_path}")
            
            return FAISS.load_local(
                str(self.vector_store_path),
                self.embeddings
            )
        
        # 신규 구축
        if self.verbose:
            print(f"📄 PDF 로드 중: {self.pdf_path}")
        
        documents = self._load_and_parse_pdf()
        
        if self.verbose:
            print(f"🔨 벡터 DB 구축 중... ({len(documents)}개 문서)")
        
        vectorstore = FAISS.from_documents(documents, self.embeddings)
        
        # 캐시 저장
        if self.vector_store_path:
            self.vector_store_path.parent.mkdir(parents=True, exist_ok=True)
            vectorstore.save_local(str(self.vector_store_path))
            if self.verbose:
                print(f"💾 벡터 DB 저장 완료: {self.vector_store_path}")
        
        return vectorstore
    
    def _load_and_parse_pdf(self) -> List[Document]:
        """
        PDF 로드 및 불량별 섹션 파싱
        
        각 불량을 독립된 Document로 생성하여
        검색 정확도 향상
        """
        # PDF 로드
        loader = PyPDFLoader(str(self.pdf_path))
        raw_docs = loader.load()
        
        # 전체 텍스트 결합
        full_text = "\n".join([doc.page_content for doc in raw_docs])
        
        # 불량별 섹션 분리
        sections = self._split_by_defect_sections(full_text)
        
        # Document 객체 생성
        documents = []
        for section in sections:
            # 메타데이터 추출
            defect_name = section.get("defect", "unknown")
            
            # 발생 원인 Document
            if section.get("cause"):
                documents.append(Document(
                    page_content=section["cause"],
                    metadata={
                        "defect": defect_name,
                        "section": "원인",
                        "source": str(self.pdf_path)
                    }
                ))
            
            # 조치 가이드 Document
            if section.get("action"):
                documents.append(Document(
                    page_content=section["action"],
                    metadata={
                        "defect": defect_name,
                        "section": "조치",
                        "source": str(self.pdf_path)
                    }
                ))
        
        return documents
    
    def _split_by_defect_sections(self, text: str) -> List[Dict[str, str]]:
        """
        텍스트를 불량별 섹션으로 분리
        
        패턴:
        ① hole (기공)
        발생 원인
        ...
        조치 가이드
        ...
        """
        sections = []
        
        # 불량 제목 패턴: ① hole (기공)
        defect_pattern = r'[①②③④⑤⑥⑦⑧⑨⑩]\s+(\w+)\s*\(([^)]+)\)'
        
        # 불량별로 분리
        defect_matches = list(re.finditer(defect_pattern, text))
        
        for i, match in enumerate(defect_matches):
            defect_en = match.group(1).strip()
            defect_ko = match.group(2).strip()
            
            # 섹션 시작/끝 위치
            start_pos = match.end()
            end_pos = defect_matches[i+1].start() if i+1 < len(defect_matches) else len(text)
            
            section_text = text[start_pos:end_pos]
            
            # "발생 원인"과 "조치 가이드" 분리
            cause_match = re.search(r'발생 원인(.*?)조치 가이드', section_text, re.DOTALL)
            action_match = re.search(r'조치 가이드(.*?)(?=(?:[①②③④⑤⑥⑦⑧⑨⑩]|$))', section_text, re.DOTALL)
            
            sections.append({
                "defect": defect_en,
                "defect_ko": defect_ko,
                "cause": cause_match.group(1).strip() if cause_match else "",
                "action": action_match.group(1).strip() if action_match else ""
            })
        
        return sections
    
    def search_defect_manual(
        self,
        product: str,
        defect_en: str,
        keywords: List[str],
        top_k: int = 3
    ) -> Dict[str, List[str]]:
        """
        불량 매뉴얼 검색
        
        Args:
            product: 제품명 (현재는 미사용, 추후 제품별 PDF 지원 시 활용)
            defect_en: 영어 불량명
            keywords: 검색 키워드 리스트
            top_k: 상위 K개 결과
        
        Returns:
            {"원인": [...], "조치": [...]}
        """
        # 키워드 조합 쿼리 생성
        query = " ".join(keywords)
        
        if self.verbose:
            print(f"🔍 매뉴얼 검색: {query}")
        
        # 벡터 검색
        results = self.vectorstore.similarity_search(
            query,
            k=top_k * 2  # 원인/조치 각각 필요하므로 더 많이 검색
        )
        
        # 섹션별 분리
        cause_docs = [
            doc.page_content for doc in results 
            if doc.metadata.get("section") == "원인"
        ]
        action_docs = [
            doc.page_content for doc in results 
            if doc.metadata.get("section") == "조치"
        ]
        
        return {
            "원인": cause_docs[:top_k],
            "조치": action_docs[:top_k]
        }
    
    def rebuild_index(self):
        """벡터 DB 재구축"""
        if self.verbose:
            print("🔄 벡터 DB 재구축 중...")
        
        documents = self._load_and_parse_pdf()
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        
        if self.vector_store_path:
            self.vectorstore.save_local(str(self.vector_store_path))
        
        if self.verbose:
            print("✅ 벡터 DB 재구축 완료")


if __name__ == "__main__":
    # 테스트
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent.parent
    pdf_path = project_root / "prod1_menual.pdf"
    vector_store_path = project_root / "web" / "vector_store"
    
    if not pdf_path.exists():
        print(f"❌ PDF 파일을 찾을 수 없습니다: {pdf_path}")
        exit(1)
    
    # RAG 매니저 초기화
    rag = RAGManager(
        pdf_path=pdf_path,
        vector_store_path=vector_store_path,
        verbose=True
    )
    
    # 검색 테스트
    print("\n=== 검색 테스트 ===")
    results = rag.search_defect_manual(
        product="prod1",
        defect_en="burr",
        keywords=["burr", "버", "날개"]
    )
    
    print("\n[발생 원인]")
    for i, cause in enumerate(results["원인"], 1):
        print(f"{i}. {cause[:100]}...")
    
    print("\n[조치 가이드]")
    for i, action in enumerate(results["조치"], 1):
        print(f"{i}. {action[:100]}...")
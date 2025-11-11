"""
RAG (Retrieval-Augmented Generation) 매니저
LangChain 기반 PDF 매뉴얼 검색
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Optional
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


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
        self.pdf_path = Path(pdf_path)
        self.vector_store_path = Path(vector_store_path) if vector_store_path else None
        self.device = device
        self.verbose = verbose
        
        if self.verbose:
            print(f"🔤 임베딩 모델 로드 중: {embedding_model}")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': self.device}
        )
        
        self.vectorstore = self._load_or_build_vectorstore()
        
        if self.verbose:
            print("✅ RAG 매니저 초기화 완료")
    
    def _load_or_build_vectorstore(self) -> FAISS:
        """벡터 스토어 로드 또는 구축"""
        if self.vector_store_path and self.vector_store_path.exists():
            index_file = self.vector_store_path / "index.faiss"
            if index_file.exists():
                if self.verbose:
                    print(f"📂 벡터 DB 로드 중: {self.vector_store_path}")
                
                try:
                    return FAISS.load_local(
                        str(self.vector_store_path),
                        self.embeddings,
                        allow_dangerous_deserialization=True
                    )
                except Exception as e:
                    if self.verbose:
                        print(f"⚠️  벡터 DB 로드 실패: {e}")
                        print("   새로 구축합니다...")
        
        if self.verbose:
            print(f"📚 PDF 문서 로드 중: {self.pdf_path}")
        
        loader = PyPDFLoader(str(self.pdf_path))
        documents = loader.load()
        
        if self.verbose:
            print(f"   로드된 페이지 수: {len(documents)}")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        texts = text_splitter.split_documents(documents)
        
        if self.verbose:
            print(f"   분할된 청크 수: {len(texts)}")
        
        if self.verbose:
            print("🔨 벡터 DB 구축 중...")
        
        vectorstore = FAISS.from_documents(texts, self.embeddings)
        
        if self.vector_store_path:
            self.vector_store_path.mkdir(parents=True, exist_ok=True)
            vectorstore.save_local(str(self.vector_store_path))
            
            if self.verbose:
                print(f"✅ 벡터 DB 저장 완료: {self.vector_store_path}")
        
        return vectorstore
    
    def search_defect_manual(
        self,
        product: str,
        defect_en: str,
        keywords: List[str],
        top_k: int = 3
    ) -> Dict[str, List[str]]:
        """불량 매뉴얼 검색 (원인/조치 분리)"""
        
        query = " ".join(keywords)
        if self.verbose:
            print(f"🔍 매뉴얼 검색: {query}")
        
        # 벡터 검색
        results = self.vectorstore.similarity_search(query, k=10)
        
        # 전체 텍스트 결합
        full_text = "\n\n".join([doc.page_content for doc in results])
        
        # 해당 불량 섹션 찾기
        defect_pattern = rf'{defect_en}\s*\([^)]+\)(.*?)(?=(?:burr|hole|scratch|Scratch)\s*\(|$)'
        match = re.search(defect_pattern, full_text, re.IGNORECASE | re.DOTALL)
        
        if not match:
            if self.verbose:
                print(f"   ⚠️ {defect_en} 섹션을 찾을 수 없음")
            return {"원인": [], "조치": []}
        
        section = match.group(1)
        
        # 원인 추출
        causes = []
        cause_match = re.search(r'발생\s*원인(.*?)조치\s*가이드', section, re.DOTALL)
        if cause_match:
            cause_text = cause_match.group(1)
            causes = [
                line.strip().lstrip('•-').strip()
                for line in cause_text.split('\n')
                if line.strip() and (line.strip().startswith('•') or line.strip().startswith('-'))
            ]
        
        # 조치 추출
        actions = []
        action_match = re.search(r'조치\s*가이드(.*?)(?:요약|$)', section, re.DOTALL)
        if action_match:
            action_text = action_match.group(1)
            actions = [
                line.strip().lstrip('•-').strip()
                for line in action_text.split('\n')
                if line.strip() and (line.strip().startswith('•') or line.strip().startswith('-'))
            ]
        
        causes = causes[:top_k]
        actions = actions[:top_k]
        
        if self.verbose:
            print(f"   추출: 원인 {len(causes)}개, 조치 {len(actions)}개")
        
        return {
            "원인": causes,
            "조치": actions
        }


if __name__ == "__main__":
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent.parent
    pdf_path = project_root / "manual_store" / "prod1_menual.pdf"
    vector_store_path = project_root / "manual_store"
    
    if not pdf_path.exists():
        print(f"❌ PDF 파일을 찾을 수 없습니다: {pdf_path}")
        exit(1)
    
    rag = RAGManager(
        pdf_path=pdf_path,
        vector_store_path=vector_store_path,
        verbose=True
    )
    
    print("\n=== 검색 테스트 ===")
    results = rag.search_defect_manual(
        product="prod1",
        defect_en="hole",
        keywords=["hole", "기공"]
    )
    
    print(f"\n[발생 원인] {len(results['원인'])}개")
    for i, cause in enumerate(results["원인"], 1):
        print(f"{i}. {cause}")
    
    print(f"\n[조치 가이드] {len(results['조치'])}개")
    for i, action in enumerate(results["조치"], 1):
        print(f"{i}. {action}")
"""
RAG (Retrieval-Augmented Generation) 매니저
LangChain 기반 PDF 매뉴얼 검색
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import re


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
        # 기존 벡터 DB 로드 시도
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
        
        # 새로 구축
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
        """불량 매뉴얼 검색 - 완전 재작성"""
        
        query = " ".join(keywords)
        
        if self.verbose:
            print(f"🔍 매뉴얼 검색: {query}")
        
        # 1. 벡터 검색으로 관련 청크 가져오기
        results = self.vectorstore.similarity_search(query, k=top_k * 3)
        
        if self.verbose:
            print(f"   검색된 청크: {len(results)}개")
        
        # 2. 불량별로 원인/조치 분리
        causes = []
        actions = []
        
        for doc in results:
            content = doc.page_content
            
            # 해당 불량(defect_en) 포함 여부 확인
            if defect_en.lower() not in content.lower():
                continue
            
            # "발생 원인" 섹션 추출
            if "발생 원인" in content:
                # "발생 원인"부터 "조치 가이드" 전까지
                
                cause_match = re.search(
                    r'발생 원인\s*(.*?)(?:조치 가이드|burr|Scratch|$)',
                    content,
                    re.DOTALL
                )
                if cause_match:
                    cause_text = cause_match.group(1).strip()
                    # 불릿 포인트만 추출
                    cause_lines = [
                        line.strip().lstrip('•').strip()
                        for line in cause_text.split('\n')
                        if line.strip().startswith('•')
                    ]
                    causes.extend(cause_lines)
            
            # "조치 가이드" 섹션 추출
            if "조치 가이드" in content or "조치" in content:
                action_match = re.search(
                    r'조치\s*가이드\s*(.*?)(?:burr|Scratch|발생 원인|$)',
                    content,
                    re.DOTALL
                )
                if action_match:
                    action_text = action_match.group(1).strip()
                    action_lines = [
                        line.strip().lstrip('•').strip()
                        for line in action_text.split('\n')
                        if line.strip().startswith('•')
                    ]
                    actions.extend(action_lines)
        
        # 3. 중복 제거 및 개수 제한
        causes = list(dict.fromkeys(causes))[:top_k]
        actions = list(dict.fromkeys(actions))[:top_k]
        
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
        defect_en="burr",
        keywords=["burr", "버", "날개"]
    )
    
    print(f"\n[발생 원인] {len(results['원인'])}개")
    for i, cause in enumerate(results["원인"], 1):
        print(f"{i}. {cause[:100]}...")
    
    print(f"\n[조치 가이드] {len(results['조치'])}개")
    for i, action in enumerate(results["조치"], 1):
        print(f"{i}. {action[:100]}...")
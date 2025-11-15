from sqlalchemy import Column, Integer, String, Text, TIMESTAMP, JSON, Float, Boolean, BigInteger, DateTime, JSON, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from .connection import Base

from datetime import datetime

class Product(Base):
    """제품 마스터"""
    __tablename__ = "products"
    
    product_id = Column(Integer, primary_key=True, autoincrement=True, comment="제품 ID")
    product_code = Column(String(50), nullable=False, comment="제품 코드")
    product_name = Column(String(100), nullable=False, comment="제품명")
    description = Column(Text, comment="제품 설명")
    is_active = Column(Integer, default=1, comment="활성 여부")
    created_at = Column(TIMESTAMP, server_default=func.now(), comment="생성일시")
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now(), comment="수정일시")


class Manual(Base):
    """매뉴얼"""
    __tablename__ = "manuals"
    
    manual_id = Column(Integer, primary_key=True, autoincrement=True, comment="매뉴얼 ID")
    product_id = Column(Integer, nullable=False, comment="제품 ID")
    file_name = Column(String(255), nullable=False, comment="파일명")
    file_path = Column(String(500), nullable=False, comment="파일 경로")
    file_size = Column(BigInteger, comment="파일 크기")
    vector_indexed = Column(Integer, default=0, comment="벡터 인덱싱 여부")
    indexed_at = Column(DateTime, comment="인덱싱 완료 일시")
    created_at = Column(TIMESTAMP, server_default=func.now(), comment="등록일시")


class DefectType(Base):
    """불량 유형"""
    __tablename__ = "defect_types"
    
    defect_type_id = Column(Integer, primary_key=True, autoincrement=True, comment="불량 유형 ID")
    product_id = Column(Integer, nullable=False, comment="제품 ID")
    defect_code = Column(String(50), nullable=False, comment="불량 코드")
    defect_name_ko = Column(String(100), nullable=False, comment="불량 명칭(한글)")
    defect_name_en = Column(String(100), comment="불량 명칭(영문)")
    full_name_ko = Column(String(200), comment="불량 전체 명칭")
    is_active = Column(Integer, default=1, comment="활성 여부")
    created_at = Column(TIMESTAMP, server_default=func.now(), comment="생성일시")


class Image(Base):
    """이미지 메타데이터"""
    __tablename__ = "images"
    
    image_id = Column(Integer, primary_key=True, autoincrement=True, comment="이미지 ID")
    product_id = Column(Integer, ForeignKey('products.product_id'), nullable=False, comment="제품 ID")
    image_type = Column(String(20), nullable=False, comment="이미지 유형")
    defect_type_id = Column(Integer, ForeignKey('defect_types.defect_type_id'), comment="불량 유형 ID")
    file_name = Column(String(255), nullable=False, comment="파일명")
    file_path = Column(String(500), nullable=False, comment="파일 경로")
    file_size = Column(BigInteger, comment="파일 크기")
    uploaded_at = Column(TIMESTAMP, server_default=func.now(), comment="업로드 일시")
    storage_url = Column(String(500), comment="Object Storage URL")
    
    # ✅ Relationship (ForeignKey 명시)
    product = relationship("Product", foreign_keys=[product_id])
    defect_type = relationship("DefectType", foreign_keys=[defect_type_id])


class ImagePreprocessing(Base):
    """이미지 전처리 설정"""
    __tablename__ = "image_preprocessing"
    
    preprocessing_id = Column(Integer, primary_key=True, autoincrement=True, comment="전처리 설정 ID")
    product_id = Column(Integer, nullable=False, comment="제품 ID")
    grayscale = Column(String(1), default='N', comment="그레이스케일")
    histogram = Column(String(1), default='N', comment="히스토그램 평활화")
    contrast = Column(String(1), default='N', comment="명암 대비 조정")
    smoothing = Column(String(1), default='N', comment="스무딩")
    normalize = Column(String(1), default='N', comment="정규화")
    is_active = Column(Integer, default=1, comment="활성 여부")
    created_at = Column(TIMESTAMP, server_default=func.now(), comment="생성일시")
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now(), comment="수정일시")


class SearchHistory(Base):
    """유사도 검색 이력"""
    __tablename__ = "search_history"
    
    search_id = Column(Integer, primary_key=True, autoincrement=True, comment="검색 ID")
    searched_at = Column(TIMESTAMP, server_default=func.now(), comment="검색 일시")
    uploaded_image_path = Column(String(500), nullable=False, comment="업로드 이미지 경로")
    product_code = Column(String(50), comment="제품 코드")
    defect_code = Column(String(50), comment="불량 코드")
    top_k_results = Column(JSON, nullable=False, comment="TOP-K 결과")
    processing_time = Column(Float, comment="처리 시간")


class ResponseHistory(Base):
    """대응 매뉴얼 생성 이력"""
    __tablename__ = "response_history"
    
    response_id = Column(Integer, primary_key=True, autoincrement=True, comment="응답 ID")
    executed_at = Column(TIMESTAMP, server_default=func.now(), comment="실행 일시")
    search_id = Column(Integer, comment="연관 검색 ID")
    product_code = Column(String(50), nullable=False, comment="제품 코드")
    defect_code = Column(String(50), nullable=False, comment="불량 코드")
    similarity_score = Column(Float, comment="유사도 점수")
    anomaly_score = Column(Float, comment="이상 점수")
    confidence_score = Column(Float, comment="신뢰도 점수")
    test_image_path = Column(String(500), comment="검사 이미지 경로")
    reference_image_path = Column(String(500), comment="기준 이미지 경로")
    heatmap_path = Column(String(500), comment="히트맵 이미지 경로")
    overlay_path = Column(String(500), comment="오버레이 이미지 경로")
    model_type = Column(String(50), comment="LLM 모델 타입")
    guide_content = Column(Text, comment="LLM 생성 대응 매뉴얼")
    guide_generated_at = Column(DateTime, comment="가이드 생성 일시")
    feedback_rating = Column(Integer, comment="사용자 평가")
    feedback_text = Column(Text, comment="피드백 내용")
    feedback_at = Column(DateTime, comment="피드백 작성 일시")
    processing_time = Column(Float, comment="처리 시간")


class ModelParams(Base):
    """모델 파라미터 설정"""
    __tablename__ = "model_params"
    
    param_id = Column(Integer, primary_key=True, autoincrement=True, comment="파라미터 ID")
    product_id = Column(Integer, nullable=False, comment="제품 ID")
    model_type = Column(String(50), nullable=False, comment="모델 타입")
    params = Column(JSON, nullable=False, comment="모델 파라미터")
    is_active = Column(Integer, default=1, comment="활성 여부")
    created_at = Column(TIMESTAMP, server_default=func.now(), comment="생성일시")
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now(), comment="수정일시")


class DeploymentLog(Base):
    """배포 실행 이력"""
    __tablename__ = "deployment_logs"
    
    deploy_id = Column(Integer, primary_key=True, autoincrement=True, comment="배포 ID")
    deploy_type = Column(String(50), nullable=False, comment="배포 타입")
    product_id = Column(Integer, comment="제품 ID")
    status = Column(String(20), default='pending', comment="상태")
    started_at = Column(TIMESTAMP, server_default=func.now(), comment="시작 일시")
    completed_at = Column(DateTime, comment="완료 일시")
    result_message = Column(Text, comment="결과 메시지")
    result_data = Column(JSON, comment="결과 상세")
    deployed_by = Column(String(50), comment="배포 실행자")


class PreprocessingConfig(Base):
    """전처리 설정 테이블"""
    __tablename__ = 'preprocessing_configs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    resize_width = Column(Integer, default=224)
    resize_height = Column(Integer, default=224)
    normalize = Column(Boolean, default=True)
    augmentation = Column(JSON, nullable=True)
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)


class SystemConfig(Base):
    """시스템 전역 설정"""
    __tablename__ = "system_config"
    
    config_key = Column(String(100), primary_key=True, comment="설정 키")
    config_value = Column(Text, nullable=False, comment="설정 값")
    description = Column(Text, comment="설명")
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now(), comment="수정일시")
    updated_by = Column(String(50), comment="수정자")
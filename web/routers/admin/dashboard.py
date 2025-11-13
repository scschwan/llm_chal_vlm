"""
관리자 대시보드 API 라우터
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import sys

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

router = APIRouter(prefix="/api/admin/dashboard", tags=["admin-dashboard"])


@router.get("/stats")
async def get_dashboard_stats():
    """
    대시보드 통계 조회
    """
    try:
        # TODO: 실제 DB에서 데이터 조회
        # 현재는 임시 데이터 반환
        
        # 제품 수 계산
        data_dir = project_root / "data" / "patchCore"
        total_products = len([d for d in data_dir.iterdir() if d.is_dir()]) if data_dir.exists() else 0
        
        # 정상 이미지 수 계산
        normal_images = 0
        if data_dir.exists():
            for prod_dir in data_dir.iterdir():
                if prod_dir.is_dir():
                    normal_images += len(list(prod_dir.glob("*.jpg"))) + len(list(prod_dir.glob("*.png")))
        
        # 불량 이미지 수 계산
        defect_dir = project_root / "data" / "def_split"
        defect_images = 0
        if defect_dir.exists():
            defect_images = len(list(defect_dir.glob("*.jpg"))) + len(list(defect_dir.glob("*.png")))
        
        return JSONResponse(content={
            "status": "success",
            "data": {
                "totalProducts": total_products,
                "totalNormalImages": normal_images,
                "totalDefectImages": defect_images,
                "todayInspections": 0,  # TODO: DB 연동
                "weeklyChange": {
                    "products": 0,
                    "normalImages": 0,
                    "defectImages": 0,
                    "inspections": 0
                }
            }
        })
        
    except Exception as e:
        print(f"[DASHBOARD] 통계 조회 실패: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"통계 조회 실패: {str(e)}")


@router.get("/inspections/recent")
async def get_recent_inspections(limit: int = 20):
    """
    최근 검사 내역 조회
    """
    try:
        # TODO: 실제 DB에서 데이터 조회
        # 현재는 임시 데이터 반환
        
        inspections = [
            {
                "id": 1,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "product": "prod1",
                "defect": "hole",
                "result": "normal",
                "worker": "작업자1",
                "score": 0.1234,
                "image_path": None
            }
        ]
        
        return JSONResponse(content={
            "status": "success",
            "data": inspections[:limit]
        })
        
    except Exception as e:
        print(f"[DASHBOARD] 검사 내역 조회 실패: {e}")
        raise HTTPException(500, f"검사 내역 조회 실패: {str(e)}")


@router.get("/products/stats")
async def get_product_stats():
    """
    제품별 통계 조회
    """
    try:
        # TODO: 실제 DB에서 데이터 조회
        
        # defect_mapping.json 로드
        mapping_file = project_root / "web" / "defect_mapping.json"
        products = []
        
        if mapping_file.exists():
            import json
            with open(mapping_file, 'r', encoding='utf-8') as f:
                mapping_data = json.load(f)
            
            data_dir = project_root / "data" / "patchCore"
            defect_dir = project_root / "data" / "def_split"
            
            for product_id, product_info in mapping_data.get("products", {}).items():
                # 정상 이미지 수
                normal_count = 0
                prod_dir = data_dir / product_id
                if prod_dir.exists():
                    normal_count = len(list(prod_dir.glob("*.jpg"))) + len(list(prod_dir.glob("*.png")))
                
                # 불량 이미지 수
                defect_count = 0
                if defect_dir.exists():
                    for defect_id in product_info.get("defects", {}).keys():
                        pattern = f"{product_id}_{defect_id}_*"
                        defect_count += len(list(defect_dir.glob(pattern)))
                
                products.append({
                    "name": product_id,
                    "name_ko": product_info.get("name_ko", product_id),
                    "normalImages": normal_count,
                    "defectImages": defect_count,
                    "totalInspections": 0,  # TODO: DB 연동
                    "defectRate": 0.0  # TODO: DB 연동
                })
        
        return JSONResponse(content={
            "status": "success",
            "data": products
        })
        
    except Exception as e:
        print(f"[DASHBOARD] 제품 통계 조회 실패: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"제품 통계 조회 실패: {str(e)}")


@router.get("/chart/inspections")
async def get_inspection_chart_data(period: str = "7d"):
    """
    검사 추이 차트 데이터
    
    Args:
        period: 7d, 30d, 90d
    """
    try:
        # TODO: 실제 DB에서 데이터 조회
        
        days = 7
        if period == "30d":
            days = 30
        elif period == "90d":
            days = 90
        
        # 임시 데이터 생성
        chart_data = []
        for i in range(days):
            date = (datetime.now() - timedelta(days=days-i-1)).strftime("%Y-%m-%d")
            chart_data.append({
                "date": date,
                "inspections": 0,  # TODO: 실제 데이터
                "anomalies": 0
            })
        
        return JSONResponse(content={
            "status": "success",
            "data": chart_data
        })
        
    except Exception as e:
        print(f"[DASHBOARD] 차트 데이터 조회 실패: {e}")
        raise HTTPException(500, f"차트 데이터 조회 실패: {str(e)}")


@router.get("/chart/defect-rate")
async def get_defect_rate_chart_data(period: str = "today"):
    """
    제품별 불량 검출 비율 차트 데이터
    
    Args:
        period: today, week, month
    """
    try:
        # TODO: 실제 DB에서 데이터 조회
        
        # 임시 데이터
        chart_data = [
            {"product": "prod1", "rate": 0.23, "count": 156},
            {"product": "grid", "rate": 0.18, "count": 98},
            {"product": "carpet", "rate": 0.31, "count": 134},
            {"product": "leather", "rate": 0.15, "count": 87}
        ]
        
        return JSONResponse(content={
            "status": "success",
            "data": chart_data
        })
        
    except Exception as e:
        print(f"[DASHBOARD] 차트 데이터 조회 실패: {e}")
        raise HTTPException(500, f"차트 데이터 조회 실패: {str(e)}")
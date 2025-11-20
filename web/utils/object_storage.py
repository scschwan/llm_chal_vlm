"""
Naver Cloud Platform Object Storage 연동 유틸리티
"""

import os
import boto3
from pathlib import Path
from typing import List, Tuple, Optional
from botocore.exceptions import ClientError
import concurrent.futures
from dotenv import load_dotenv

class ObjectStorageManager:
    """Object Storage 관리 클래스"""
    
    def __init__(self):
        # .env 파일 로드
        load_dotenv()
        
        """초기화"""
        self.endpoint_url = "https://kr.object.ncloudstorage.com"
        self.region = "kr-standard"
        self.access_key = os.getenv("NCP_ACCESS_KEY")
        self.secret_key = os.getenv("NCP_SECRET_KEY")
        self.bucket = os.getenv("NCP_BUCKET", "dm-obs")
        
        if not self.access_key or not self.secret_key:
            raise ValueError("NCP_ACCESS_KEY, NCP_SECRET_KEY 환경변수가 설정되지 않았습니다")
        
        self.s3 = boto3.client(
            's3',
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            region_name=self.region
        )
    
    def create_folder(self, folder_path: str) -> bool:
        """
        폴더 생성
        
        Args:
            folder_path: 폴더 경로 (예: 'images/normal/123/')
        
        Returns:
            bool: 성공 여부
        """
        try:
            if not folder_path.endswith('/'):
                folder_path += '/'
            
            self.s3.put_object(Bucket=self.bucket, Key=folder_path)
            return True
        except ClientError as e:
            print(f"폴더 생성 실패: {e}")
            return False
    
    def upload_file(self, local_path: str, s3_key: str) -> bool:
        """
        파일 업로드
        
        Args:
            local_path: 로컬 파일 경로
            s3_key: Object Storage 키
        
        Returns:
            bool: 성공 여부
        """
        try:
            self.s3.upload_file(local_path, self.bucket, s3_key)
            print(f"✅ 업로드 완료: {s3_key}")
            return True
        except ClientError as e:
            print(f"❌ 업로드 실패: {e}")
            return False
    
    def upload_fileobj(self, file_obj, s3_key: str) -> bool:
        """
        파일 객체 업로드 (FastAPI UploadFile 지원)
        
        Args:
            file_obj: 파일 객체
            s3_key: Object Storage 키
        
        Returns:
            bool: 성공 여부
        """
        try:
            self.s3.upload_fileobj(file_obj, self.bucket, s3_key)
            print(f"✅ 업로드 완료: {s3_key}")
            return True
        except ClientError as e:
            print(f"❌ 업로드 실패: {e}")
            return False
    
    def upload_multiple_files(self, file_list: List[str], base_s3_path: str) -> dict:
        """
        여러 파일 병렬 업로드
        
        Args:
            file_list: 로컬 파일 경로 리스트
            base_s3_path: Object Storage 기본 경로
        
        Returns:
            dict: {filename: success}
        """
        results = {}
        
        def upload_single(file_path):
            filename = Path(file_path).name
            s3_key = f"{base_s3_path}/{filename}"
            success = self.upload_file(file_path, s3_key)
            return filename, success
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(upload_single, f) for f in file_list]
            
            for future in concurrent.futures.as_completed(futures):
                filename, success = future.result()
                results[filename] = success
        
        return results
    
    def download_file(self, s3_key: str, local_path: str) -> bool:
        """
        파일 다운로드
        
        Args:
            s3_key: Object Storage 키
            local_path: 로컬 저장 경로
        
        Returns:
            bool: 성공 여부
        """
        try:
            # 로컬 디렉토리 생성
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            
            self.s3.download_file(self.bucket, s3_key, local_path)
            print(f"✅ 다운로드 완료: {local_path}")
            return True
        except ClientError as e:
            print(f"❌ 다운로드 실패: {e}")
            return False
    
    def download_multiple_files(self, s3_keys: List[str], local_base_path: str) -> dict:
        """
        여러 파일 병렬 다운로드
        
        Args:
            s3_keys: Object Storage 키 리스트
            local_base_path: 로컬 저장 기본 경로
        
        Returns:
            dict: {s3_key: success}
        """
        results = {}
        
        def download_single(s3_key):
            filename = Path(s3_key).name
            local_path = Path(local_base_path) / filename
            success = self.download_file(s3_key, str(local_path))
            return s3_key, success
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(download_single, key) for key in s3_keys]
            
            for future in concurrent.futures.as_completed(futures):
                s3_key, success = future.result()
                results[s3_key] = success
        
        return results
    
    def list_objects(self, prefix: str = '') -> List[dict]:
        """
        객체 목록 조회
        
        Args:
            prefix: 경로 접두사
        
        Returns:
            List[dict]: 객체 목록
        """
        objects = []
        max_keys = 1000
        marker = None
        
        while True:
            params = {
                'Bucket': self.bucket,
                'MaxKeys': max_keys
            }
            
            if prefix:
                params['Prefix'] = prefix
            if marker:
                params['Marker'] = marker
            
            response = self.s3.list_objects(**params)
            
            if 'Contents' in response:
                objects.extend(response['Contents'])
            
            if response.get('IsTruncated'):
                marker = response.get('NextMarker')
            else:
                break
        
        return objects
    
    def list_folders_and_files(self, prefix: str = '') -> Tuple[List[str], List[dict]]:
        """
        폴더와 파일 구분 조회
        
        Args:
            prefix: 경로 접두사
        
        Returns:
            Tuple[List[str], List[dict]]: (folders, files)
        """
        delimiter = '/'
        folders = []
        files = []
        max_keys = 1000
        marker = None
        
        while True:
            params = {
                'Bucket': self.bucket,
                'Delimiter': delimiter,
                'MaxKeys': max_keys
            }
            
            if prefix:
                params['Prefix'] = prefix
            if marker:
                params['Marker'] = marker
            
            response = self.s3.list_objects(**params)
            
            # 폴더 목록
            if 'CommonPrefixes' in response:
                for folder in response['CommonPrefixes']:
                    folders.append(folder['Prefix'])
            
            # 파일 목록
            if 'Contents' in response:
                for obj in response['Contents']:
                    if not obj['Key'].endswith('/'):
                        files.append(obj)
            
            if response.get('IsTruncated'):
                marker = response.get('NextMarker')
            else:
                break
        
        return folders, files
    
    def delete_file(self, s3_key: str) -> bool:
        """
        파일 삭제
        
        Args:
            s3_key: Object Storage 키
        
        Returns:
            bool: 성공 여부
        """
        try:
            self.s3.delete_object(Bucket=self.bucket, Key=s3_key)
            print(f"✅ 삭제 완료: {s3_key}")
            return True
        except ClientError as e:
            print(f"❌ 삭제 실패: {e}")
            return False
    
    def delete_folder(self, folder_path: str) -> int:
        """
        폴더 전체 삭제 (재귀)
        
        Args:
            folder_path: 삭제할 폴더 경로
        
        Returns:
            int: 삭제된 파일 수
        """
        objects = self.list_objects(folder_path)
        
        deleted_count = 0
        for obj in objects:
            if self.delete_file(obj['Key']):
                deleted_count += 1
        
        return deleted_count
    
    def get_url(self, s3_key: str) -> str:
        """
        공개 URL 생성
        
        Args:
            s3_key: Object Storage 키
        
        Returns:
            str: 공개 URL
        """
        return f"{self.endpoint_url}/{self.bucket}/{s3_key}"
    
    def get_file_info(self, s3_key: str) -> Optional[dict]:
        """
        파일 정보 조회
        
        Args:
            s3_key: Object Storage 키
        
        Returns:
            Optional[dict]: 파일 정보
        """
        try:
            response = self.s3.head_object(Bucket=self.bucket, Key=s3_key)
            return {
                'size': response['ContentLength'],
                'last_modified': response['LastModified'],
                'content_type': response.get('ContentType', 'unknown')
            }
        except ClientError as e:
            print(f"파일 정보 조회 실패: {e}")
            return None


# 싱글톤 인스턴스
_obs_manager = None

def get_obs_manager() -> ObjectStorageManager:
    """
    ObjectStorageManager 싱글톤 인스턴스 반환
    """
    global _obs_manager
    if _obs_manager is None:
        _obs_manager = ObjectStorageManager()
    return _obs_manager
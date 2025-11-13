# file: ncp_obj_public_get_test.py
import os
import sys
import boto3
from botocore.exceptions import ClientError
from PIL import Image

ENDPOINT = "https://kr.object.ncloudstorage.com"   # 공식 엔드포인트
REGION   = "kr-standard"

ACCESS_KEY = os.environ.get("NCP_ACCESS_KEY")
SECRET_KEY = os.environ.get("NCP_SECRET_KEY")
BUCKET     = os.environ.get("NCP_BUCKET")

# 가져올 원본 객체
TEST_KEY = "def_undefined/carpet_hole_016.png"

# 저장할 로컬 파일 경로
DOWNLOAD_PATH = "/tmp/carpet_hole_016.png"

if not all([ACCESS_KEY, SECRET_KEY, BUCKET]):
    print("Please set NCP_ACCESS_KEY, NCP_SECRET_KEY, NCP_BUCKET envs.")
    sys.exit(1)

# S3 클라이언트 초기화
s3 = boto3.client(
    "s3",
    endpoint_url=ENDPOINT,
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    region_name=REGION,
)

def download_object(bucket, key, path):
    try:
        s3.download_file(bucket, key, path)
        return True
    except ClientError as e:
        print(f"Download error: {e}")
        return False

def main():
    print(f"[1] Downloading s3://{BUCKET}/{TEST_KEY} → {DOWNLOAD_PATH}")
    ok = download_object(BUCKET, TEST_KEY, DOWNLOAD_PATH)
    if not ok:
        print("Download failed.")
        sys.exit(2)
    print("  -> Download OK.")

    print(f"[2] Loading image: {DOWNLOAD_PATH}")
    try:
        img = Image.open(DOWNLOAD_PATH)
        img.show()  # OS 기본 이미지 뷰어로 표시
        print("  -> Image opened successfully.")
    except Exception as e:
        print(f"Image open error: {e}")

if __name__ == "__main__":
    main()

import boto3

service_name = 's3'
endpoint_url = 'https://kr.object.ncloudstorage.com'
region_name = 'kr-standard'
access_key = 'ACCESS_KEY_ID'
secret_key = 'SECRET_KEY'

if __name__ == "__main__":
    s3 = boto3.client(service_name, endpoint_url=endpoint_url, aws_access_key_id=access_key,
                      aws_secret_access_key=secret_key)
    bucket_name = 'sample-bucket'
    object_name = 'sample-large-object'
    
    local_file = '/tmp/sample.file'

    # initialize and get upload ID
    create_multipart_upload_response = s3.create_multipart_upload(Bucket=bucket_name, Key=object_name)
    upload_id = create_multipart_upload_response['UploadId']

    part_size = 10 * 1024 * 1024
    parts = []

    # upload parts
    with open(local_file, 'rb') as f:
        part_number = 1
        while True:
            data = f.read(part_size)
            if not len(data):
                break
            upload_part_response = s3.upload_part(Bucket=bucket_name, Key=object_name, PartNumber=part_number, UploadId=upload_id, Body=data)
            parts.append({
                'PartNumber': part_number,
                'ETag': upload_part_response['ETag']
            })
            part_number += 1

    multipart_upload = {'Parts': parts}

    # abort multipart upload
    # s3.abort_multipart_upload(Bucket=bucket_name, Key=object_name, UploadId=upload_id)

    # complete multipart upload
    s3.complete_multipart_upload(Bucket=bucket_name, Key=object_name, UploadId=upload_id, MultipartUpload=multipart_upload)


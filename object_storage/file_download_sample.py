import boto3

service_name = 's3'
endpoint_url = 'https://kr.object.ncloudstorage.com'
region_name = 'kr-standard'
access_key = 'ACCESS_KEY'
secret_key = 'SECRET_KEY'

if __name__ == "__main__":
    s3 = boto3.client(service_name, endpoint_url=endpoint_url, aws_access_key_id=access_key,
                      aws_secret_access_key=secret_key)
    bucket_name = 'sample-bucket'

    object_name = 'sample-object'
    local_file_path = '/tmp/test.txt'

    s3.download_file(bucket_name, object_name, local_file_path)


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

    # list all in the bucket
    max_keys = 300
    response = s3.list_objects(Bucket=bucket_name, MaxKeys=max_keys)

    print('list all in the bucket')

    while True:
        print('IsTruncated=%r' % response.get('IsTruncated'))
        print('Marker=%s' % response.get('Marker'))
        print('NextMarker=%s' % response.get('NextMarker'))

        print('Object List')
        for content in response.get('Contents'):
            print(' Name=%s, Size=%d, Owner=%s' % \
                  (content.get('Key'), content.get('Size'), content.get('Owner').get('ID')))

        if response.get('IsTruncated'):
            response = s3.list_objects(Bucket=bucket_name, MaxKeys=max_keys,
                                       Marker=response.get('NextMarker'))
        else:
            break

    # top level folders and files in the bucket
    delimiter = '/'
    max_keys = 300

    response = s3.list_objects(Bucket=bucket_name, Delimiter=delimiter, MaxKeys=max_keys)

    print('top level folders and files in the bucket')

    while True:
        print('IsTruncated=%r' % response.get('IsTruncated'))
        print('Marker=%s' % response.get('Marker'))
        print('NextMarker=%s' % response.get('NextMarker'))

        print('Folder List')
        for folder in response.get('CommonPrefixes'):
            print(' Name=%s' % folder.get('Prefix'))

        print('File List')
        for content in response.get('Contents'):
            print(' Name=%s, Size=%d, Owner=%s' % \
                  (content.get('Key'), content.get('Size'), content.get('Owner').get('ID')))

        if response.get('IsTruncated'):
            response = s3.list_objects(Bucket=bucket_name, Delimiter=delimiter, MaxKeys=max_keys,
                                       Marker=response.get('NextMarker'))
        else:
            break


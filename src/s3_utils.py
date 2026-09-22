import boto3

from config import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_DEFAULT_REGION,
    S3_BUCKET_NAME
)


def get_s3_client():

    return boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_DEFAULT_REGION
    )


def upload_file_to_s3(
    local_file_path,
    s3_key
):

    s3 = get_s3_client()

    s3.upload_file(
        local_file_path,
        S3_BUCKET_NAME,
        s3_key
    )

    print(
        f"Uploaded -> s3://{S3_BUCKET_NAME}/{s3_key}"
    )
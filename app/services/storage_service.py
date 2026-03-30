import boto3
import os
import logging
from botocore.exceptions import ClientError

logger = logging.getLogger("ZenithRAG.S3Storage")

class S3Storage:
    def __init__(self):
        """
        Initializes the ZenithRAG Cloud Sync Service.
        Specifically tuned for the eu-north-1 (Stockholm) region.
        """
        # Retrieve keys directly from the environment loaded by bootstrap_environment()
        access_key = os.getenv('AWS_ACCESS_KEY_ID')
        secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        self.region = os.getenv('AWS_REGION', 'eu-north-1')
        self.bucket = os.getenv('S3_BUCKET_NAME')

        # Robust Check: Prevents 'NoneType' crashes before initializing boto3
        if not all([access_key, secret_key, self.bucket]):
            logger.error("--- S3 Service: Missing Configuration. Check ZenithRAG .env ---")
            self.s3 = None
            return

        try:
            self.s3 = boto3.client(
                's3',
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                region_name=self.region
            )
            logger.info(f"--- S3 Storage Linked: {self.bucket} ({self.region}) ---")
        except Exception as e:
            logger.error(f"S3 Connection failed: {e}")
            self.s3 = None

    def upload_file(self, file_obj, filename):
        """
        Syncs a file to the AWS Cloud Bucket.
        Returns the public URL if successful.
        """
        if not self.s3:
            logger.error("Upload aborted: S3 Client is offline.")
            return False
            
        try:
            # Ensure the file pointer is at the start (crucial for Flask file objects)
            file_obj.seek(0)
            
            # Perform the upload
            self.s3.upload_fileobj(file_obj, self.bucket, filename)
            
            # Construct the Stockholm S3 Public URL
            file_url = f"https://{self.bucket}.s3.{self.region}.amazonaws.com/{filename}"
            logger.info(f"Cloud Sync Success: {file_url}")
            return file_url
        except Exception as e:
            logger.error(f"Upload failed for {filename}: {e}")
            return False

    def check_file_exists(self, filename):
        """Utility for ZenithRAG verification."""
        if not self.s3: return False
        try:
            self.s3.head_object(Bucket=self.bucket, Key=filename)
            return True
        except ClientError:
            return False
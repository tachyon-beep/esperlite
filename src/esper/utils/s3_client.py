"""
S3 Client utilities for interacting with MinIO/S3 storage.

This module provides functions for configuring S3 clients and managing buckets
for the Esper platform's artifact storage needs.
"""

import logging
import os

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


def get_s3_client():
    """
    Configures and returns a boto3 client for MinIO/S3.

    Returns:
        boto3.client: Configured S3 client

    Raises:
        ValueError: If required environment variables are missing
    """
    endpoint_url = os.getenv("S3_ENDPOINT_URL")
    access_key = os.getenv("MINIO_ROOT_USER")
    secret_key = os.getenv("MINIO_ROOT_PASSWORD")

    if not all([endpoint_url, access_key, secret_key]):
        raise ValueError(
            "Missing required S3 environment variables: "
            "S3_ENDPOINT_URL, MINIO_ROOT_USER, MINIO_ROOT_PASSWORD"
        )

    client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version="s3v4"),
    )

    logger.info("S3 client configured for endpoint: %s", endpoint_url)
    return client


def ensure_bucket_exists(client, bucket_name: str) -> None:
    """
    Creates an S3 bucket if it doesn't already exist.

    Args:
        client: Configured S3 client
        bucket_name: Name of the bucket to create

    Raises:
        ClientError: If bucket creation fails for reasons other than already exists
    """
    try:
        client.head_bucket(Bucket=bucket_name)
        logger.info("Bucket '%s' already exists", bucket_name)
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code == "404":
            try:
                client.create_bucket(Bucket=bucket_name)
                logger.info("Bucket '%s' created successfully", bucket_name)
            except ClientError as create_error:
                logger.error(
                    "Failed to create bucket '%s': %s", bucket_name, create_error
                )
                raise
        else:
            logger.error("Error checking bucket '%s': %s", bucket_name, e)
            raise


def upload_file(client, file_path: str, bucket_name: str, object_key: str) -> str:
    """
    Uploads a file to S3 bucket.

    Args:
        client: Configured S3 client
        file_path: Local path to the file to upload
        bucket_name: Name of the S3 bucket
        object_key: S3 object key (path within bucket)

    Returns:
        str: S3 URI of the uploaded file

    Raises:
        ClientError: If upload fails
    """
    try:
        client.upload_file(file_path, bucket_name, object_key)
        s3_uri = f"s3://{bucket_name}/{object_key}"
        logger.info("File uploaded successfully to %s", s3_uri)
        return s3_uri
    except ClientError as e:
        logger.error(
            "Failed to upload file %s to %s/%s: %s",
            file_path,
            bucket_name,
            object_key,
            e,
        )
        raise


def upload_bytes(client, data: bytes, bucket_name: str, object_key: str) -> str:
    """
    Uploads bytes data to S3 bucket.

    Args:
        client: Configured S3 client
        data: Bytes data to upload
        bucket_name: Name of the S3 bucket
        object_key: S3 object key (path within bucket)

    Returns:
        str: S3 URI of the uploaded file

    Raises:
        ClientError: If upload fails
    """
    try:
        client.put_object(Bucket=bucket_name, Key=object_key, Body=data)
        s3_uri = f"s3://{bucket_name}/{object_key}"
        logger.info("Data uploaded successfully to %s", s3_uri)
        return s3_uri
    except ClientError as e:
        logger.error("Failed to upload data to %s/%s: %s", bucket_name, object_key, e)
        raise


def download_file(client, bucket_name: str, object_key: str, local_path: str) -> None:
    """
    Downloads a file from S3 bucket.

    Args:
        client: Configured S3 client
        bucket_name: Name of the S3 bucket
        object_key: S3 object key (path within bucket)
        local_path: Local path to save the downloaded file

    Raises:
        ClientError: If download fails
    """
    try:
        client.download_file(bucket_name, object_key, local_path)
        logger.info(
            "File downloaded successfully from s3://%s/%s to %s",
            bucket_name,
            object_key,
            local_path,
        )
    except ClientError as e:
        logger.error(
            "Failed to download file from %s/%s to %s: %s",
            bucket_name,
            object_key,
            local_path,
            e,
        )
        raise


def download_bytes(client, bucket_name: str, object_key: str) -> bytes:
    """
    Downloads bytes data from S3 bucket.

    Args:
        client: Configured S3 client
        bucket_name: Name of the S3 bucket
        object_key: S3 object key (path within bucket)

    Returns:
        bytes: Downloaded data

    Raises:
        ClientError: If download fails
    """
    try:
        response = client.get_object(Bucket=bucket_name, Key=object_key)
        data = response["Body"].read()
        logger.info(
            "Data downloaded successfully from s3://%s/%s", bucket_name, object_key
        )
        return data
    except ClientError as e:
        logger.error(
            "Failed to download data from %s/%s: %s", bucket_name, object_key, e
        )
        raise


def parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    """
    Parses an S3 URI into bucket and key components.

    Args:
        s3_uri: S3 URI in format s3://bucket/key

    Returns:
        tuple: (bucket_name, object_key)

    Raises:
        ValueError: If URI format is invalid
    """
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI format: {s3_uri}")

    # Remove s3:// prefix and split on first /
    uri_parts = s3_uri[5:].split("/", 1)
    if len(uri_parts) != 2:
        raise ValueError(f"Invalid S3 URI format: {s3_uri}")

    bucket_name, object_key = uri_parts
    return bucket_name, object_key

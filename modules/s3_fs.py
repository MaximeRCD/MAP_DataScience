"""
Module: s3_fs

This module provides functions to import files and buckets from SSP (Secure and Scalable Platform)
cloud storage to the local system using the S3 protocol.

Usage:
    The module can be used to import files or entire buckets from SSP cloud storage to the local
    system.
"""

import s3fs
from constants import (
    S3_DATA_BUCKET_NAME,
    S3_ENDPOINT_URL,
    S3_JSON_BUCKET_NAME,
    S3_PRETRAINED_MODEL_NAME,
    S3_USER_BUCKET,
    PRETRAINED_MODEL_PATH,
    DATA_ROOT_DIR,
)


def import_bucket_from_ssp_cloud(
    fs_obj, source_bucket_name, destination_folder, recursive=True
):
    """
    Import entire bucket from SSP cloud storage to local system.

    Args:
        fs_obj (s3fs.S3FileSystem): S3FileSystem object for accessing cloud storage.
        source_bucket_name (str): Name of the source bucket in the cloud storage.
        destination_folder (str): Path of the destination folder in the local system.
        recursive (bool, optional): Flag to indicate whether to import recursively or not.
                                    Defaults to True.

    Raises:
        PermissionError: If user does not have the right permission to access the bucket.

    Returns:
        None
    """
    try:
        fs_obj.get(source_bucket_name, destination_folder, recursive=recursive)
        print(
            f"The bucket {source_bucket_name} has been downloaded to {destination_folder}"
        )
    except PermissionError:
        print(
            f"You tried to import {source_bucket_name} but you do not have the right permission"
            " to do so !"
        )


def import_file_from_ssp_cloud(fs_obj, source_file_name, destination_file_path):
    """
    Import a file from SSP cloud storage to local system.

    Args:
        fs_obj (s3fs.S3FileSystem): S3FileSystem object for accessing cloud storage.
        source_file_name (str): Name of the source file in the cloud storage.
        destination_file_path (str): Path of the destination file in the local system.

    Raises:
        PermissionError: If user does not have the right permission to access the file.
        FileNotFoundError: If the specified file does not exist in the cloud storage.

    Returns:
        None
    """
    try:
        fs_obj.get(source_file_name, destination_file_path)
        print(
            f"The file {source_file_name} has been downloaded to {destination_file_path}"
        )
    except PermissionError:
        print(
            f"You tried to import {source_file_name} but you do not have the right permission"
            " to do so !"
        )
    except FileNotFoundError:
        print(f"The file {source_file_name} does not exist")


if __name__ == "__main__":
    # Create filesystem object
    fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": S3_ENDPOINT_URL})

    # Example usage:
    import_bucket_from_ssp_cloud(
        fs, "/".join([S3_USER_BUCKET, S3_DATA_BUCKET_NAME]), DATA_ROOT_DIR
    )
    import_bucket_from_ssp_cloud(
        fs,
        "/".join([S3_USER_BUCKET, S3_JSON_BUCKET_NAME]),
        "/".join([".", S3_JSON_BUCKET_NAME]),
    )
    import_file_from_ssp_cloud(
        fs,
        "/".join([S3_USER_BUCKET, S3_PRETRAINED_MODEL_NAME]),
        "/".join([".", PRETRAINED_MODEL_PATH]),
    )
    import_file_from_ssp_cloud(
        fs,
        "/".join([S3_USER_BUCKET, "failing_test.txt"]),
        "/".join([".", "failing_test.txt"]),
    )

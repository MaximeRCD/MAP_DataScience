"""
Module: s3_fs_manager

This module, `s3_fs_manager`, offers a structured and object-oriented approach to managing
interactions with SSP (Secure and Scalable Platform) cloud storage using the S3 protocol. It
encapsulates functionality to import files and entire buckets from the SSP cloud storage to the
local system. Through the `S3FileManager` class, users can perform these operations more
efficiently and with greater ease, leveraging the `s3fs` library for underlying S3 filesystem
interactions.

Features include:
- Importing entire buckets from SSP cloud storage to the local filesystem, with optional recursive
  downloading.
- Importing specific files from SSP cloud storage to the local filesystem, handling permissions
  and existence checks gracefully.

Usage:
    Instantiate the `S3FileManager` class and use its methods `import_bucket_from_ssp_cloud` and
    `import_file_from_ssp_cloud` to import buckets and files, respectively. The module requires
    setup with appropriate S3 access credentials and endpoint configurations.

Dependencies:
    - s3fs: For handling S3 filesystem operations.
    - constants: A module containing configuration constants like S3 bucket names, paths, and
      endpoint URLs.

Example:
    >>> manager = S3FileManager()
    >>> manager.import_bucket_from_ssp_cloud('source_bucket_name', 'destination_folder_path')
    >>> manager.import_file_from_ssp_cloud('source_file_name', 'destination_file_path')
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


class S3FileManager:
    """
    Class to manage import operations from SSP (Secure and Scalable Platform)
    cloud storage to the local system using the S3 protocol.
    """

    def __init__(self, endpoint_url=S3_ENDPOINT_URL):
        """
        Initializes the S3FileManager with an S3FileSystem object.

        Args:
            endpoint_url (str): The endpoint URL for the S3 connection.
        """
        self.fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": endpoint_url}, anon=True)

    def import_bucket_from_ssp_cloud(
        self, source_bucket_name, destination_folder, recursive=True
    ):
        """
        Import entire bucket from SSP cloud storage to local system.

        Args:
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
            self.fs.get(source_bucket_name, destination_folder, recursive=recursive)
            print(
                f"The bucket {source_bucket_name} has been downloaded to {destination_folder}"
            )
        except PermissionError:
            print(
                f"You tried to import {source_bucket_name} but you do not have the right permission"
                " to do so!"
            )

    def import_file_from_ssp_cloud(self, source_file_name, destination_file_path):
        """
        Import a file from SSP cloud storage to local system.

        Args:
            source_file_name (str): Name of the source file in the cloud storage.
            destination_file_path (str): Path of the destination file in the local system.

        Raises:
            PermissionError: If user does not have the right permission to access the file.
            FileNotFoundError: If the specified file does not exist in the cloud storage.

        Returns:
            None
        """
        try:
            self.fs.get(source_file_name, destination_file_path)
            print(
                f"The file {source_file_name} has been downloaded to {destination_file_path}"
            )
        except PermissionError:
            print(
                f"You tried to import {source_file_name} but you do not have the right permission"
                " to do so!"
            )
        except FileNotFoundError:
            print(f"The file {source_file_name} does not exist")


if __name__ == "__main__":
    # Initialize the S3FileManager
    manager = S3FileManager()

    # Example usage:
    manager.import_bucket_from_ssp_cloud(
        "/".join([S3_USER_BUCKET, S3_DATA_BUCKET_NAME]), DATA_ROOT_DIR
    )
    manager.import_bucket_from_ssp_cloud(
        "/".join([S3_USER_BUCKET, S3_JSON_BUCKET_NAME]),
        "/".join([".", S3_JSON_BUCKET_NAME]),
    )
    manager.import_file_from_ssp_cloud(
        "/".join([S3_USER_BUCKET, S3_PRETRAINED_MODEL_NAME]),
        "/".join([".", PRETRAINED_MODEL_PATH]),
    )
    manager.import_file_from_ssp_cloud(
        "/".join([S3_USER_BUCKET, "failing_test.txt"]),
        "/".join([".", "failing_test.txt"]),
    )

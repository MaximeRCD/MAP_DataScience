

import os
import s3fs


def import_bucket_from_ssp_cloud(fs_obj, source_bucket_name, destination_folder, recursive=True):
    try:
        fs_obj.get(source_bucket_name, destination_folder, recursive=recursive)
        print(f"The bucket {source_bucket_name}, has been downloaded to {destination_folder}")
    except PermissionError:
        print(f"You tried to import {source_bucket_name} but you do not have the  right permission to do so !")


def import_file_from_ssp_cloud(fs_obj, source_file_name, destination_file_path):
    try:
        fs_obj.get(source_file_name, destination_file_path)
        print(f"The File {source_file_name}, has been downloaded to {destination_file_path}")
    except PermissionError:
        print(f"You tried to import {source_file_name} but you do not have the  right permission to do so !")
    except FileNotFoundError:
        print(f"The file {source_file_name} does not exist")


if __name__ == "__main__":
    # Create filesystem object
    S3_ENDPOINT_URL = "https://" + os.environ["AWS_S3_ENDPOINT"]
    fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': S3_ENDPOINT_URL})
    # import_bucket_from_ssp_cloud(fs, "maximerichaudeau1/data", './data')
    # import_bucket_from_ssp_cloud(fs, "maximerichaudeau1/json", './json')
    import_file_from_ssp_cloud(fs,
     "maximerichaudeau1/cross_entropy_weighted10_batch64_32_16.pth",
     './cross_entropy_weighted10_batch64_32_16.pth')

    import_file_from_ssp_cloud(fs,
     "maximerichaudeau1/test.txt",
     './test.txt')


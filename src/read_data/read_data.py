from kfp.v2.dsl import component, Output, Dataset
from minio import Minio
import pandas as pd


@component
def get_data(
    minio_host: str,
    access_key: str,
    secret_key: str,
    bucket_name: str,
    file_name: str,
    data_output: Output[Dataset],
):
    # Initialize Minio client
    client = Minio(
        endpoint=minio_host, access_key=access_key, secret_key=secret_key, secure=False
    )
    # Download the file from Minio
    client.fget_object(bucket_name, file_name, file_name)
    # Load the Parquet file into a DataFrame
    df = pd.read_parquet(file_name)
    # Save the DataFrame as a pickle file in the output directory
    df.to_pickle(data_output.path)
